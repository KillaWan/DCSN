"""Block mean predictor: shared temporal MLP, channel embedding and two skips.

Inputs are B x C x L: the transformed history and its block-constant projection.
The output is B x C x H, constant within each forecast block. Attribute names
remain compatible with archived checkpoints; the computation order is retained.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._initialization import consume_linear, consume_tcn


def block_basis(length, block_len):
    """Orthonormal block-constant basis, including a shorter final block."""
    columns = []
    for start in range(0, length, block_len):
        size = min(block_len, length - start)
        constant = torch.zeros(length)
        constant[start:start + size] = 1 / math.sqrt(size)
        columns.append(constant)
    return torch.stack(columns, dim=-1)


class _ChannelConcat(nn.Sequential):
    def forward(self, x):
        return super().forward(x) + 0.1 * self.channel_embedding


class _FFN(nn.Module):
    def __init__(self, width, ff, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width, ff, bias=False), nn.LeakyReLU(),
            nn.Linear(ff, width, bias=False), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class BlockMeanPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, channels, block_len, width, ff,
                 layers, dropout, dataset):
        super().__init__()
        # The two projections start identical, then learn independently.
        self.m_project = nn.Sequential(
            nn.Linear(seq_len, width, bias=False), nn.Dropout(dropout))
        self.mean_proj = copy.deepcopy(self.m_project)
        consume_linear(width, width)
        self.m_concat = _ChannelConcat(
            nn.Linear(2 * width, width), nn.Dropout(dropout))
        consume_linear(2 * width, width)
        consume_tcn(channels, dataset)
        self.mean_ffn = nn.Sequential(
            *[_FFN(width, ff, dropout) for _ in range(layers)])
        time_readout = nn.Linear(width, pred_len, bias=False)
        consume_linear(width, pred_len, bias=False)
        for i in range(4):
            consume_linear(seq_len // (2 ** i), pred_len)

        self.register_buffer("output_basis", block_basis(pred_len, block_len))
        # Initialize the block readout by projecting the original time readout.
        with torch.random.fork_rng(devices=[]):
            self.coordinate_pred = nn.Linear(
                width, self.output_basis.shape[1], bias=False)
        with torch.no_grad():
            self.coordinate_pred.weight.copy_(self.output_basis.T @ time_readout.weight)
        self.register_buffer("history_basis", block_basis(seq_len, block_len))
        self.origin_gain = nn.Parameter(torch.ones(1, channels, 1))
        with torch.random.fork_rng(devices=[]):
            self.linear_skip = nn.Linear(
                self.history_basis.shape[1], self.output_basis.shape[1], bias=False)
        nn.init.zeros_(self.linear_skip.weight)
        with torch.random.fork_rng(devices=[]):
            self.innovation_skip = nn.Linear(seq_len, self.output_basis.shape[1], bias=False)
        nn.init.zeros_(self.innovation_skip.weight)
        self.m_concat.channel_embedding = nn.Parameter(torch.zeros(1, channels, width))
        self._last_origin = None

    def forward(self, mean, raw, gain):
        """Predict block means from projected history, full history and gains."""
        origin = mean.mean(-1, keepdim=True)
        self._last_origin = origin.detach()
        # Separate reductions preserve the original backward accumulation.
        reference = mean.mean(-1, keepdim=True)
        features = self.m_concat(torch.cat([
            self.m_project(raw - reference), self.mean_proj(mean - reference),
        ], dim=-1))
        coefficients = self.coordinate_pred(self.mean_ffn(features))
        coefficients = coefficients + reference * self.output_basis.sum(0)
        original = F.linear(gain * coefficients, self.output_basis)
        features = (mean - origin) @ self.history_basis
        correction = self.linear_skip(features)
        correction = correction + self.innovation_skip(raw - mean)
        # Keep the three decodes/additions separate for numerical parity.
        result = original + F.linear(gain * correction, self.output_basis)
        weights = self.output_basis.sum(0).square() / self.output_basis.shape[0]
        removed_gain = (gain * weights).sum(-1, keepdim=True)
        result = result + F.linear(
            (self.origin_gain - removed_gain) * origin * self.output_basis.sum(0),
            self.output_basis)
        return result
