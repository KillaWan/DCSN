"""History-fitted shape map, reversible coordinates and block-mean supervision."""

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from .reference import BlockMeanPredictor, block_basis


class ShapeNormalizer(nn.Module):
    """History-only shape map and shared block mean predictor.

    forward/restore/reference_loss use BTC; project/coordinates use BCT.
    reference_prediction and canonical_reference_prediction are live BCT caches
    from the latest forward, not persistent state. The map cache is detached;
    applying that fixed map remains differentiable with respect to its input.

    Construct on CPU before the backbone, then move with .to(device). The
    initialization compatibility draws are isolated in _initialization.py.
    """

    def __init__(self, seq_len, pred_len, channels, block_len, width, ff,
                 layers, dropout, *, dataset=None):
        super().__init__()
        dimensions = (seq_len, pred_len, channels, block_len, width, ff)
        if any(not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("Sequence lengths, channels and widths must be positive integers")
        if not isinstance(layers, int) or layers < 0 or not 0 <= dropout <= 1:
            raise ValueError("layers must be nonnegative and dropout must be in [0, 1]")
        if not 3 <= block_len <= min(seq_len, pred_len):
            raise ValueError("block_len must be at least 3 and fit history and forecast")
        if seq_len < 2 * block_len:
            raise ValueError("History must contain at least one complete block pair")
        if torch.empty(0).device.type != "cpu":
            raise ValueError("Construct on CPU for legacy RNG compatibility, then use .to(device)")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.block_len = block_len
        self.support = 2 * block_len
        self.mlp = BlockMeanPredictor(seq_len, pred_len, channels, block_len,
                                     width, ff, layers, dropout, dataset)
        self.register_buffer("input_basis", block_basis(seq_len, block_len))
        self.feature_gain_raw = nn.Parameter(torch.full(
            (1, channels, self.mlp.output_basis.shape[1]), math.atanh(0.5)))
        self._analysis = None
        self.reference_prediction = None
        self.canonical_reference_prediction = None

    def _check_btc(self, x, length):
        if x.ndim != 3 or x.shape[1:] != (length, self.channels):
            raise ValueError(f"Expected BTC tensor with T={length}, C={self.channels}")

    def _contrast_source(self, x):
        # Coefficient = (first mean - second mean) * sqrt(block_len / 2).
        source = torch.ones(self.support, device=x.device, dtype=x.dtype) / math.sqrt(self.support)
        source[self.block_len:] *= -1
        return source

    @torch.no_grad()
    def _estimate_map(self, x):
        """Fit one map per sample using only its lookback window (BCT).

        With N complete block pairs, C variables and w points per block:
        coefficients: B x N x C; residual/dual: B x N x (2*w*C).
        The dual solve is N x N; thin QR limits the SVD to a small matrix.
        Cached factors encode A = coefficients.T @ dual, with ||A||_2 <= 1.
        """
        stop = x.shape[-1] // self.support * self.support
        blocks = x.detach().float()[..., :stop].reshape(*x.shape[:-1], -1, self.support)
        blocks = blocks.permute(0, 2, 1, 3)
        source = self._contrast_source(x)
        coefficients = blocks @ source
        parts = blocks.reshape(*blocks.shape[:-1], 2, self.block_len)
        residual = (parts - parts.mean(-1, keepdim=True)).flatten(-3)
        gram = coefficients @ coefficients.transpose(-1, -2)
        power = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-6)
        ridge = 0.1 * power[:, None, None] * torch.eye(gram.shape[-1], device=x.device)
        dual = torch.linalg.solve(gram + ridge, residual)
        parts = dual.reshape(*dual.shape[:2], self.channels, 2, self.block_len)
        dual = (parts - parts.mean(-1, keepdim=True)).flatten(-3)
        left = torch.linalg.qr(coefficients.transpose(-1, -2), mode="r").R
        right = torch.linalg.qr(dual.transpose(-1, -2), mode="r").R
        core = left @ right.transpose(-1, -2)
        norm = torch.linalg.svdvals(core)[..., 0]
        norm = norm * (1 + 16 * torch.finfo(norm.dtype).eps)
        divisor = norm.clamp_min(1)
        dual = dual / divisor[:, None, None]
        return torch.cat((coefficients, dual), dim=-1).to(x.dtype)

    def project(self, x):
        """Orthogonal block-constant projection of BCT history or forecast."""
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError("project expects BCT with the configured channels")
        if x.shape[-1] == self.seq_len:
            basis = self.input_basis
        elif x.shape[-1] == self.pred_len:
            basis = self.mlp.output_basis
        else:
            raise ValueError("Projection length does not match history or forecast")
        return (x @ basis) @ basis.T

    def coordinates(self, x, inverse=False):
        """Apply the cached I-B (or I+B inverse) to BCT; leave the tail intact."""
        if (x.ndim != 3 or self._analysis is None
                or self._analysis.shape[0] != x.shape[0] or x.shape[1] != self.channels):
            raise RuntimeError("Coordinate restoration requires the matching historical batch")
        coefficients = self._analysis[..., :self.channels]
        dual = self._analysis[..., self.channels:]
        stop = x.shape[-1] // self.support * self.support
        blocks = x[..., :stop].reshape(*x.shape[:-1], -1, self.support).permute(0, 2, 1, 3)
        shape = ((blocks @ self._contrast_source(x)) @ coefficients.transpose(-1, -2)) @ dual
        transformed = blocks + (1 if inverse else -1) * shape.reshape_as(blocks)
        transformed = transformed.permute(0, 2, 1, 3).reshape(*x.shape[:-1], stop)
        return torch.cat((transformed, x[..., stop:]), dim=-1)

    def forward(self, x):
        """Cache this history's map/reference and return the backbone input BTC."""
        self._check_btc(x, self.seq_len)
        x = x.transpose(1, 2)
        self._analysis = self._estimate_map(x)
        canonical = self.coordinates(x)
        mean = self.project(canonical)
        residual = canonical - mean
        gain = 2 * torch.tanh(self.feature_gain_raw)
        self.canonical_reference_prediction = self.mlp(mean, canonical, gain)
        self.reference_prediction = self.coordinates(
            self.canonical_reference_prediction, inverse=True)
        return residual.transpose(1, 2)

    def restore(self, output):
        """Inverse-map the unrestricted backbone output, then add the reference."""
        self._check_btc(output, self.pred_len)
        if self.reference_prediction is None:
            raise RuntimeError("Call forward before restore")
        restored = self.coordinates(output.transpose(1, 2), inverse=True).transpose(1, 2)
        return restored + self.reference_prediction.transpose(1, 2)

    def reference_loss(self, target):
        """Adaptive pseudo-Huber plus origin penalty in train; raw refMSE in eval."""
        self._check_btc(target, self.pred_len)
        if self.reference_prediction is None:
            raise RuntimeError("Call forward before reference_loss")
        prediction = self.coordinates(self.reference_prediction)
        truth = self.project(self.coordinates(target.transpose(1, 2)))
        if not self.training or not torch.is_grad_enabled():
            # The old state-loss path inverse-mapped the target reference first.
            truth = self.coordinates(self.coordinates(truth, inverse=True))
            return F.mse_loss(prediction, truth)
        error = prediction - truth
        samples = error.detach().abs().transpose(0, 1).flatten(1)
        delta = (2.0 * samples.median(dim=-1).values).clamp_min(1e-6)[None, :, None]
        loss = (2 * error.square() / ((1 + (error / delta).square()).sqrt() + 1)).mean()
        basis = self.mlp.output_basis
        weights = basis.sum(0).square() / basis.shape[0]
        gain = 2 * self.feature_gain_raw.tanh()
        shared = (gain * weights).sum(-1, keepdim=True)
        difference = (self.mlp.origin_gain - shared) * self.mlp._last_origin
        return loss + 0.25 * difference.square().mean()

    def load_legacy_state(self, state_dict: Mapping):
        """Load APN-only or full legacy nm.apn_module.* state dictionaries.

        Unrelated backbone keys and the two diagnostic-only APN buffers are
        ignored. Missing/unknown active keys and shape mismatches remain errors.
        No checkpoint deserialization or legacy imports are performed here.
        """
        prefixes = ("module.nm.apn_module.", "nm.apn_module.", "apn_module.")
        prefix = next((p for p in prefixes if any(k.startswith(p) for k in state_dict)), "")
        ignored = {"contrast_frame", "diagnostic_low_basis"}
        state = {key[len(prefix):]: value for key, value in state_dict.items()
                 if key.startswith(prefix) and key[len(prefix):] not in ignored}
        result = self.load_state_dict(state, strict=True)
        self._analysis = None
        self.reference_prediction = None
        self.canonical_reference_prediction = None
        self.mlp._last_origin = None
        return result
