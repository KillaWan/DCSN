"""Reproduce archived seed streams without retaining discarded model layers.

These helpers run only during construction. Their random draws must stay in
the original order; they have no parameters and do not participate in forward.
"""
import math

import torch
from torch import nn


def consume_linear(in_features, out_features, bias=True):
    weight = torch.empty(out_features, in_features)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias:
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        torch.empty(out_features).uniform_(-bound, bound)


def consume_tcn(channels, dataset):
    """Discarded Conv1d initializations, including each block's normal reset."""
    if dataset is None:
        dataset = {21: "Weather", 321: "Electricity", 862: "Traffic"}.get(channels, "")
    key = dataset.lower()
    if "traffic" in key:
        schedule = (512, 1024, 1024, 512, channels)
    elif "elec" in key:
        schedule = (256, 512, 1024, 512, channels)
    elif "wea" in key:
        schedule = (32, 64, 32, channels)
    else:
        schedule = (16, 32, 64, 32, channels)
    previous = channels
    for current in schedule:
        shapes = [(current, previous, 3), (current, current, 3)]
        if previous != current:
            shapes.append((current, previous, 1))
        for shape in shapes:
            weight = torch.empty(shape)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(shape[1] * shape[2])
            torch.empty(shape[0]).uniform_(-bound, bound)
        for shape in shapes:
            torch.empty(shape).normal_(0, 0.01)
        previous = current
