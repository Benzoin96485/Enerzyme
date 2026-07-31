# Copyright (c) EquiformerV2 authors (Liao et al., ICLR 2024).
# Ported from https://github.com/atomicarchitects/equiformer_v2 (MIT License).
"""Dropout helpers used by EquiformerV2 TransBlockV2."""

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class GraphDropPath(nn.Module):
    """Drop paths with per-graph masks via ``batch`` indices."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (x.ndim - 1)
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = drop_path(ones, self.drop_prob, self.training)
        return x * drop[batch]

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class EquivariantDropoutArraySphericalHarmonics(nn.Module):
    def __init__(self, drop_prob, drop_graph=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.drop_graph = drop_graph

    def forward(self, x, batch=None):
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3

        if self.drop_graph:
            assert batch is not None
            batch_size = batch.max() + 1
            shape = (batch_size, 1, x.shape[2])
            mask = torch.ones(shape, dtype=x.dtype, device=x.device)
            mask = self.drop(mask)
            return x * mask[batch]

        shape = (x.shape[0], 1, x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        return x * mask

    def extra_repr(self):
        return "drop_prob={}, drop_graph={}".format(self.drop_prob, self.drop_graph)
