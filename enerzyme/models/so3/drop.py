"""Equivariant dropout for SH-array features ``[N, sphere_basis, C]``."""

from __future__ import annotations

import torch
import torch.nn as nn


class EquivariantDropoutArraySphericalHarmonics(nn.Module):
    """Channel dropout for SH-array features (broadcast over ``m``)."""

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
