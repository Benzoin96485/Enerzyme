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


class EquivariantDegreeDropout(nn.Module):
    """Degree×channel dropout for SH-array features (all ``m`` of a type-L share a mask).

    Used by EquiformerV3. Distinct from
    :class:`EquivariantDropoutArraySphericalHarmonics` (channel-only broadcast).
    """

    def __init__(self, lmax, mmax, drop_prob, use_m_primary=False):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.drop_prob = drop_prob
        self.use_m_primary = use_m_primary
        self.drop = torch.nn.Dropout(drop_prob, True)

        expand_index = []
        if not self.use_m_primary:
            for l in range(self.lmax + 1):
                mmax_l = min(l, self.mmax)
                expand_index.append(
                    torch.ones((2 * mmax_l + 1,), dtype=torch.long) * l
                )
        else:
            for m in range(self.mmax + 1):
                l_index = torch.arange(self.lmax + 1 - m)
                expand_index.append(l_index)
                if m > 0:
                    expand_index.append(l_index)  # +- m
        self.register_buffer("expand_index", torch.cat(expand_index, dim=0).long())

    def forward(self, x):
        # x: (N, num_m_coefficients, C)
        if not self.training or self.drop_prob == 0.0:
            return x
        assert len(x.shape) == 3
        shape = (x.shape[0], self.lmax + 1, x.shape[2])
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        mask = torch.index_select(mask, dim=1, index=self.expand_index)
        return x * mask

    def extra_repr(self):
        return "lmax={}, mmax={}, drop_prob={}, use_m_primary={}".format(
            self.lmax, self.mmax, self.drop_prob, self.use_m_primary
        )
