"""Equivariant dropout for flat e3nn irreps features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3


class EquivariantDropout(nn.Module):
    """Per-irrep-copy dropout for flat e3nn irreps features."""

    def __init__(self, irreps, drop_prob):
        super().__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps("{}x0e".format(self.num_irreps))
        )

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        return self.mul(x, mask)


class EquivariantScalarsDropout(nn.Module):
    """Dropout only on scalar (l=0) slices of flat e3nn irreps features."""

    def __init__(self, irreps, drop_prob):
        super().__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        return torch.cat(out, dim=-1)

    def extra_repr(self):
        return "irreps={}, drop_prob={}".format(self.irreps, self.drop_prob)
