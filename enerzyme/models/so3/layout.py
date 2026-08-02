"""e3nn flat-Irreps ↔ degree-primary (D, C) layout for SO(2) / TECE.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import torch
from e3nn import o3


class LayoutTransform(torch.nn.Module):
    """Convert between e3nn flat layout and ``[N, D, C]`` (degree-major)."""

    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.muls = []
        self.dims = []
        for mul, ir in self.irreps:
            self.muls.append(mul)
            self.dims.append(ir.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = 0
        out = []
        batch = x.size(0)
        for mul, d in zip(self.muls, self.dims):
            field = x[:, start : start + mul * d]
            start += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1).transpose(-1, -2).contiguous()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2).contiguous()
        start = 0
        out = []
        batch = x.size(0)
        for _, d in zip(self.muls, self.dims):
            field = x[:, :, start : start + d]
            start += d
            field = field.reshape(batch, -1)
            out.append(field)
        return torch.cat(out, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps})"
