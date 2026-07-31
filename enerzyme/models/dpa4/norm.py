"""Normalization layers for DPA4.

Reimplemented after deepmd-kit ``dpa4_nn.norm`` (Li et al., arXiv:2606.02419).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .indexing import build_m_major_l_index, map_degree_idx


class EquivariantRMSNorm(nn.Module):
    """Degree-balanced equivariant RMSNorm on packed ``(N, D, F, C)`` features."""

    def __init__(
        self,
        lmax: int,
        channels: int,
        n_focus: int = 1,
        eps: float = 1e-5,
        mmax: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.eps = float(eps)
        self.mmax = None if mmax is None else int(mmax)
        self.scale = nn.Parameter(torch.ones(self.lmax + 1, self.n_focus, self.channels))
        self.bias = nn.Parameter(torch.zeros(self.n_focus, self.channels))
        if self.mmax is None:
            expand = map_degree_idx(self.lmax)
        else:
            expand = build_m_major_l_index(self.lmax, self.mmax)
        self.register_buffer("expand_index", torch.as_tensor(expand, dtype=torch.long))
        weights = []
        scale = 1.0 / ((self.lmax + 1) * self.channels)
        # For reduced m-major layouts, approximate with packed degree weights
        # on the expanded degree index length.
        for l in expand.tolist():
            weights.append(scale / (2 * int(l) + 1))
        self.register_buffer(
            "balance_weight", torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, D, F, C) or (F, N, D, C) — we use ndfc
        in_dtype = x.dtype
        x = x.float()
        x0 = x[:, :1]
        xt = x[:, 1:]
        x0 = x0 - x0.mean(dim=-1, keepdim=True)
        bw = self.balance_weight.to(device=x.device, dtype=x.dtype)
        mean_var = (x0 * x0).sum(dim=(1, 3)) * bw[0]
        if xt.numel() > 0:
            mean_var = mean_var + (
                (xt * xt) * bw[1:].view(1, -1, 1, 1)
            ).sum(dim=(1, 3))
        # mean_var: (N, F)
        inv_rms = torch.rsqrt(mean_var.clamp_min(self.eps)).view(
            x.shape[0], 1, self.n_focus, 1
        )
        x = torch.cat([x0, xt], dim=1) * inv_rms
        scale = self.scale.index_select(0, self.expand_index).to(dtype=x.dtype)
        # scale: (D, F, C)
        x = x * scale.unsqueeze(0)
        x = torch.cat(
            [x[:, :1] + self.bias.to(dtype=x.dtype).view(1, 1, self.n_focus, self.channels), x[:, 1:]],
            dim=1,
        )
        return x.to(dtype=in_dtype)


class IdentityNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
