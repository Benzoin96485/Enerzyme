"""Equivariant FFN for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .activation import GatedActivation, SwiGLU
from .projection import S2GridProjector


class SO3Linear(nn.Module):
    """Per-degree linear with optional l=0 bias."""

    def __init__(self, lmax: int, in_channels: int, out_channels: int,
                 bias: bool = False, init_std: float | None = None) -> None:
        super().__init__()
        self.lmax = lmax
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Shared weight across all (l,m) — simple channel mixing
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if init_std is not None and init_std == 0.0:
            nn.init.zeros_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, D, F, C_in) -> (N, D, F, C_out)."""
        out = torch.matmul(x, self.weight)
        if self.bias is not None:
            # Add bias to l=0 component only, out-of-place
            bias_contrib = torch.zeros_like(out)
            bias_contrib[:, 0, :, :] = self.bias
            out = out + bias_contrib
        return out


class EquivariantFFN(nn.Module):
    """Equivariant feed-forward network.

    SO3Linear -> GatedActivation (or grid SwiGLU) -> SO3Linear.

    When ffn_so3_grid=True, projects to Lebedev grid, applies SwiGLU,
    projects back.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        hidden_channels: int,
        glu_activation: bool = True,
        activation: str = "silu",
        ffn_so3_grid: bool = False,
        lebedev_quadrature: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.glu_activation = glu_activation
        self.use_grid = ffn_so3_grid

        if self.use_grid:
            # Grid path: project to grid, SwiGLU, project back
            self.so3_linear_1 = SO3Linear(lmax, channels, 2 * hidden_channels)
            self.grid_proj = S2GridProjector(lmax)
            self.grid_act = SwiGLU()
            # Scalar linear bypass for l=0
            self.scalar_gate = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels, bias=False),
                nn.SiLU(),
            )
        else:
            out_ch = 2 * hidden_channels if glu_activation else hidden_channels
            self.so3_linear_1 = SO3Linear(lmax, channels, out_ch)
            self.act = GatedActivation(
                lmax=lmax, channels=hidden_channels,
                activation=activation, layout="ndfc",
            )

        self.so3_linear_2 = SO3Linear(lmax, hidden_channels, channels, init_std=0.0)

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, D, F, C) -> (N, D, F, C)."""
        h = self.so3_linear_1(x)

        if self.use_grid:
            # Grid path
            nc = self.hidden_channels
            N, D, F, _ = h.shape
            # Reshape to (N*F, D, 2*hidden) for grid projection
            h_flat = h.reshape(N * F, D, -1)
            g = self.grid_proj.to_grid(h_flat)  # (N*F, G, 2*hidden)
            g = self.grid_act(g)  # SwiGLU on grid
            h_back = self.grid_proj.from_grid(g)  # (N*F, D, hidden)
            h = h_back.reshape(N, D, F, nc)
        else:
            if self.glu_activation:
                nc = self.hidden_channels
                h_val = h[..., :nc]
                h_gate = h[..., nc:]
                h = self.act(h_val, gate=h_gate)
            else:
                h = self.act(h)

        return self.so3_linear_2(h)
