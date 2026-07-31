"""Equivariant FFN for DPA4.

Uses shared :class:`~enerzyme.models.so3.linear.SO3FocusLinear` and Lebedev
projection rather than a channel-only linear that ignores degree structure.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..so3.gated import SO3GatedActivation
from ..so3.linear import SO3FocusLinear
from .activation import SwiGLU
from .projection import S2GridProjector


class EquivariantFFN(nn.Module):
    """Equivariant feed-forward network.

    ``SO3FocusLinear`` → gated / Lebedev-grid SwiGLU → zero-init ``SO3FocusLinear``.
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
        del lebedev_quadrature  # always Lebedev when grid path is enabled
        self.lmax = lmax
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.glu_activation = glu_activation
        self.use_grid = ffn_so3_grid

        if self.use_grid:
            self.so3_linear_1 = SO3FocusLinear(
                lmax, channels, 2 * hidden_channels, n_focus=1
            )
            self.grid_proj = S2GridProjector(lmax)
            self.grid_act = SwiGLU()
        else:
            out_ch = 2 * hidden_channels if glu_activation else hidden_channels
            self.so3_linear_1 = SO3FocusLinear(
                lmax, channels, out_ch, n_focus=1
            )
            self.act = SO3GatedActivation(
                lmax=lmax,
                channels=hidden_channels,
                activation=activation,
                layout="ndfc",
            )

        self.so3_linear_2 = SO3FocusLinear(
            lmax, hidden_channels, channels, n_focus=1, init_std=0.0
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, D, F, C) -> (N, D, F, C)."""
        h = self.so3_linear_1(x)

        if self.use_grid:
            nc = self.hidden_channels
            n, d, f, _ = h.shape
            h_flat = h.reshape(n * f, d, -1)
            g = self.grid_proj.to_grid(h_flat)
            g = self.grid_act(g)
            h = self.grid_proj.from_grid(g).reshape(n, d, f, nc)
        elif self.glu_activation:
            nc = self.hidden_channels
            h = self.act(h[..., :nc], gate=h[..., nc:])
        else:
            h = self.act(h)

        return self.so3_linear_2(h)
