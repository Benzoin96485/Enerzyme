"""Interaction block and equivariant FFN for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..activation import SwiGLU
from ..so3.gated import SO3GatedActivation
from ..so3.layer_norm import EquivariantDegreeRMSNorm
from ..so3.lebedev import S2LebedevProjector
from ..so3.linear import SO3FocusLinear
from .so2 import EdgeCache, SO2Convolution


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
            self.grid_proj = S2LebedevProjector(lmax)
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


class SeZMInteractionBlock(nn.Module):
    """SeZM interaction block: SO(2) conv + FFN with pre/post norms.

    sandwich_norm controls which norms are applied:
    [so2_pre, so2_post, ffn_pre, ffn_post]
    Default for water-mini: [False, True, True, False]
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        mixing_layers: int = 3,
        n_atten_head: int = 1,
        radial_so2_mode: str = "degree_channel",
        radial_so2_rank: int = 1,
        n_radial: int = 16,
        ffn_neurons: int = 96,
        ffn_blocks: int = 1,
        ffn_so3_grid: bool = True,
        lebedev_quadrature: bool = True,
        glu_activation: bool = True,
        activation: str = "silu",
        sandwich_norm: Optional[list] = None,
        message_node_so3: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.ffn_blocks_count = ffn_blocks

        if sandwich_norm is None:
            sandwich_norm = [False, True, True, False]

        self.pre_so2_norm = (
            EquivariantDegreeRMSNorm(lmax, channels)
            if sandwich_norm[0]
            else nn.Identity()
        )
        self.post_so2_norm = (
            EquivariantDegreeRMSNorm(lmax, channels)
            if sandwich_norm[1]
            else nn.Identity()
        )

        self.so2_conv = SO2Convolution(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            n_focus=n_focus,
            focus_dim=focus_dim,
            mixing_layers=mixing_layers,
            n_atten_head=n_atten_head,
            radial_so2_mode=radial_so2_mode,
            radial_so2_rank=radial_so2_rank,
            n_radial=n_radial,
            glu_activation=glu_activation,
            activation=activation,
            message_node_so3=message_node_so3,
        )

        self.pre_ffn_norms = nn.ModuleList()
        self.post_ffn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(ffn_blocks):
            self.pre_ffn_norms.append(
                EquivariantDegreeRMSNorm(lmax, channels)
                if sandwich_norm[2]
                else nn.Identity()
            )
            self.post_ffn_norms.append(
                EquivariantDegreeRMSNorm(lmax, channels)
                if sandwich_norm[3]
                else nn.Identity()
            )
            self.ffns.append(
                EquivariantFFN(
                    lmax=lmax,
                    channels=channels,
                    hidden_channels=ffn_neurons,
                    glu_activation=glu_activation,
                    activation=activation,
                    ffn_so3_grid=ffn_so3_grid,
                    lebedev_quadrature=lebedev_quadrature,
                )
            )

    def forward(self, x: Tensor, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            x: (N, D, 1, C) canonical node layout
            edge_cache: EdgeCache
            radial_feat: (E, lmax+1, C)

        Returns:
            (N, D, 1, C) updated features
        """
        x_pre = self.pre_so2_norm(x)
        so2_out = self.so2_conv(x_pre.squeeze(2), edge_cache, radial_feat)
        so2_out = self.post_so2_norm(so2_out.unsqueeze(2))
        x = x + so2_out

        for i in range(self.ffn_blocks_count):
            x_ffn = self.pre_ffn_norms[i](x)
            y = self.ffns[i](x_ffn)
            y = self.post_ffn_norms[i](y)
            x = x + y

        return x
