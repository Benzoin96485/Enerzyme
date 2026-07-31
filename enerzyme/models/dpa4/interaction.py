"""Interaction block for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .edge_cache import EdgeCache
from .ffn import EquivariantFFN
from .norm import EquivariantRMSNorm
from .so2 import SO2Convolution


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
        sandwich_norm: list[bool] | None = None,
        message_node_so3: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.channels = channels
        self.ffn_blocks_count = ffn_blocks

        if sandwich_norm is None:
            sandwich_norm = [False, True, True, False]

        # SO(2) pre-norm
        self.pre_so2_norm = (
            EquivariantRMSNorm(lmax, channels)
            if sandwich_norm[0] else nn.Identity()
        )
        # SO(2) post-norm
        self.post_so2_norm = (
            EquivariantRMSNorm(lmax, channels)
            if sandwich_norm[1] else nn.Identity()
        )

        self.so2_conv = SO2Convolution(
            lmax=lmax, mmax=mmax, channels=channels,
            n_focus=n_focus, focus_dim=focus_dim,
            mixing_layers=mixing_layers,
            n_atten_head=n_atten_head,
            radial_so2_mode=radial_so2_mode,
            radial_so2_rank=radial_so2_rank,
            n_radial=n_radial,
            glu_activation=glu_activation,
            activation=activation,
            message_node_so3=message_node_so3,
        )

        # FFN subblocks
        self.pre_ffn_norms = nn.ModuleList()
        self.post_ffn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(ffn_blocks):
            self.pre_ffn_norms.append(
                EquivariantRMSNorm(lmax, channels)
                if sandwich_norm[2] else nn.Identity()
            )
            self.post_ffn_norms.append(
                EquivariantRMSNorm(lmax, channels)
                if sandwich_norm[3] else nn.Identity()
            )
            self.ffns.append(EquivariantFFN(
                lmax=lmax, channels=channels,
                hidden_channels=ffn_neurons,
                glu_activation=glu_activation,
                activation=activation,
                ffn_so3_grid=ffn_so3_grid,
                lebedev_quadrature=lebedev_quadrature,
            ))

    def forward(self, x: Tensor, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            x: (N, D, 1, C) canonical node layout
            edge_cache: EdgeCache
            radial_feat: (E, lmax+1, C)

        Returns:
            (N, D, 1, C) updated features
        """
        N, D, _, C = x.shape

        # SO(2) branch
        x_so2 = x
        x_pre = self.pre_so2_norm(x_so2)
        so2_out = self.so2_conv(
            x_pre.squeeze(2),  # (N, D, C)
            edge_cache,
            radial_feat,
        )
        so2_out = self.post_so2_norm(so2_out.unsqueeze(2))  # (N, D, 1, C)
        x = x + so2_out

        # FFN branch(es)
        for i in range(self.ffn_blocks_count):
            x_ffn = self.pre_ffn_norms[i](x)
            y = self.ffns[i](x_ffn)
            y = self.post_ffn_norms[i](y)
            x = x + y

        return x
