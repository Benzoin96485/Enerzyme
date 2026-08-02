# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
"""E2Former transformer block: E2 attention + reused EquiformerV2 S2 FFN."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..equiformer_v2.interaction import FeedForwardNetwork
from ..so3 import SO3_Embedding, get_normalization_layer
from .attention import E2AttentionSparse


class _S2FeedForward(nn.Module):
    """Thin adapter: raw SH tensor ↔ EquiformerV2 ``FeedForwardNetwork``."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        SO3_grid,
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        ffn_activation: str = "scaled_silu",
    ) -> None:
        super().__init__()
        self.lmax_list = [lmax]
        self.mmax_list = [lmax]
        self.sphere_channels = sphere_channels
        self.ffn = FeedForwardNetwork(
            sphere_channels,
            hidden_channels,
            sphere_channels,
            self.lmax_list,
            self.mmax_list,
            SO3_grid,
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
        )

    def forward(self, node_irreps: Tensor) -> Tensor:
        emb = SO3_Embedding(
            node_irreps.shape[0],
            self.lmax_list,
            self.sphere_channels,
            node_irreps.device,
            node_irreps.dtype,
        )
        emb.set_embedding(node_irreps)
        emb.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())
        return self.ffn(emb).embedding


class TransBlock(nn.Module):
    """Pre-norm E2 attention + residual + S2 FFN (EquiformerV2 FFN reuse)."""

    def __init__(
        self,
        irreps_node_input: str | o3.Irreps,
        attn_weight_input_dim: int,
        num_attn_heads: int,
        attn_scalar_head: int,
        irreps_head: str | o3.Irreps,
        SO3_grid,
        ffn_hidden_channels: Optional[int] = None,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: str = "rms_norm_sh",
        attn_type: str = "first-order",
        tp_type: str = "QK_alpha",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        ffn_activation: str = "scaled_silu",
        atom_type_cnt: int = 256,
    ) -> None:
        super().__init__()
        self.irreps_node_input = (
            o3.Irreps(irreps_node_input)
            if isinstance(irreps_node_input, str)
            else irreps_node_input
        )
        self.lmax = self.irreps_node_input[-1][1].l
        self.scalar_dim = self.irreps_node_input[0][0]
        if ffn_hidden_channels is None:
            ffn_hidden_channels = 2 * self.scalar_dim

        self.norm_1 = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.ga = E2AttentionSparse(
            irreps_node_input=self.irreps_node_input,
            attn_weight_input_dim=attn_weight_input_dim,
            num_attn_heads=num_attn_heads,
            attn_scalar_head=attn_scalar_head,
            irreps_head=irreps_head,
            alpha_drop=alpha_drop,
            tp_type=tp_type,
            attn_type=attn_type,
            atom_type_cnt=atom_type_cnt,
        )
        self.norm_s2 = get_normalization_layer(
            norm_layer, lmax=self.lmax, num_channels=self.scalar_dim
        )
        self.ffn_s2 = _S2FeedForward(
            self.scalar_dim,
            ffn_hidden_channels,
            self.lmax,
            SO3_grid,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            ffn_activation=ffn_activation,
        )
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else None

    def forward(
        self,
        node_pos: Tensor,
        node_irreps: Tensor,
        edge_dis: Tensor,
        edge_vec: Tensor,
        attn_weight: Tensor,
        atomic_numbers: Tensor,
        attn_mask: Tensor,
        batched_data: Dict[str, Tensor],
        **kwargs,
    ):
        residual = node_irreps
        node_irreps = self.norm_1(node_irreps)
        node_irreps, attn_weight = self.ga(
            node_pos=node_pos,
            node_irreps_input=node_irreps,
            edge_dis=edge_dis,
            edge_vec=edge_vec,
            attn_weight=attn_weight,
            atomic_numbers=atomic_numbers,
            attn_mask=attn_mask,
            batched_data=batched_data,
        )
        node_irreps = residual + node_irreps

        residual = node_irreps
        node_irreps = self.norm_s2(node_irreps)
        node_irreps = self.ffn_s2(node_irreps)
        if self.proj_drop is not None:
            node_irreps = self.proj_drop(node_irreps)
        node_irreps = residual + node_irreps
        return node_irreps, attn_weight
