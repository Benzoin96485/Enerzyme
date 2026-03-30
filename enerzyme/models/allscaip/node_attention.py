# SPDX-License-Identifier: MIT
# Adapted from fairchem AllScAIP (https://github.com/FAIR-Chem/fairchem), MIT License.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_attention import BaseAttention
from .nn_utils import get_normalization
from .types import GraphAttentionData


class NodeAttention(BaseAttention):
    """All-to-all node self-attention with optional sinc radial bias (fairchem AllScAIP)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        attn_num_heads: int,
        attn_num_freq: int,
        atten_dropout: float,
        use_sincx_mask: bool,
        use_residual_scaling: bool,
        normalization: str,
    ) -> None:
        super().__init__(hidden_size, attn_num_heads, atten_dropout)
        self.num_heads = attn_num_heads
        self.use_sincx_mask = use_sincx_mask
        if use_sincx_mask:
            self.radial_weight = nn.Parameter(
                torch.ones(attn_num_heads, attn_num_freq) / attn_num_freq,
                requires_grad=True,
            )
        else:
            self.radial_weight = None

        self.node_norm = get_normalization(normalization, hidden_size)
        if use_residual_scaling:
            self.node_attn_res_scale = nn.Parameter(
                torch.tensor(1.0 / num_layers), requires_grad=True
            )
        else:
            self.node_attn_res_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, data: GraphAttentionData, node_reps: Tensor) -> Tensor:
        node_reps_normalized = self.node_norm(node_reps)
        node_attn_mask = self.get_node_attention_mask(data, self.radial_weight)

        q, k, v = self.qkv_projection(
            node_reps_normalized.unsqueeze(0),
            node_reps_normalized.unsqueeze(0),
            node_reps_normalized.unsqueeze(0),
        )

        if node_attn_mask is not None:
            node_attn_mask = node_attn_mask[None, :, :, :].to(q.dtype)
        attn_output = self.scaled_dot_product_attention(q, k, v, node_attn_mask)
        return self.node_attn_res_scale * attn_output.squeeze(0) + node_reps

    def get_node_attention_mask(
        self,
        data: GraphAttentionData,
        radial_weight: Tensor | None,
        eps: float = 1e-6,
        normalize: bool = True,
    ) -> Tensor | None:
        if data.node_base_attn_mask is None and not self.use_sincx_mask:
            return None

        if self.use_sincx_mask:
            if radial_weight is None:
                raise ValueError("radial_weight is None when use_sincx_mask is True")
            if data.node_sincx_matrix is None:
                raise ValueError("node_sincx_matrix is None when use_sincx_mask is True")
            if data.node_valid_mask is None:
                raise ValueError("node_valid_mask is None when use_sincx_mask is True")

            freq_weight = torch.einsum("ijk,hk->hij", data.node_sincx_matrix, radial_weight)
            freq_weight = freq_weight.masked_fill(~data.node_valid_mask.unsqueeze(0), 0)
            freq_weight = F.softplus(freq_weight) + eps
            if normalize:
                denom = freq_weight.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
                freq_weight = freq_weight * (data.node_valid_mask.sum() / denom)
            attn_bias = freq_weight.log()
            if data.node_base_attn_mask is not None:
                return attn_bias + data.node_base_attn_mask.expand(self.num_heads, -1, -1)
            return attn_bias

        if data.node_base_attn_mask is None:
            return None
        return data.node_base_attn_mask.expand(self.num_heads, -1, -1)
