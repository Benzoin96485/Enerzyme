# SPDX-License-Identifier: MIT
# Adapted from fairchem AllScAIP (https://github.com/FAIR-Chem/fairchem), MIT License.
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections (fairchem AllScAIP layout)."""

    def __init__(
        self,
        hidden_size: int,
        attn_num_heads: int,
        atten_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_in_proj_q = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_in_proj_k = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_in_proj_v = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_num_heads = attn_num_heads
        self.attn_dropout = atten_dropout

    def qkv_projection(
        self, q: Tensor, k: Tensor, v: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, q_seq_len, hidden_dim = q.shape
        vk_seq_len = k.shape[1]
        head_dim = hidden_dim // self.attn_num_heads
        q = (
            self.attn_in_proj_q(q)
            .reshape(batch_size, q_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.attn_in_proj_k(k)
            .reshape(batch_size, vk_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.attn_in_proj_v(v)
            .reshape(batch_size, vk_seq_len, self.attn_num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        return q, k, v

    def scaled_dot_product_attention(
        self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None
    ) -> Tensor:
        batch_size, num_heads, q_seq_len, _ = q.shape
        head_dim_v = v.shape[-1]
        embed_dim = q.shape[-1]
        if embed_dim != head_dim_v:
            v = F.pad(v, (0, embed_dim - head_dim_v))

        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(embed_dim),
        )

        if attn_output.shape[-1] != head_dim_v:
            attn_output = attn_output[..., :head_dim_v]

        hidden_dim = num_heads * head_dim_v
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, q_seq_len, hidden_dim
        )
        attn_output = self.attn_out_proj(attn_output)
        return attn_output
