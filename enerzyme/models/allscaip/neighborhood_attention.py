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


class NeighborhoodAttention(nn.Module):
    """Self-attention over neighbors per source node (fairchem AllScAIP).

    With rotation-invariant scalar messages (``EquivariantInputBlock`` + no SH frequency
    mask), query/key dot-product logits are invariant; value projections remain standard.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        attn_num_heads: int,
        atten_dropout: float,
        use_freq_mask: bool,
        freequency_list: list[int],
        use_residual_scaling: bool,
        normalization: str,
    ) -> None:
        super().__init__()
        self.attn_num_heads = attn_num_heads
        self.src_attn = BaseAttention(hidden_size, attn_num_heads, atten_dropout)
        self.dst_attn = BaseAttention(hidden_size, attn_num_heads, atten_dropout)
        self.use_freq_mask = use_freq_mask

        if use_freq_mask:
            self.repeating_dimensions_list = list(freequency_list)
            self.rep_dim_len = len(self.repeating_dimensions_list)
            freq_dim = 0
            for _l, rep_count in enumerate(freequency_list):
                if rep_count > 0:
                    freq_dim += rep_count * (2 * _l + 1)
            padding_size = (8 - freq_dim % 8) % 8
            self.padding_size = padding_size
        else:
            self.repeating_dimensions_list = []
            self.rep_dim_len = 0
            self.padding_size = 0

        self.src_attn_norm = get_normalization(normalization, hidden_size)
        self.dst_attn_norm = get_normalization(normalization, hidden_size)

        if use_residual_scaling:
            s = 1.0 / num_layers
            self.src_attn_res_scale = nn.Parameter(torch.tensor(s), requires_grad=True)
            self.dst_attn_res_scale = nn.Parameter(torch.tensor(s), requires_grad=True)
        else:
            self.src_attn_res_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.dst_attn_res_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, data: GraphAttentionData, neighbor_reps: Tensor) -> Tensor:
        neighbor_reps = (
            neighbor_reps
            + self.src_attn_res_scale
            * self.multi_head_self_attention(
                attn_module=self.src_attn,
                input_tensor=self.src_attn_norm(neighbor_reps),
                attn_mask=data.src_neighbor_attn_mask[:, None, None, :],
                frequency_vectors=data.frequency_vectors,
            )
        )

        neighbor_reps = neighbor_reps[data.dst_index[0], data.dst_index[1]]
        if self.use_freq_mask and data.frequency_vectors is not None:
            frequency_vectors_dst = data.frequency_vectors[data.dst_index[0], data.dst_index[1]]
        else:
            frequency_vectors_dst = None

        neighbor_reps = (
            neighbor_reps
            + self.dst_attn_res_scale
            * self.multi_head_self_attention(
                attn_module=self.dst_attn,
                input_tensor=self.dst_attn_norm(neighbor_reps),
                attn_mask=data.dst_neighbor_attn_mask[:, None, None, :],
                frequency_vectors=frequency_vectors_dst,
            )
        )

        neighbor_reps = neighbor_reps[data.src_index[0], data.src_index[1]]
        return neighbor_reps

    def multi_head_self_attention(
        self,
        attn_module: BaseAttention,
        input_tensor: Tensor,
        attn_mask: Tensor,
        frequency_vectors: Tensor | None = None,
    ) -> Tensor:
        q, k, v = attn_module.qkv_projection(input_tensor, input_tensor, input_tensor)
        if self.use_freq_mask and frequency_vectors is not None:
            q, k = self.apply_frequency_embedding(q, k, frequency_vectors)
        return attn_module.scaled_dot_product_attention(q, k, v, attn_mask)

    def apply_frequency_embedding(
        self,
        q: Tensor,
        k: Tensor,
        frequency_vectors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        num_nodes, num_heads, num_neighbors, head_dim = q.shape
        freq_vecs = frequency_vectors.unsqueeze(1)

        q_expanded_sections: list[Tensor] = []
        k_expanded_sections: list[Tensor] = []
        curr_pos = 0

        for _l in range(self.rep_dim_len):
            rep_count = self.repeating_dimensions_list[_l]
            if rep_count == 0:
                continue
            if curr_pos >= head_dim:
                break
            sh_dim = 2 * _l + 1
            end_pos = min(curr_pos + rep_count, head_dim)
            q_section = q[..., curr_pos:end_pos]
            k_section = k[..., curr_pos:end_pos]
            q_section = q_section.unsqueeze(-1)
            k_section = k_section.unsqueeze(-1)
            q_expanded = q_section.expand(-1, -1, -1, -1, sh_dim)
            k_expanded = k_section.expand(-1, -1, -1, -1, sh_dim)
            q_expanded = q_expanded.reshape(num_nodes, self.attn_num_heads, num_neighbors, -1)
            k_expanded = k_expanded.reshape(num_nodes, self.attn_num_heads, num_neighbors, -1)
            q_expanded_sections.append(q_expanded)
            k_expanded_sections.append(k_expanded)
            curr_pos = end_pos

        if q_expanded_sections:
            q = torch.cat(q_expanded_sections, dim=-1)
            k = torch.cat(k_expanded_sections, dim=-1)

        q = q * freq_vecs
        k = k * freq_vecs

        if self.padding_size > 0:
            q = F.pad(q, (0, self.padding_size))
            k = F.pad(k, (0, self.padding_size))
        return q, k
