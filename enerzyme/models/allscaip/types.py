"""Graph tensors for AllScAIP-style attention (layout matches fairchem AllScAIP)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class GraphAttentionData:
    """Per-batch graph features for neighborhood / node self-attention (fairchem-compatible)."""

    atomic_numbers: Tensor
    charge: Tensor
    spin: Tensor
    node_direction_expansion: Tensor
    edge_distance_expansion: Tensor
    edge_direction_expansion: Tensor
    edge_direction: Tensor
    src_neighbor_attn_mask: Tensor
    dst_neighbor_attn_mask: Tensor
    src_index: Tensor
    dst_index: Tensor
    frequency_vectors: Tensor | None
    node_base_attn_mask: Tensor | None
    node_sincx_matrix: Tensor | None
    node_valid_mask: Tensor | None
    neighbor_index: Tensor
    node_batch: Tensor
    node_padding_mask: Tensor
    max_batch_size: int
    num_graphs: int
    max_num_nodes: int
    num_nodes: int

    def to(self, device: torch.device) -> GraphAttentionData:
        new: dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                new[k] = v.to(device)
            else:
                new[k] = v
        return GraphAttentionData(**new)
