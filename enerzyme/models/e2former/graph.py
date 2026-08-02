# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
"""Sparse Enerzyme edges → padded top-K neighborhoods for E2 attention."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def convert_neighbor_list(
    edge_index: Tensor, max_neighbors: int, num_nodes: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert COO ``edge_index`` ``[2, E]`` (src→dst) into padded top-K lists.

    Returns
    -------
    neighbor_list : LongTensor [N, K]
        Source index per padded slot (``-1`` when unused).
    mask : BoolTensor [N, K]
        ``True`` where a real neighbor is present.
    index_mapping : LongTensor [E]
        Flat index of each edge inside the ``N * K`` padded buffer.
    """
    src = edge_index[0]
    dst = edge_index[1]
    device = edge_index.device

    neighbor_counts = torch.bincount(dst, minlength=num_nodes)
    if int(neighbor_counts.max().item()) > max_neighbors:
        raise ValueError(
            f"max_neighbors={max_neighbors} is smaller than observed degree "
            f"{int(neighbor_counts.max().item())}; increase Core max_neighbors "
            "or reduce the cutoff / graph density."
        )

    offset = max_neighbors - neighbor_counts
    offset = torch.cat(
        [torch.zeros(1, dtype=offset.dtype, device=device), torch.cumsum(offset, dim=0)]
    )
    index_mapping = torch.arange(edge_index.shape[1], device=device) + offset[dst]

    neighbor_list = torch.full(
        (num_nodes * max_neighbors,), -1, dtype=torch.long, device=device
    )
    mask = torch.zeros((num_nodes * max_neighbors,), dtype=torch.bool, device=device)
    neighbor_list.scatter_(0, index_mapping, src)
    mask.scatter_(
        0, index_mapping, torch.ones_like(src, dtype=torch.bool, device=device)
    )
    return (
        neighbor_list.view(num_nodes, max_neighbors),
        mask.view(num_nodes, max_neighbors),
        index_mapping,
    )


def map_neighbor_list(
    x: Tensor, index_mapping: Tensor, max_neighbors: int, num_nodes: int
) -> Tensor:
    """Scatter edge features ``[E, H]`` into padded ``[N, K, H]``."""
    if x.ndim == 1:
        x = x.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    out = x.new_zeros((num_nodes * max_neighbors, x.shape[-1]))
    out.scatter_(0, index_mapping.unsqueeze(1).expand(-1, x.shape[-1]), x)
    out = out.view(num_nodes, max_neighbors, x.shape[-1])
    return out.squeeze(-1) if squeeze else out


def build_topk_neighborhood(
    Ra: Tensor,
    idx_i_sr: Tensor,
    idx_j_sr: Tensor,
    vij_sr: Tensor,
    rbf: Tensor,
    max_neighbors: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Build E2Former-style dense neighborhood tensors from Enerzyme SR edges.

    Enerzyme convention: ``idx_i`` is the destination (center), ``idx_j`` source.
    ``vij_sr`` is the vector from ``i`` to ``j`` (``R_j - R_i`` in DistanceLayer).
    """
    num_nodes = Ra.shape[0]
    device = Ra.device
    dtype = Ra.dtype

    # edge_index[0]=src (j), edge_index[1]=dst (i) for convert_neighbor_list
    edge_index = torch.stack([idx_j_sr.long(), idx_i_sr.long()], dim=0)
    if max_neighbors is None:
        deg = torch.bincount(edge_index[1], minlength=num_nodes)
        max_neighbors = int(deg.max().item()) if deg.numel() else 1
        max_neighbors = max(max_neighbors, 1)

    neighbor_list, present, index_mapping = convert_neighbor_list(
        edge_index, max_neighbors, num_nodes
    )
    # Attention mask: True = padded / invalid (matches upstream)
    attn_mask = (~present).unsqueeze(-1)

    edge_vec = map_neighbor_list(vij_sr.to(dtype=dtype), index_mapping, max_neighbors, num_nodes)
    edge_dis = torch.linalg.norm(edge_vec, dim=-1)
    edge_dis = edge_dis.masked_fill(~present, 0.0)
    attn_weight = map_neighbor_list(rbf.to(dtype=dtype), index_mapping, max_neighbors, num_nodes)
    attn_weight = attn_weight.masked_fill(attn_mask, 0.0)

    # Safe neighbor gather index (unused slots → 0; masked later)
    safe_neighbors = neighbor_list.clamp(min=0)

    return {
        "f_sparse_idx_node": safe_neighbors,
        "f_sparse_idx_expnode": safe_neighbors,
        "f_exp_node_pos": Ra,
        "f_outcell_index": torch.arange(num_nodes, device=device, dtype=torch.long),
        "edge_vec": edge_vec,
        "edge_dis": edge_dis,
        "attn_weight": attn_weight,
        "attn_mask": attn_mask,
        "present": present,
        "max_neighbors": torch.tensor(max_neighbors, device=device),
    }
