# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
"""Sparse Enerzyme edges → padded top-K neighborhoods for E2 attention."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def center_positions_by_batch(Ra: Tensor, batch_seg: Tensor) -> Tensor:
    """Subtract per-graph centers of mass from absolute coordinates.

    Wigner-6j attention expands solid harmonics of absolute node positions. Those
    harmonics are mathematically equivalent to relative-edge expansions after
    recoupling, but are ill-conditioned when |R| is large. Centering each graph
    restores stable translational invariance without changing relative geometry.
    """
    if batch_seg.numel() == 0:
        return Ra
    num_graphs = int(batch_seg.max().item()) + 1
    com = Ra.new_zeros((num_graphs, 3))
    counts = Ra.new_zeros((num_graphs, 1))
    com.index_add_(0, batch_seg.long(), Ra)
    counts.index_add_(
        0,
        batch_seg.long(),
        torch.ones(Ra.shape[0], 1, device=Ra.device, dtype=Ra.dtype),
    )
    com = com / counts.clamp_min(1.0)
    return Ra - com[batch_seg.long()]


def _topk_slots(dst: Tensor, dist: Tensor, max_neighbors: int, num_nodes: int):
    """Return original edge IDs and their flattened slots in the padded neighborhood."""
    if not dst.numel() or max_neighbors <= 0:
        return dst[:0], dst[:0]
    # Preserve the original distance sort and tie order; keep its span on-device.
    span = dist.detach().max().clamp_min(1.0) + 1.0
    order = torch.argsort(dst.double() * span.double() + dist.double())
    grouped = dst[order]
    centers = torch.arange(num_nodes, device=dst.device)
    starts = torch.searchsorted(grouped, centers)
    rank = torch.arange(dst.numel(), device=dst.device) - starts[grouped]
    kept = (rank < max_neighbors).nonzero().flatten()
    return order[kept], grouped[kept] * max_neighbors + rank[kept]


def packed_neighbors(src, dst, dist, max_neighbors, num_nodes):
    """Build source indices, occupancy and scatter locations from the same distance sort."""
    selected, slots = _topk_slots(dst, dist, max_neighbors, num_nodes)
    neighbors = src.new_full((num_nodes * max_neighbors,), -1)
    neighbors.index_copy_(0, slots, src[selected])
    neighbors = neighbors.view(num_nodes, max_neighbors)
    return selected, slots, neighbors.clamp_min(0), neighbors >= 0


def select_closest_neighbors(
    src: Tensor,
    dst: Tensor,
    dist: Tensor,
    max_neighbors: int,
    num_nodes: int,
    *edge_attrs: Tensor,
) -> Tuple[Tensor, ...]:
    """Keep at most ``max_neighbors`` closest edges per destination atom.

    Matches upstream E2Former / FairChem radius-graph behavior: when the cutoff
    neighborhood is denser than ``max_neighbors``, retain nearest neighbors
    instead of aborting.
    """
    selected, _ = _topk_slots(dst, dist, max_neighbors, num_nodes)
    return (src[selected], dst[selected], dist[selected]) + tuple(a[selected] for a in edge_attrs)


def pad_neighbor_list(
    src: Tensor,
    dst: Tensor,
    max_neighbors: int,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pad truncated COO edges into ``[N, K]`` neighbor lists.

    Assumes each destination already has ``degree <= max_neighbors``.
    """
    device = dst.device
    neighbor_counts = torch.bincount(dst, minlength=num_nodes)
    if neighbor_counts.numel() and int(neighbor_counts.max().item()) > max_neighbors:
        raise RuntimeError(
            "pad_neighbor_list received degree > max_neighbors; call "
            "select_closest_neighbors first"
        )

    offset = max_neighbors - neighbor_counts
    offset = torch.cat(
        [torch.zeros(1, dtype=offset.dtype, device=device), torch.cumsum(offset, dim=0)]
    )
    index_mapping = torch.arange(dst.shape[0], device=device) + offset[dst]

    neighbor_list = torch.full(
        (num_nodes * max_neighbors,), -1, dtype=torch.long, device=device
    )
    mask = torch.zeros((num_nodes * max_neighbors,), dtype=torch.bool, device=device)
    if dst.numel() > 0:
        neighbor_list.scatter_(0, index_mapping, src)
        mask.scatter_(
            0, index_mapping, torch.ones_like(src, dtype=torch.bool, device=device)
        )
    return (
        neighbor_list.view(num_nodes, max_neighbors),
        mask.view(num_nodes, max_neighbors),
        index_mapping,
    )


# Backward-compatible alias used by older call sites / tests.
def convert_neighbor_list(
    edge_index: Tensor, max_neighbors: int, num_nodes: int
) -> Tuple[Tensor, Tensor, Tensor]:
    return pad_neighbor_list(
        edge_index[0], edge_index[1], max_neighbors, num_nodes
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
    if index_mapping.numel() > 0:
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
    batch_seg: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Build E2Former-style dense neighborhood tensors from Enerzyme SR edges.

    Enerzyme convention: ``idx_i`` is the destination (center), ``idx_j`` source.
    ``vij_sr`` is the vector from ``i`` to ``j`` (``R_j - R_i`` in DistanceLayer).

    When a center has more short-range neighbors than ``max_neighbors``, the
    closest ``max_neighbors`` are kept (upstream E2Former radius-graph behavior).

    Absolute positions stored for Wigner-6j (``f_exp_node_pos``) are COM-centered
    per ``batch_seg``; relative ``edge_vec`` / ``edge_dis`` stay translation-free.
    """
    num_nodes = Ra.shape[0]
    device = Ra.device
    dtype = Ra.dtype

    if batch_seg is None:
        batch_seg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    ra_wigner = center_positions_by_batch(Ra.to(dtype=dtype), batch_seg)

    # edge_index[0]=src (j), edge_index[1]=dst (i)
    src = idx_j_sr.long()
    dst = idx_i_sr.long()
    dist = torch.linalg.norm(vij_sr.to(dtype=dtype), dim=-1)
    vij = vij_sr.to(dtype=dtype)
    rbf = rbf.to(dtype=dtype)

    if max_neighbors is None:
        deg = torch.bincount(dst, minlength=num_nodes) if dst.numel() else torch.zeros(
            num_nodes, dtype=torch.long, device=device
        )
        max_neighbors = int(deg.max().item()) if deg.numel() else 1
        max_neighbors = max(max_neighbors, 1)

    selected, index_mapping, safe_neighbors, present = packed_neighbors(
        src, dst, dist, max_neighbors, num_nodes
    )
    attn_mask = (~present).unsqueeze(-1)
    edge_vec = map_neighbor_list(vij[selected], index_mapping, max_neighbors, num_nodes)
    edge_dis = map_neighbor_list(dist[selected], index_mapping, max_neighbors, num_nodes)
    attn_weight = map_neighbor_list(rbf[selected], index_mapping, max_neighbors, num_nodes)

    return {
        "f_sparse_idx_node": safe_neighbors,
        "f_sparse_idx_expnode": safe_neighbors,
        "f_exp_node_pos": ra_wigner,
        "f_node_pos_wigner": ra_wigner,
        "f_outcell_index": torch.arange(num_nodes, device=device, dtype=torch.long),
        "edge_vec": edge_vec,
        "edge_dis": edge_dis,
        "attn_weight": attn_weight,
        "attn_mask": attn_mask,
        "present": present,
        "max_neighbors": torch.tensor(max_neighbors, device=device),
    }
