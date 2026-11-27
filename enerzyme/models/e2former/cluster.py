# Adapted from IQuestLab/UBio-MolFM tag E2Former-LSR (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""Fragment construction and atom–fragment bipartite neighborhoods for E2Former-LSR.

Assignment and per-graph ids live in
:mod:`enerzyme.models.layers.fragments` so ViSNet-LSRM can share them. This
module keeps the padded atom–fragment neighborhood used by E2Former attention.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor
from ..mp.ops import scatter_mean

from ..layers.fragments import (
    build_kmeans_fragments,
    centers_from_ids,
    flatten_cluster_ids,
    num_clusters_for_graph,
)

# Historical private name. The count now lives in layers.fragments.
_num_clusters_for_graph = num_clusters_for_graph
from .graph import map_neighbor_list, packed_neighbors


class FragmentNeighbors(NamedTuple):
    """Occupied slots in the padded atom-to-fragment neighborhood."""

    neighbors: Tensor
    occupied: Tensor
    atoms: Tensor
    sources: Tensor
    slots: Tensor

    def geometry(self, positions, centers):
        n, k = self.neighbors.shape
        if self.slots.numel():
            vectors = centers[self.sources] - positions[self.atoms]
            edge_vec = map_neighbor_list(vectors, self.slots, k, n)
            edge_dis = map_neighbor_list(vectors.norm(dim=-1), self.slots, k, n)
        else:
            edge_vec = positions.new_zeros(n, k, 3)
            edge_dis = positions.new_zeros(n, k)
        graph = {"f_sparse_idx_expnode": self.neighbors, "f_sparse_idx_node": self.neighbors,
                 "edge_vec": edge_vec, "edge_dis": edge_dis, "present": self.occupied,
                 "attn_mask": (~self.occupied).unsqueeze(-1), "f_cluster_pos": centers,
                 "max_neighbors": self.neighbors.new_tensor(k)}
        if self.slots.numel():
            graph["f_exp_node_pos"] = centers
        return graph


class FragmentTopology(NamedTuple):
    """Reuse the discrete assignments while differentiating positions and centers."""

    ids: Tensor
    counts: Tensor
    neighbors: FragmentNeighbors

    def apply(self, ctx):
        ctx.cache["lsr_fragments"] = self

    def geometry(self, positions):
        centers = positions.new_zeros(self.counts.shape[0], 3).index_add(0, self.ids, positions)
        centers = centers / self.counts.clamp_min(1).to(positions.dtype).unsqueeze(-1)
        return centers, self.neighbors.geometry(positions, centers)


__all__ = [
    "build_atom_fragment_topk",
    "build_kmeans_fragments",
    "centers_from_ids",
    "flatten_cluster_ids",
    "num_clusters_for_graph",
    "pool_fragment_irreps",
    "resolve_fragments",
]


def pool_fragment_irreps(node_irreps: Tensor, flat_cluster_ids: Tensor, num_clusters: Optional[int] = None) -> Tensor:
    """Mean-pool atom SH features onto fragments: ``[N,M,C] → [U,M,C]``."""
    n, m, c = node_irreps.shape
    flat = node_irreps.reshape(n, m * c)
    pooled = scatter_mean(flat, flat_cluster_ids.long(), dim_size=num_clusters)
    return pooled.reshape(-1, m, c)


def build_atom_fragment_topk(
    atom_pos: Tensor,
    cluster_pos: Tensor,
    flat_cluster_ids: Tensor,
    batch_seg: Tensor,
    cluster_batch: Tensor,
    radius: float,
    max_neighbors: int = 64,
    remove_self_cluster: bool = True,
) -> Dict[str, Tensor]:
    """Bipartite atom→fragment radius graph, padded to ``[N, K]``.

    Matches upstream ``construct_radius_neighbor`` semantics for the LSR path:
    top-K nearest fragments within ``radius``, optionally excluding an atom's
    own fragment (``remove_self_cluster``).
    """
    neighbors = select_fragment_neighbors(atom_pos, cluster_pos, flat_cluster_ids, batch_seg,
                                         cluster_batch, radius, max_neighbors, remove_self_cluster)
    return neighbors.geometry(atom_pos, cluster_pos)


@torch.no_grad()
def select_fragment_neighbors(
    atom_pos: Tensor,
    cluster_pos: Tensor,
    flat_cluster_ids: Tensor,
    batch_seg: Tensor,
    cluster_batch: Tensor,
    radius: float,
    max_neighbors: int = 64,
    remove_self_cluster: bool = True,
) -> FragmentNeighbors:
    """Rank pairs once. Excluded zero-length pairs never enter coordinate derivatives."""
    device = atom_pos.device
    num_atoms = atom_pos.shape[0]
    num_clusters = cluster_pos.shape[0]
    src_list = []
    dst_list = []
    dist_list = []

    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms and num_clusters and max_neighbors > 0 else 0
    for g in range(num_graphs):
        atom_mask = batch_seg == g
        cluster_mask = cluster_batch == g
        atom_idx = torch.where(atom_mask)[0]
        cluster_idx = torch.where(cluster_mask)[0]
        if atom_idx.numel() == 0 or cluster_idx.numel() == 0:
            continue
        a_pos = atom_pos[atom_idx]
        c_pos = cluster_pos[cluster_idx]
        # [Na, Nc, 3]: fragment - atom (matches upstream gather of expand - node after mask)
        delta = c_pos.unsqueeze(0) - a_pos.unsqueeze(1)
        dist = torch.linalg.norm(delta, dim=-1)
        if remove_self_cluster:
            local_self = flat_cluster_ids[atom_idx]
            # local_self is global flat id; map to local cluster column
            # cluster_idx are global; compare global ids
            self_mask = cluster_idx.unsqueeze(0) == local_self.unsqueeze(1)
            dist = dist.masked_fill(self_mask, float("inf"))
        # Sort clusters by distance per atom; keep top-K within radius.
        na, nc = dist.shape
        k = min(max_neighbors, nc)
        sorted_dist, sorted_local = torch.sort(dist, dim=-1)
        sorted_dist = sorted_dist[:, :k]
        sorted_local = sorted_local[:, :k]
        within = sorted_dist <= radius
        if not within.any():
            continue
        atom_local = (
            torch.arange(na, device=device).unsqueeze(1).expand(-1, k)[within]
        )
        local_c = sorted_local[within]
        src_list.append(cluster_idx[local_c])
        dst_list.append(atom_idx[atom_local])
        dist_list.append(sorted_dist[within])

    if not src_list:
        neighbors = torch.zeros(num_atoms, max(max_neighbors, 1), dtype=torch.long, device=device)
        empty = neighbors.new_empty(0)
        return FragmentNeighbors(neighbors, torch.zeros_like(neighbors, dtype=torch.bool), empty, empty, empty)

    src, dst = torch.cat(src_list), torch.cat(dst_list)
    selected, slots, neighbors, present = packed_neighbors(src, dst, torch.cat(dist_list), max_neighbors, num_atoms)
    return FragmentNeighbors(neighbors, present, dst[selected], src[selected], slots)


def resolve_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    fragment_mode: str = "kmeans",
    cluster_ids: Optional[Tensor] = None,
    cluster_centers: Optional[Tensor] = None,
    min_nodes_per_group: int = 24,
    random_state: int = 0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve fragment assignment for LSR.

    Fragment centers are always the mean of member atom positions in ``Ra``'s
    coordinate frame. Optional ``cluster_centers`` are ignored for geometry so
    absolute BRICS centers cannot be mixed with COM-centered atom positions.

    Returns ``(flat_cluster_ids, cluster_pos, cluster_batch, local_cluster_ids)``.
    """
    del cluster_centers  # membership from ids; centers always mean(Ra)
    mode = fragment_mode.lower()
    if mode == "precomputed":
        if cluster_ids is None:
            raise ValueError(
                "fragment_mode='precomputed' requires cluster_ids "
                "(e.g. offline BRICS labels)"
            )
        local_ids, cluster_pos, cluster_batch = centers_from_ids(
            Ra, cluster_ids.long(), batch_seg
        )
    elif mode == "kmeans":
        local_ids, _km_pos, _km_batch = build_kmeans_fragments(
            Ra,
            batch_seg,
            min_nodes_per_group=min_nodes_per_group,
            random_state=random_state,
        )
        # Recompute centers from current Ra so long-range geometry stays in the
        # autograd graph (sklearn centroids are detached).
        local_ids, cluster_pos, cluster_batch = centers_from_ids(
            Ra, local_ids, batch_seg
        )
    else:
        raise ValueError(
            f"Unknown fragment_mode={fragment_mode!r}; use 'kmeans' or 'precomputed'"
        )
    flat_ids = flatten_cluster_ids(local_ids, batch_seg, cluster_batch)
    return flat_ids, cluster_pos, cluster_batch, local_ids
