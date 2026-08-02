# Adapted from IQuestLab/UBio-MolFM tag E2Former-LSR (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""Fragment construction and atom–fragment bipartite neighborhoods for E2Former-LSR."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_scatter import scatter_mean

from .graph import map_neighbor_list, pad_neighbor_list, select_closest_neighbors


def _num_clusters_for_graph(num_atoms: int, min_nodes_per_group: int) -> int:
    if num_atoms <= 0:
        return 0
    denom = min(min_nodes_per_group, max(num_atoms // 2, 1) + 1)
    return int(num_atoms / denom) + 1


def build_kmeans_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    min_nodes_per_group: int = 24,
    random_state: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-graph k-means fragments (upstream E2FormerCluster default).

    Returns
    -------
    local_cluster_ids : LongTensor [N]
        Cluster id within each graph (0 .. n_c-1).
    cluster_pos : Tensor [U, 3]
        Flattened cluster centers across the batch.
    cluster_batch : LongTensor [U]
        Graph id for each cluster.
    """
    from sklearn.cluster import KMeans

    device = Ra.device
    dtype = Ra.dtype
    num_atoms = Ra.shape[0]
    local_ids = torch.zeros(num_atoms, dtype=torch.long, device=device)
    centers = []
    cluster_batch = []

    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms else 0
    for g in range(num_graphs):
        mask = batch_seg == g
        pos = Ra[mask]
        n = int(pos.shape[0])
        if n == 0:
            continue
        n_c = max(_num_clusters_for_graph(n, min_nodes_per_group), 1)
        n_c = min(n_c, n)
        if n_c == 1:
            ids_np = torch.zeros(n, dtype=torch.long)
            center = pos.mean(dim=0, keepdim=True)
        else:
            km = KMeans(n_clusters=n_c, random_state=random_state, n_init=10)
            # sklearn / numpy path; detach so fragment assignment is not differentiated
            ids_np = torch.as_tensor(
                km.fit_predict(pos.detach().cpu().numpy()), dtype=torch.long
            )
            center = torch.as_tensor(km.cluster_centers_, dtype=dtype)
        local_ids[mask] = ids_np.to(device=device)
        centers.append(center.to(device=device, dtype=dtype))
        cluster_batch.append(
            torch.full((center.shape[0],), g, dtype=torch.long, device=device)
        )

    if centers:
        cluster_pos = torch.cat(centers, dim=0)
        cluster_batch_t = torch.cat(cluster_batch, dim=0)
    else:
        cluster_pos = Ra.new_zeros((0, 3))
        cluster_batch_t = torch.zeros(0, dtype=torch.long, device=device)
    return local_ids, cluster_pos, cluster_batch_t


def centers_from_ids(
    Ra: Tensor,
    local_cluster_ids: Tensor,
    batch_seg: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mean fragment centers from precomputed per-graph cluster ids (BRICS path)."""
    device = Ra.device
    dtype = Ra.dtype
    num_atoms = Ra.shape[0]
    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms else 0
    centers = []
    cluster_batch = []
    # Remap local ids in case they are not dense 0..n_c-1 within a graph
    remapped = local_cluster_ids.clone()
    for g in range(num_graphs):
        mask = batch_seg == g
        ids = local_cluster_ids[mask]
        if ids.numel() == 0:
            continue
        uniq = torch.unique(ids, sorted=True)
        remap = torch.full(
            (int(ids.max().item()) + 1,), -1, dtype=torch.long, device=device
        )
        remap[uniq] = torch.arange(uniq.numel(), device=device, dtype=torch.long)
        remapped[mask] = remap[ids]
        pos = Ra[mask]
        n_c = int(uniq.numel())
        flat = remapped[mask]
        sum_pos = Ra.new_zeros((n_c, 3))
        counts = Ra.new_zeros((n_c, 1))
        sum_pos.index_add_(0, flat, pos)
        counts.index_add_(
            0, flat, torch.ones(pos.shape[0], 1, device=device, dtype=dtype)
        )
        centers.append(sum_pos / counts.clamp_min(1.0))
        cluster_batch.append(
            torch.full((n_c,), g, dtype=torch.long, device=device)
        )
    if centers:
        return remapped, torch.cat(centers, dim=0), torch.cat(cluster_batch, dim=0)
    return remapped, Ra.new_zeros((0, 3)), torch.zeros(0, dtype=torch.long, device=device)


def flatten_cluster_ids(
    local_cluster_ids: Tensor,
    batch_seg: Tensor,
    cluster_batch: Tensor,
) -> Tensor:
    """Map per-graph local cluster ids to global flat indices ``[0, U)``."""
    if local_cluster_ids.numel() == 0:
        return local_cluster_ids
    device = local_cluster_ids.device
    num_graphs = int(batch_seg.max().item()) + 1
    counts = torch.bincount(cluster_batch, minlength=num_graphs)
    offsets = torch.zeros(num_graphs, dtype=torch.long, device=device)
    if num_graphs > 1:
        offsets[1:] = torch.cumsum(counts, dim=0)[:-1]
    return local_cluster_ids + offsets[batch_seg]


def pool_fragment_irreps(node_irreps: Tensor, flat_cluster_ids: Tensor) -> Tensor:
    """Mean-pool atom SH features onto fragments: ``[N,M,C] → [U,M,C]``."""
    n, m, c = node_irreps.shape
    flat = node_irreps.reshape(n, m * c)
    pooled = scatter_mean(flat, flat_cluster_ids.long(), dim=0)
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
    device = atom_pos.device
    dtype = atom_pos.dtype
    num_atoms = atom_pos.shape[0]
    num_clusters = cluster_pos.shape[0]

    if num_atoms == 0 or num_clusters == 0 or max_neighbors <= 0:
        k = max(max_neighbors, 1)
        empty_idx = torch.zeros(num_atoms, k, dtype=torch.long, device=device)
        present = torch.zeros(num_atoms, k, dtype=torch.bool, device=device)
        return {
            "f_sparse_idx_expnode": empty_idx,
            "f_sparse_idx_node": empty_idx,
            "edge_vec": atom_pos.new_zeros(num_atoms, k, 3),
            "edge_dis": atom_pos.new_zeros(num_atoms, k),
            "attn_mask": (~present).unsqueeze(-1),
            "present": present,
            "f_cluster_pos": cluster_pos,
            "max_neighbors": torch.tensor(k, device=device),
        }

    src_list = []
    dst_list = []
    dist_list = []
    vec_list = []

    num_graphs = int(batch_seg.max().item()) + 1
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
        vec_list.append(delta[atom_local, local_c])

    if not src_list:
        k = max_neighbors
        empty_idx = torch.zeros(num_atoms, k, dtype=torch.long, device=device)
        present = torch.zeros(num_atoms, k, dtype=torch.bool, device=device)
        return {
            "f_sparse_idx_expnode": empty_idx,
            "f_sparse_idx_node": empty_idx,
            "edge_vec": atom_pos.new_zeros(num_atoms, k, 3),
            "edge_dis": atom_pos.new_zeros(num_atoms, k),
            "attn_mask": (~present).unsqueeze(-1),
            "present": present,
            "f_cluster_pos": cluster_pos,
            "max_neighbors": torch.tensor(k, device=device),
        }

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    dist = torch.cat(dist_list)
    vec = torch.cat(vec_list)

    src, dst, dist, vec = select_closest_neighbors(
        src, dst, dist, max_neighbors, num_atoms, vec
    )
    neighbor_list, present, index_mapping = pad_neighbor_list(
        src, dst, max_neighbors, num_atoms
    )
    attn_mask = (~present).unsqueeze(-1)
    edge_vec = map_neighbor_list(vec, index_mapping, max_neighbors, num_atoms)
    edge_dis = map_neighbor_list(dist, index_mapping, max_neighbors, num_atoms)
    edge_dis = edge_dis.masked_fill(~present, 0.0)
    safe = neighbor_list.clamp(min=0)
    return {
        "f_sparse_idx_expnode": safe,
        "f_sparse_idx_node": safe,
        "edge_vec": edge_vec,
        "edge_dis": edge_dis,
        "attn_mask": attn_mask,
        "present": present,
        "f_cluster_pos": cluster_pos,
        "f_exp_node_pos": cluster_pos,
        "max_neighbors": torch.tensor(max_neighbors, device=device),
    }


def resolve_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    fragment_mode: str = "kmeans",
    cluster_ids: Optional[Tensor] = None,
    cluster_centers: Optional[Tensor] = None,
    min_nodes_per_group: int = 24,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve fragment assignment for LSR.

    Returns ``(flat_cluster_ids, cluster_pos, cluster_batch, local_cluster_ids)``.
    """
    mode = fragment_mode.lower()
    if mode == "precomputed":
        if cluster_ids is None:
            raise ValueError(
                "fragment_mode='precomputed' requires cluster_ids "
                "(e.g. offline BRICS labels)"
            )
        local_ids = cluster_ids.long()
        if cluster_centers is not None:
            # Assume centers are already flattened with matching batch layout via
            # consecutive per-graph packing; rebuild batch from counts.
            remapped, computed_pos, cluster_batch = centers_from_ids(
                Ra, local_ids, batch_seg
            )
            # Prefer provided centers when shapes match; else fall back to means.
            if cluster_centers.shape[0] == computed_pos.shape[0]:
                cluster_pos = cluster_centers.to(dtype=Ra.dtype, device=Ra.device)
            else:
                cluster_pos = computed_pos
            local_ids = remapped
        else:
            local_ids, cluster_pos, cluster_batch = centers_from_ids(
                Ra, local_ids, batch_seg
            )
    elif mode == "kmeans":
        local_ids, _km_pos, _km_batch = build_kmeans_fragments(
            Ra, batch_seg, min_nodes_per_group=min_nodes_per_group
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
