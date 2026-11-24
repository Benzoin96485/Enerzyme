"""Fragment assignment shared by long–short range Cores.

E2Former-LSR and ViSNet-LSRM both cut a molecule into fragments, then pass
messages on the atom–fragment bipartite graph. Assignment (k-means, spectral
clustering, or precomputed BRICS ids) lives here. Each architecture still owns
its bipartite update.

Centers returned by the clustering helpers are detached sklearn centroids.
Callers that need forces recompute centers from the live coordinates.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def num_clusters_for_graph(num_atoms: int, min_nodes_per_group: int) -> int:
    """Fragment count used by the online clustering fallbacks.

    ``num_atoms / min(min_nodes, num_atoms // 2 + 1) + 1``, at least one
    when the graph is non-empty, and never more than ``num_atoms``.
    """
    if num_atoms <= 0:
        return 0
    min_nodes = max(int(min_nodes_per_group), 1)
    denom = min(min_nodes, max(num_atoms // 2, 1) + 1)
    count = int(num_atoms / denom) + 1
    return max(min(count, num_atoms), 1)


@torch.library.custom_op("enerzyme::fit_fragments", mutates_args=())
def _fit_fragments(positions: Tensor, n_clusters: int, seed: int, spectral: bool) -> Tuple[Tensor, Tensor]:
    """Fit discrete sklearn labels on detached coordinates, outside tensor autodiff."""
    try:
        from sklearn.cluster import KMeans, SpectralClustering
    except ImportError as exc:
        raise ImportError("Online fragment assignment requires scikit-learn. "
                          "Supply precomputed cluster_ids for offline fragments.") from exc

    xyz = positions.cpu().numpy()
    if spectral:
        model = SpectralClustering(
            n_clusters=n_clusters, random_state=seed, affinity="nearest_neighbors",
            n_neighbors=min(10, len(xyz) - 1), eigen_solver="arpack", assign_labels="kmeans",
        )
        ids = torch.as_tensor(model.fit_predict(xyz), dtype=torch.long, device=positions.device)
        centers = []
        for cluster in range(n_clusters):
            members = positions[ids == cluster]
            centers.append((members if members.shape[0] else positions).mean(dim=0))
        return ids, torch.stack(centers)
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    ids = torch.as_tensor(model.fit_predict(xyz), dtype=torch.long, device=positions.device)
    centers = torch.as_tensor(model.cluster_centers_, dtype=positions.dtype, device=positions.device)
    return ids, centers


@_fit_fragments.register_fake
def _fit_fragments_shape(positions, n_clusters, seed, spectral):
    return positions.new_empty(positions.shape[0], dtype=torch.long), positions.new_empty((n_clusters, 3))


def build_kmeans_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    min_nodes_per_group: int = 24,
    random_state: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-graph k-means fragments.

    Returns
    -------
    local_cluster_ids : LongTensor [N]
        Cluster id within each graph (0 .. n_c-1).
    cluster_pos : Tensor [U, 3]
        Flattened cluster centers across the batch. These centroids are
        detached; recompute them from ``Ra`` when the center must be in the
        autograd graph.
    cluster_batch : LongTensor [U]
        Graph id for each cluster.
    """
    device = Ra.device
    dtype = Ra.dtype
    num_atoms = Ra.shape[0]
    local_ids = torch.zeros(num_atoms, dtype=torch.long, device=device)
    centers = []
    cluster_batch = []

    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms else 0
    for graph in range(num_graphs):
        mask = batch_seg == graph
        pos = Ra[mask]
        n_atoms = int(pos.shape[0])
        if n_atoms == 0:
            continue
        n_clusters = num_clusters_for_graph(n_atoms, min_nodes_per_group)
        if n_clusters == 1:
            ids = torch.zeros(n_atoms, dtype=torch.long)
            center = pos.mean(dim=0, keepdim=True)
        else:
            ids, center = _fit_fragments(pos.detach(), n_clusters, random_state, False)
        local_ids[mask] = ids.to(device=device)
        centers.append(center.to(device=device, dtype=dtype))
        cluster_batch.append(
            torch.full((center.shape[0],), graph, dtype=torch.long, device=device)
        )

    if centers:
        cluster_pos = torch.cat(centers, dim=0)
        cluster_batch_t = torch.cat(cluster_batch, dim=0)
    else:
        cluster_pos = Ra.new_zeros((0, 3))
        cluster_batch_t = torch.zeros(0, dtype=torch.long, device=device)
    return local_ids, cluster_pos, cluster_batch_t


def build_spectral_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    min_nodes_per_group: int = 24,
    random_state: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Per-graph spectral clustering on a nearest-neighbor affinity.

    Same return contract as :func:`build_kmeans_fragments`. The released
    LSR-MP ``group_builder=spectral`` path uses this affinity. Graphs too
    small for a nearest-neighbor Laplacian collapse to one fragment.
    """
    device = Ra.device
    dtype = Ra.dtype
    num_atoms = Ra.shape[0]
    local_ids = torch.zeros(num_atoms, dtype=torch.long, device=device)
    centers = []
    cluster_batch = []

    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms else 0
    for graph in range(num_graphs):
        mask = batch_seg == graph
        pos = Ra[mask]
        n_atoms = int(pos.shape[0])
        if n_atoms == 0:
            continue
        n_clusters = num_clusters_for_graph(n_atoms, min_nodes_per_group)
        if n_clusters == 1 or n_atoms < 4:
            ids = torch.zeros(n_atoms, dtype=torch.long)
            center = pos.mean(dim=0, keepdim=True)
        else:
            ids, center = _fit_fragments(pos.detach(), n_clusters, random_state, True)
        local_ids[mask] = ids.to(device=device)
        centers.append(center.to(device=device, dtype=dtype))
        cluster_batch.append(
            torch.full((center.shape[0],), graph, dtype=torch.long, device=device)
        )

    if centers:
        return local_ids, torch.cat(centers, dim=0), torch.cat(cluster_batch, dim=0)
    return (
        local_ids,
        Ra.new_zeros((0, 3)),
        torch.zeros(0, dtype=torch.long, device=device),
    )


def centers_from_ids(
    Ra: Tensor,
    local_cluster_ids: Tensor,
    batch_seg: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mean fragment centers from per-graph cluster ids.

    Ids are remapped to a dense ``0 .. n_c-1`` range inside each graph.
    The returned centers are means of ``Ra`` and stay in the autograd graph.
    """
    device = Ra.device
    dtype = Ra.dtype
    num_atoms = Ra.shape[0]
    num_graphs = int(batch_seg.max().item()) + 1 if num_atoms else 0
    centers = []
    cluster_batch = []
    remapped = local_cluster_ids.clone()
    for graph in range(num_graphs):
        mask = batch_seg == graph
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
        n_clusters = int(uniq.numel())
        flat = remapped[mask]
        sum_pos = Ra.new_zeros((n_clusters, 3))
        counts = Ra.new_zeros((n_clusters, 1))
        sum_pos.index_add_(0, flat, pos)
        counts.index_add_(
            0, flat, torch.ones(pos.shape[0], 1, device=device, dtype=dtype)
        )
        centers.append(sum_pos / counts.clamp_min(1.0))
        cluster_batch.append(
            torch.full((n_clusters,), graph, dtype=torch.long, device=device)
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


def weighted_fragment_centers(
    Ra: Tensor,
    flat_cluster_ids: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Weighted mean of ``Ra`` on flat fragment ids.

    ``weights is None`` is the uniform mean. A fragment whose weights sum to
    zero falls back to that uniform mean. The result follows ``Ra``.
    """
    num_atoms = int(Ra.shape[0])
    if num_atoms == 0 or flat_cluster_ids.numel() == 0:
        return Ra.new_zeros((0, 3))
    n_fragments = int(flat_cluster_ids.max().item()) + 1
    ids = flat_cluster_ids.long()
    uniform = Ra.new_zeros((n_fragments, 3))
    counts = Ra.new_zeros((n_fragments, 1))
    uniform.index_add_(0, ids, Ra)
    counts.index_add_(
        0, ids, torch.ones(num_atoms, 1, device=Ra.device, dtype=Ra.dtype)
    )
    uniform = uniform / counts.clamp_min(1.0)
    if weights is None:
        return uniform
    weight = weights.reshape(-1).to(dtype=Ra.dtype, device=Ra.device)
    if weight.shape[0] != num_atoms:
        raise ValueError(
            f"fragment weights must have length {num_atoms}, got {weight.shape[0]}"
        )
    weighted = Ra.new_zeros((n_fragments, 3))
    sum_w = Ra.new_zeros((n_fragments, 1))
    weighted.index_add_(0, ids, Ra * weight.unsqueeze(-1))
    sum_w.index_add_(0, ids, weight.unsqueeze(-1))
    mixed = weighted / sum_w.clamp_min(1e-12)
    empty = sum_w.squeeze(-1) <= 0
    return torch.where(empty.unsqueeze(-1), uniform, mixed)
