"""Directed edge triplets ``(k, j, i)`` for DimeNet message passing."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def directed_triplets(
    idx_i: Tensor,
    idx_j: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build ``k → j → i`` triplets on a directed neighbor graph.

    Enerzyme edges use ``idx_i`` = receiver, ``idx_j`` = sender (message ``j→i``).
    For each edge ``j→i``, collect incoming edges ``k→j`` with ``k ≠ i``.

    Returns:
        idx_kj: edge indices of ``k→j``
        idx_ji: edge indices of ``j→i``
        triplet_i, triplet_j, triplet_k: atom indices of each triplet
    """
    device = idx_i.device
    num_edges = idx_i.numel()
    if num_edges == 0:
        empty = idx_i.new_empty(0)
        return empty, empty, empty, empty, empty

    counts = torch.bincount(idx_i, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, dtype=torch.long, device=device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    perm = torch.argsort(idx_i, stable=True)

    n_trip = counts[idx_j]
    total = int(n_trip.sum().item())
    if total == 0:
        empty = idx_i.new_empty(0)
        return empty, empty, empty, empty, empty

    edge_ids = torch.arange(num_edges, device=device)
    idx_ji = torch.repeat_interleave(edge_ids, n_trip)
    starts = offsets[idx_j]
    prefix = torch.cumsum(n_trip, dim=0) - n_trip
    local = torch.arange(total, device=device) - torch.repeat_interleave(prefix, n_trip)
    idx_kj = perm[torch.repeat_interleave(starts, n_trip) + local]

    triplet_i = idx_i[idx_ji]
    triplet_j = idx_j[idx_ji]
    triplet_k = idx_j[idx_kj]
    mask = triplet_k != triplet_i
    return (
        idx_kj[mask],
        idx_ji[mask],
        triplet_i[mask],
        triplet_j[mask],
        triplet_k[mask],
    )


def triplet_angles(
    vij: Tensor,
    idx_kj: Tensor,
    idx_ji: Tensor,
) -> Tensor:
    """Angle at atom ``j`` between directed messages ``k→j`` and ``j→i``.

    ``vij[e] = R[sender] - R[receiver]`` from :class:`DistanceLayer`.
    The paper / DimeNet++ convention (not the official pretrained TF bug)::

        vec_{j→i} = R_i - R_j = -vij[ji]
        vec_{j→k} = R_k - R_j =  vij[kj]
        α = atan2(|vec_ji × vec_jk|, vec_ji · vec_jk)
    """
    if idx_kj.numel() == 0:
        return vij.new_zeros(0)
    vec_ji = -vij[idx_ji]
    vec_jk = vij[idx_kj]
    cosine_like = (vec_ji * vec_jk).sum(dim=-1)
    sine_like = torch.linalg.cross(vec_ji, vec_jk, dim=-1).norm(dim=-1)
    return torch.atan2(sine_like, cosine_like)
