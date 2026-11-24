"""Directed-edge geometry for GemNet (Gasteiger, Becker, Günnemann, NeurIPS 2021).

Enerzyme stores a receiver/sender pair ``(idx_i, idx_j)`` with
``vij = R[sender] - R[receiver]``. GemNet's triplet is two incoming edges
that meet at the receiver. The quadruplet is ``c → a − b ← d``: ``c→a`` and
``d→b`` lie inside the embedding cutoff, and the central bond ``a−b`` lies
inside the interaction cutoff.

Angles use the unsigned ``atan2(|u × v|, u · v)`` convention, so the dihedral
lies in ``[0, π]``.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
from torch import Tensor

from ..mp.graph import (  # noqa: F401
    GraphContext,
    cross_receiver_pairs,
    receiver_csr,
    receiver_pairs,
    receiver_rank,
    reverse_edge_index,
)


from ..mp.graph import _OPPOSITE_TOL
from ..mp.quadruplets import QuadrupletGraph, angle_between


class TripletGraph(NamedTuple):
    """Pairs of incoming embedding edges that share a receiver."""

    idx_edge: Tensor
    idx_other: Tensor
    angle: Tensor


def split_cutoff_edges(
    dist: Tensor,
    emb_cutoff: float,
    int_cutoff: float,
    *,
    triplets_only: bool,
) -> Tuple[Tensor, Tensor]:
    """Indices of embedding edges and, for GemNet-Q, interaction edges.

    Both selections use a strict ``<`` cutoff, matching
    :class:`~enerzyme.models.layers.geometry.RangeSeparationLayer`.
    """
    emb = torch.where(dist < emb_cutoff)[0]
    if triplets_only:
        return emb, emb
    inter = torch.where(dist < int_cutoff)[0]
    return emb, inter


def triplet_graph(
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    num_nodes: int,
) -> TripletGraph:
    """Incoming pairs ``k → i ← j`` with the angle at the receiver ``i``.

    ``idx_edge`` is the edge being updated (``j→i``). ``idx_other`` is the
    other incoming edge (``k→i``). Edges that share a sender atom are dropped,
    which removes the edge itself and any other image of that neighbor.
    """
    num_edges = int(idx_i.numel())
    empty = idx_i.new_empty(0)
    if num_edges == 0:
        return TripletGraph(empty, empty, vij.new_zeros(0))

    pairs = receiver_pairs(idx_i, num_nodes)
    idx_edge, idx_other = pairs.first, pairs.second
    keep = idx_j[idx_other] != idx_j[idx_edge]
    idx_edge = idx_edge[keep]
    idx_other = idx_other[keep]
    angle = angle_between(vij[idx_edge], vij[idx_other])
    return TripletGraph(idx_edge, idx_other, angle)


def quadruplet_graph(idx_i, idx_j, dist, vij, emb_index, int_index, num_nodes, images=False) -> QuadrupletGraph:
    """Dihedral indices and angles for embedding and interaction edge selections."""
    ctx = GraphContext(idx_i, idx_j, num_nodes, vec=vij, dist=dist)
    return ctx.subset(emb_index).quadruplets(ctx.subset(int_index), images=images)


class MixedTriplets(NamedTuple):
    """Incoming pairs drawn from two edge lists that share a receiver.

    ``idx_edge`` indexes the output list. ``idx_other`` indexes the input
    list. A pair is dropped when it is the same neighbor image.
    """

    idx_edge: Tensor
    idx_other: Tensor
    angle: Tensor


def cap_incoming_mask(
    idx_i: Tensor,
    distance: Tensor,
    num_nodes: int,
    max_neighbors: int,
    cutoff: float,
) -> Tensor:
    """Keep the closest incoming edges inside a strict cutoff.

    Edges at or beyond ``cutoff`` are removed first, so they do not consume
    a neighbor slot. Each receiver then keeps ``max_neighbors`` edges.
    Ties keep the earlier edge. The mask is true for every edge when no
    receiver exceeds the cap.
    """
    if int(max_neighbors) < 1:
        raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")
    if float(cutoff) <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if idx_i.shape != distance.shape:
        raise ValueError(
            "idx_i and distance must share a shape, got "
            f"{tuple(idx_i.shape)} and {tuple(distance.shape)}"
        )
    within = distance < float(cutoff)
    if idx_i.numel() == 0 or int(num_nodes) == 0 or not bool(within.any()):
        return within
    local_i = idx_i[within]
    if int(local_i.min()) < 0 or int(local_i.max()) >= int(num_nodes):
        raise ValueError("receiver index is outside num_nodes")
    graph = GraphContext(idx_i, None, num_nodes, dist=distance)
    mask = torch.zeros_like(within)
    mask[graph.closest_indices(cutoff, max_neighbors)] = True
    return mask


def _lexicographic_negative(vij: Tensor) -> Tensor:
    """True when the first nonzero component of each row is negative."""
    if vij.numel() == 0:
        return vij.new_zeros(vij.shape[0], dtype=torch.bool)
    detached = vij.detach()
    nonzero = detached != 0
    any_nonzero = nonzero.any(dim=-1)
    first = nonzero.to(dtype=torch.long).argmax(dim=-1)
    component = detached.gather(1, first.unsqueeze(-1)).squeeze(-1)
    return any_nonzero & (component < 0)


def canonical_edge_indices(idx_i: Tensor, idx_j: Tensor, vij: Tensor) -> Tensor:
    same = idx_j == idx_i
    canonical = idx_j < idx_i
    if bool(same.any()):
        length = vij.detach().square().sum(dim=-1).sqrt()
        image = _lexicographic_negative(vij) & ~(length <= _OPPOSITE_TOL)
        canonical = canonical | (same & image)
    return torch.where(canonical)[0]


def symmetrize_edges(
    idx_i: Tensor,
    idx_j: Tensor,
    dist: Tensor,
    vij: Tensor,
    canonical: Tensor | None = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mirror the canonical half of a directed neighbor list.

    ``idx_i`` is the receiver and ``idx_j`` is the sender. The canonical
    direction has a smaller sender index than receiver index. A periodic
    self-image is canonical when the first nonzero component of ``vij`` is
    negative. The reverse of each canonical edge is appended, and ``swap``
    maps each edge onto that reverse. A one-way edge whose sender index is
    larger than its receiver is dropped. That is the GemNet-OC symmetrization
    of a capped neighbor list.
    """
    empty = idx_i.new_empty(0)
    if idx_i.numel() == 0:
        return idx_i, idx_j, dist, vij, empty
    if idx_i.shape != idx_j.shape or idx_i.shape != dist.shape:
        raise ValueError("idx_i, idx_j, and dist must share a shape")
    if vij.ndim != 2 or vij.shape[0] != idx_i.shape[0] or vij.shape[-1] != 3:
        raise ValueError(f"vij must have shape ({idx_i.shape[0]}, 3), got {tuple(vij.shape)}")
    if canonical is None:
        canonical = canonical_edge_indices(idx_i, idx_j, vij)
    kept_i = idx_i[canonical]
    kept_j = idx_j[canonical]
    if kept_i.numel() == 0:
        return empty, empty, dist.new_empty(0), vij.new_empty(0, 3), empty
    kept_d = dist[canonical]
    kept_v = vij[canonical]
    n_kept = int(kept_i.numel())
    idx_i_s = torch.cat([kept_i, kept_j])
    idx_j_s = torch.cat([kept_j, kept_i])
    dist_s = torch.cat([kept_d, kept_d])
    vij_s = torch.cat([kept_v, -kept_v])
    swap = torch.cat(
        [
            torch.arange(n_kept, 2 * n_kept, device=idx_i.device),
            torch.arange(n_kept, device=idx_i.device),
        ]
    )
    return idx_i_s, idx_j_s, dist_s, vij_s, swap


def mixed_triplets(
    out_i: Tensor,
    out_j: Tensor,
    out_v: Tensor,
    in_i: Tensor,
    in_j: Tensor,
    in_v: Tensor,
    num_nodes: int,
) -> MixedTriplets:
    """Pair each output edge with the other incoming edges at its receiver.

    The two lists may be the same graph. A pair whose sender images coincide
    is the edge itself, or the same periodic image stored twice, and is
    dropped. A different image of the same atom is kept.
    """
    query = GraphContext(out_i, out_j, num_nodes, vec=out_v)
    keys = GraphContext(in_i, in_j, num_nodes, vec=in_v)
    return mixed_graph_triplets(query, keys)


def mixed_triplet_indices(query, keys) -> Tuple[Tensor, Tensor]:
    """Filter cross-graph receiver pairs by sender image."""
    idx_edge, idx_other = query.cross_pairs(keys)
    same = keys.sender[idx_other] == query.sender[idx_edge]
    if bool(same.any()):
        residual = (keys.vec[idx_other] - query.vec[idx_edge]).detach().norm(dim=-1)
        same = same & (residual <= _OPPOSITE_TOL)
    keep = ~same
    return idx_edge[keep], idx_other[keep]


def mixed_graph_triplets(query, keys, indices=None) -> MixedTriplets:
    """Compute each angle from its two current displacement vectors."""
    idx_edge, idx_other = mixed_triplet_indices(query, keys) if indices is None else indices
    angle = angle_between(query.vec[idx_edge], keys.vec[idx_other])
    return MixedTriplets(idx_edge, idx_other, angle)
