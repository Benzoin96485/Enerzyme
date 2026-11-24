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
    cross_receiver_pairs,
    receiver_csr,
    receiver_pairs,
    receiver_rank,
    reverse_edge_index,
)


_INV_SQRT_EPS = 1e-12
# A true reverse edge has displacement -v, so the residual is identically zero.
# Anything larger is a different periodic image and must not be paired.
_OPPOSITE_TOL = 1e-4


class TripletGraph(NamedTuple):
    """Pairs of incoming embedding edges that share a receiver."""

    idx_edge: Tensor
    idx_other: Tensor
    angle: Tensor


class QuadrupletGraph(NamedTuple):
    """Quadruplets ``c → a − b ← d`` on one neighbor list."""

    edge_ca: Tensor
    edge_db: Tensor
    int_local: Tensor
    angle_cab: Tensor
    angle_abd: Tensor
    dihedral: Tensor
    atom_c: Tensor
    atom_a: Tensor
    atom_b: Tensor
    atom_d: Tensor


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


def angle_between(left: Tensor, right: Tensor) -> Tensor:
    """Unsigned angle in ``[0, π]`` between row vectors."""
    if left.numel() == 0:
        return left.new_zeros(0)
    sine = torch.linalg.cross(left, right, dim=-1).norm(dim=-1)
    sine = torch.maximum(sine, sine.new_tensor(1e-9))
    cosine = (left * right).sum(dim=-1)
    return torch.atan2(sine, cosine)


def _reject(vec: Tensor, normal: Tensor) -> Tensor:
    denom = (normal * normal).sum(dim=-1, keepdim=True).clamp(min=_INV_SQRT_EPS)
    coef = (vec * normal).sum(dim=-1, keepdim=True) / denom
    return vec - coef * normal


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


def _distinct_image(atom_a: Tensor, vec_a: Tensor, atom_b: Tensor, vec_b: Tensor) -> Tensor:
    """True when the two neighbor images are not the same atom in the same cell."""
    gap = (vec_a - vec_b).detach().square().sum(dim=-1).sqrt()
    return (atom_a != atom_b) | (gap > _OPPOSITE_TOL)


def quadruplet_graph(
    idx_i: Tensor,
    idx_j: Tensor,
    dist: Tensor,
    vij: Tensor,
    emb_index: Tensor,
    int_index: Tensor,
    num_nodes: int,
    images: bool = False,
) -> QuadrupletGraph:
    """Build ``c → a − b ← d`` quadruplets.

    ``edge_ca`` and ``edge_db`` index the embedding-edge subset. ``int_local``
    indexes ``int_index``. ``images=False`` drops any repeated atom index.
    ``images=True`` keeps a second periodic image of the same atom when the
    displacements differ. GemNet-Q uses the default. GemNet-OC uses ``True``.
    """
    empty_l = idx_i.new_empty(0)
    empty_f = vij.new_zeros(0)
    blank = QuadrupletGraph(
        empty_l, empty_l, empty_l, empty_f, empty_f, empty_f,
        empty_l, empty_l, empty_l, empty_l,
    )
    if emb_index.numel() == 0 or int_index.numel() == 0:
        return blank

    idx_i_emb = idx_i[emb_index]
    idx_j_emb = idx_j[emb_index]
    counts, offsets, perm = receiver_csr(idx_i_emb, num_nodes)
    a_node = idx_i[int_index]
    b_node = idx_j[int_index]
    n_a = counts[a_node]
    n_b = counts[b_node]
    n_quad = n_a * n_b
    total = int(n_quad.sum().item())
    if total == 0:
        return blank

    device = idx_i.device
    int_local = torch.repeat_interleave(torch.arange(int_index.numel(), device=device), n_quad)
    prefix = torch.cumsum(n_quad, dim=0) - n_quad
    slot = torch.arange(total, device=device) - torch.repeat_interleave(prefix, n_quad)
    n_b_rep = torch.repeat_interleave(n_b, n_quad)
    c_local = torch.div(slot, n_b_rep, rounding_mode="floor")
    d_local = slot - c_local * n_b_rep
    a_rep = torch.repeat_interleave(a_node, n_quad)
    b_rep = torch.repeat_interleave(b_node, n_quad)
    edge_ca = perm[offsets[a_rep] + c_local]
    edge_db = perm[offsets[b_rep] + d_local]
    atom_c = idx_j_emb[edge_ca]
    atom_d = idx_j_emb[edge_db]
    if images:
        vij_emb = vij[emb_index]
        vij_ab = vij[int_index][int_local]
        vij_ca = vij_emb[edge_ca]
        vij_db = vij_emb[edge_db]
        keep = (
            _distinct_image(atom_c, vij_ca, b_rep, vij_ab)
            & _distinct_image(a_rep, -vij_ab, atom_d, vij_db)
            & _distinct_image(atom_c, vij_ca, atom_d, vij_ab + vij_db)
        )
    else:
        keep = (atom_c != b_rep) & (a_rep != atom_d) & (atom_c != atom_d)
    edge_ca = edge_ca[keep]
    edge_db = edge_db[keep]
    int_local = int_local[keep]
    atom_c = atom_c[keep]
    atom_a = a_rep[keep]
    atom_b = b_rep[keep]
    atom_d = atom_d[keep]
    if edge_ca.numel() == 0:
        return blank

    vij_emb = vij[emb_index]
    vij_ab = vij[int_index[int_local]]
    vij_ca = vij_emb[edge_ca]
    vij_db = vij_emb[edge_db]
    angle_cab = angle_between(vij_ca, vij_ab)
    angle_abd = angle_between(-vij_ab, vij_db)
    dihedral = angle_between(_reject(vij_ca, vij_ab), _reject(vij_db, -vij_ab))
    return QuadrupletGraph(
        edge_ca, edge_db, int_local, angle_cab, angle_abd, dihedral,
        atom_c, atom_a, atom_b, atom_d,
    )


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
    local_d = distance[within]
    counts = torch.bincount(local_i, minlength=int(num_nodes))
    if int(counts.max().item()) <= int(max_neighbors):
        return within
    chosen = receiver_rank(local_i, local_d, int(num_nodes)) < int(max_neighbors)
    mask = torch.zeros(idx_i.shape[0], dtype=torch.bool, device=idx_i.device)
    mask[torch.where(within)[0][chosen]] = True
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


def symmetrize_edges(
    idx_i: Tensor,
    idx_j: Tensor,
    dist: Tensor,
    vij: Tensor,
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
    same = idx_j == idx_i
    canonical = (idx_j < idx_i) | (same & _lexicographic_negative(vij))
    if bool(same.any()):
        zero = same & (vij.detach().square().sum(dim=-1).sqrt() <= _OPPOSITE_TOL)
        canonical = canonical & ~zero
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
    empty = out_i.new_empty(0)
    blank = MixedTriplets(empty, empty, out_v.new_zeros(0))
    if out_i.numel() == 0 or in_i.numel() == 0 or int(num_nodes) == 0:
        return blank
    idx_edge, idx_other = cross_receiver_pairs(out_i, in_i, num_nodes)
    if idx_edge.numel() == 0:
        return blank
    same = in_j[idx_other] == out_j[idx_edge]
    if bool(same.any()):
        residual = (in_v[idx_other] - out_v[idx_edge]).detach().square().sum(dim=-1).sqrt()
        same = same & (residual <= _OPPOSITE_TOL)
    keep = ~same
    idx_edge = idx_edge[keep]
    idx_other = idx_other[keep]
    angle = angle_between(out_v[idx_edge], in_v[idx_other])
    return MixedTriplets(idx_edge, idx_other, angle)
