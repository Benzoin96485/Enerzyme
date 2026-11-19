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


_INV_SQRT_EPS = 1e-12


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


def reverse_edge_index(
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    num_nodes: int,
) -> Tensor:
    """Index of the opposite edge, matched by ``-vij`` when images repeat.

    GemNet adds each directed interaction onto both orientations. The neighbor
    list has to contain both directions.
    """
    num_edges = int(idx_i.numel())
    if num_edges == 0:
        return idx_i.new_empty(0)
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive when the graph has edges")
    if int(idx_i.max()) >= num_nodes or int(idx_j.max()) >= num_nodes:
        raise ValueError("edge index is outside num_nodes")
    if int(idx_i.min()) < 0 or int(idx_j.min()) < 0:
        raise ValueError("edge index must be non-negative")

    key = idx_i * num_nodes + idx_j
    rev = idx_j * num_nodes + idx_i
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    start = torch.searchsorted(sorted_key, rev, right=False)
    end = torch.searchsorted(sorted_key, rev, right=True)
    if torch.any(end == start):
        raise ValueError(
            "GemNet requires both directions of every edge in the neighbor list"
        )
    width = int((end - start).max().item())
    if width == 1:
        return order[start]

    device = idx_i.device
    offsets = torch.arange(width, device=device)
    pos = start[:, None] + offsets
    valid = offsets[None, :] < (end - start)[:, None]
    cand = order[pos.clamp(max=num_edges - 1)]
    mismatch = (vij[cand] + vij[:, None, :]).square().sum(dim=-1)
    mismatch = mismatch.masked_fill(~valid, float("inf"))
    best = mismatch.argmin(dim=1)
    if not torch.isfinite(mismatch[torch.arange(num_edges, device=device), best]).all():
        raise ValueError(
            "GemNet could not match every edge to an opposite displacement"
        )
    return cand[torch.arange(num_edges, device=device), best]


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


def _incoming_csr(receivers: Tensor, num_nodes: int) -> Tuple[Tensor, Tensor, Tensor]:
    counts = torch.bincount(receivers, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, dtype=torch.long, device=receivers.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    perm = torch.argsort(receivers, stable=True)
    return counts, offsets, perm


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

    counts, offsets, perm = _incoming_csr(idx_i, num_nodes)
    local = torch.empty(num_edges, dtype=torch.long, device=idx_i.device)
    local[perm] = torch.arange(num_edges, device=idx_i.device) - offsets[idx_i[perm]]
    n_other = counts[idx_i] - 1
    total = int(n_other.sum().item())
    if total == 0:
        return TripletGraph(empty, empty, vij.new_zeros(0))

    idx_edge = torch.repeat_interleave(torch.arange(num_edges, device=idx_i.device), n_other)
    prefix = torch.cumsum(n_other, dim=0) - n_other
    slot = torch.arange(total, device=idx_i.device) - torch.repeat_interleave(prefix, n_other)
    self_local = local[idx_edge]
    other_local = slot + (slot >= self_local).to(slot.dtype)
    starts = offsets[idx_i[idx_edge]]
    idx_other = perm[starts + other_local]
    keep = idx_j[idx_other] != idx_j[idx_edge]
    idx_edge = idx_edge[keep]
    idx_other = idx_other[keep]
    angle = angle_between(vij[idx_edge], vij[idx_other])
    return TripletGraph(idx_edge, idx_other, angle)


def quadruplet_graph(
    idx_i: Tensor,
    idx_j: Tensor,
    dist: Tensor,
    vij: Tensor,
    emb_index: Tensor,
    int_index: Tensor,
    num_nodes: int,
) -> QuadrupletGraph:
    """Build ``c → a − b ← d`` quadruplets.

    ``edge_ca`` and ``edge_db`` index the embedding-edge subset. ``int_local``
    indexes ``int_index``.
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
    counts, offsets, perm = _incoming_csr(idx_i_emb, num_nodes)
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
