"""Undirected bonds and the CHGNet bond graph.

Enerzyme stores a directed edge as ``idx_i`` (receiver) and ``idx_j``
(sender), with ``vij = R[sender] - R[receiver]``. CHGNet keeps one feature
for a bond and its reverse, including the opposite periodic image. The bond
graph is the line graph of edges inside ``bond_cutoff``: two directed edges
that share a receiver. That index is the M3GNet receiver-triplet list, with
the reference code's strict cutoff on the second edge.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from ..m3gnet.geometry import ThreeBodyPairs, receiver_triplets


_PAIR_ATOL = 1e-4


class UndirectedBonds(NamedTuple):
    """``directed_to_undirected`` addresses the shared bond of each directed edge.

    ``representative`` is one directed edge of that bond. Its length is the
    length stored on the undirected bond.
    """

    directed_to_undirected: Tensor
    representative: Tensor


def undirected_bond_index(
    idx_i: Tensor,
    idx_j: Tensor,
    vij: Tensor,
    atol: float = _PAIR_ATOL,
) -> UndirectedBonds:
    """Pair each directed edge with the reverse edge ``(j, i, -vij)``.

    The neighbor list must contain both directions. A periodic image is a
    different bond from the in-cell pair when the displacements differ.
    """
    n_edges = int(idx_i.shape[0])
    empty = idx_i.new_empty(0)
    if n_edges == 0:
        return UndirectedBonds(empty, empty)
    if idx_j.shape[0] != n_edges or vij.shape[0] != n_edges or vij.shape[-1] != 3:
        raise ValueError(
            "idx_i, idx_j, and vij must describe the same directed edges, "
            f"got {tuple(idx_i.shape)}, {tuple(idx_j.shape)}, {tuple(vij.shape)}"
        )
    centers = idx_i.detach().to(dtype=torch.long).cpu().tolist()
    neighbors = idx_j.detach().to(dtype=torch.long).cpu().tolist()
    vectors = vij.detach().cpu()
    grouped: dict[tuple[int, int], list[int]] = {}
    for edge, (center, neighbor) in enumerate(zip(centers, neighbors)):
        grouped.setdefault((center, neighbor), []).append(edge)

    directed = [-1] * n_edges
    representative: list[int] = []
    used = [False] * n_edges
    tolerance = float(atol)
    for edge, (center, neighbor) in enumerate(zip(centers, neighbors)):
        if used[edge]:
            continue
        match = None
        for reverse in grouped.get((neighbor, center), ()):
            if used[reverse] or reverse == edge:
                continue
            gap = vectors[edge] + vectors[reverse]
            scale = vectors[edge].abs().max().clamp(min=1.0)
            if float(gap.abs().max()) <= tolerance + 1e-5 * float(scale):
                match = reverse
                break
        if match is None:
            raise ValueError(
                "CHGNet needs both directions of every bond, including the "
                f"opposite periodic image. Edge {edge} ({center} -> {neighbor}) "
                "has no reverse."
            )
        bond = len(representative)
        directed[edge] = bond
        directed[match] = bond
        used[edge] = True
        used[match] = True
        representative.append(edge)
    if any(not flag for flag in used):
        raise ValueError(
            "CHGNet could not pair every directed edge with its reverse"
        )
    return UndirectedBonds(
        torch.tensor(directed, dtype=torch.long, device=idx_i.device),
        torch.tensor(representative, dtype=torch.long, device=idx_i.device),
    )


def bond_graph(
    idx_i: Tensor,
    dist: Tensor,
    n_atoms: int,
    bond_cutoff: float,
) -> ThreeBodyPairs:
    """Ordered pairs of directed edges that meet at the receiver.

    ``idx_ij`` is the bond being updated. ``idx_ik`` is the other bond.
    Edges longer than ``bond_cutoff`` are omitted. An edge sitting exactly
    on the cutoff can be the updated bond and cannot be the other bond,
    matching the reference line graph.
    """
    pairs = receiver_triplets(idx_i, dist, n_atoms, bond_cutoff)
    if pairs.idx_ij.numel() == 0:
        return pairs
    keep = dist[pairs.idx_ik] < float(bond_cutoff)
    return ThreeBodyPairs(pairs.idx_ij[keep], pairs.idx_ik[keep])


def directed_bond_angle(vij: Tensor, idx_left: Tensor, idx_right: Tensor) -> Tensor:
    """Angle at the shared receiver, in radians.

    Vectors are normalized, then the cosine is scaled by ``1 - 1e-6`` before
    ``acos``, the same stability factor as the reference angle encoder.
    """
    if idx_left.numel() == 0:
        return vij.new_zeros(0)
    left = vij[idx_left]
    right = vij[idx_right]
    left = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    cosine = (left * right).sum(dim=-1) * (1.0 - 1e-6)
    cosine = cosine.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    return torch.acos(cosine)
