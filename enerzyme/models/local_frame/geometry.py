"""Local frames shared by LEFTNet and AlphaNet.

Receivers are ``idx_i`` and senders are ``idx_j``. ``vij`` points from the
receiver to the sender (``R_j - R_i``), which is the Enerzyme short-range
vector and the direction both reference implementations pass into the
vector update.

Frames are stored as ``[..., 3, 3]`` with the three axes in the last
dimension (columns). Scalarization takes the absolute value of the middle
axis. That axis is a pseudovector, so the absolute value makes the
resulting scalars invariant under reflections (Du et al., Eq. 11).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..functional import segment_sum_coo


def reduce_nodes(src: Tensor, index: Tensor, dim_size: int, reduce: str = "sum") -> Tensor:
    """Aggregate ``src`` onto ``dim_size`` nodes. Empty neighborhoods stay zero."""
    if index.numel() == 0:
        return src.new_zeros((dim_size, *src.shape[1:]))
    if reduce == "sum":
        return segment_sum_coo(src, index, dim_size=dim_size)
    if reduce == "mean":
        total = segment_sum_coo(src, index, dim_size=dim_size)
        count = segment_sum_coo(
            src.new_ones(index.shape[0]), index, dim_size=dim_size
        ).clamp_min(1)
        view = (dim_size,) + (1,) * (src.ndim - 1)
        return total / count.reshape(view)
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


def neighbor_mean(pos: Tensor, idx_i: Tensor, idx_j: Tensor) -> tuple[Tensor, Tensor]:
    """Mean sender position on each receiver, and the sender count."""
    n_atoms = pos.shape[0]
    if idx_i.numel() == 0:
        return pos.new_zeros(n_atoms, 3), pos.new_zeros(n_atoms)
    total = segment_sum_coo(pos[idx_j], idx_i, dim_size=n_atoms)
    count = segment_sum_coo(pos.new_ones(idx_i.shape[0]), idx_i, dim_size=n_atoms)
    mean = total / count.clamp_min(1).unsqueeze(-1)
    return mean, count


def subtract_graph_mean(pos: Tensor, batch: Tensor | None) -> Tensor:
    """Subtract each graph's mean position.

    LEFTNet builds ``pos_i × pos_j`` after centering the molecule
    (``posc`` in the reference). The cross product is not translation
    invariant until that center is removed. Graphs in one batch are
    centered separately.
    """
    n_atoms = pos.shape[0]
    if n_atoms == 0:
        return pos
    if batch is None:
        batch = torch.zeros(n_atoms, dtype=torch.long, device=pos.device)
    n_graph = int(batch.max().item()) + 1
    total = segment_sum_coo(pos, batch, dim_size=n_graph)
    count = segment_sum_coo(pos.new_ones(n_atoms), batch, dim_size=n_graph).clamp_min(1)
    return pos - (total / count.unsqueeze(-1))[batch]


def unit_vector(vec: Tensor, eps: float) -> Tensor:
    norm = torch.linalg.vector_norm(vec, dim=-1, keepdim=True)
    return vec / (norm + eps)


def scalarize(vectors: Tensor, frame: Tensor) -> Tensor:
    """Project ``vectors`` ``[..., 3, F]`` onto ``frame`` ``[..., 3, 3]``.

    The middle component is the projection on the pseudovector axis and is
    replaced by its absolute value.
    """
    projected = torch.einsum("...cf,...ca->...af", vectors, frame)
    return torch.stack(
        (projected[..., 0, :], projected[..., 1, :].abs(), projected[..., 2, :]),
        dim=-2,
    )


def leftnet_edge_frame(
    pos: Tensor,
    vij: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """LEFTNet edge frame on already centered positions.

    ``e1`` is the unit sender−receiver vector. ``e2`` is the unit cross
    product of the centered sender and receiver positions, matching
    ``normalize(pos[source] × pos[target])`` in the reference. ``e3`` is
    ``e1 × e2`` and is left unnormalized, as in that code.
    """
    edge_unit = unit_vector(vij, eps)
    cross = unit_vector(torch.cross(pos[idx_j], pos[idx_i], dim=-1), eps)
    vertical = torch.cross(edge_unit, cross, dim=-1)
    return torch.stack((edge_unit, cross, vertical), dim=-1), edge_unit


def leftnet_node_frame(
    pos: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    eps: float,
) -> Tensor:
    """Per-atom frame from the neighbor centroid (Du et al., Eq. 10).

    The reference crosses ``pos × mean`` rather than ``mean × pos``, which
    flips the pseudovector axis relative to the printed formula. Degenerate
    atoms (no neighbors, or a neighbor centroid on the atom) get the
    identity so the frame scalarization stays finite.
    """
    mean, _ = neighbor_mean(pos, idx_i, idx_j)
    diff = unit_vector(pos - mean, eps)
    cross = unit_vector(torch.cross(pos, mean, dim=-1), eps)
    vertical = torch.cross(diff, cross, dim=-1)
    frame = torch.stack((diff, cross, vertical), dim=-1)
    degenerate = torch.linalg.vector_norm(diff, dim=-1) < 1e-6
    if bool(degenerate.any()):
        frame = frame.clone()
        frame[degenerate] = torch.eye(3, device=pos.device, dtype=pos.dtype)
    return frame


def alphanet_edge_frame(
    pos: Tensor,
    vij: Tensor,
    dij: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """AlphaNet edge frame.

    ``e1`` is ``vij / (r + eps)``. ``e2`` is the unnormalized cross product
    of the receiver and sender relative to the receiver's neighbor centroid.
    The released model does not normalize ``e2`` or ``e3``; the paper writes
    both as unit vectors. This follows the released model. ``pos`` is the
    in-cell coordinate, while ``vij`` may already include the periodic image.
    """
    edge_unit = vij / (dij.reshape(-1, 1) + eps)
    mean, _ = neighbor_mean(pos, idx_i, idx_j)
    center = mean[idx_i]
    cross = torch.cross(pos[idx_i] - center, pos[idx_j] - center, dim=-1)
    vertical = torch.cross(edge_unit, cross, dim=-1)
    return torch.stack((edge_unit, cross, vertical), dim=-1), edge_unit
