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

from ...ops import scatter_sum


def reduce_nodes(src: Tensor, index: Tensor, dim_size: int, reduce: str = "sum") -> Tensor:
    """Aggregate ``src`` onto ``dim_size`` nodes. Empty neighborhoods stay zero."""
    if index.numel() == 0:
        return src.new_zeros((dim_size, *src.shape[1:]))
    if reduce == "sum":
        return scatter_sum(src, index, dim_size)
    if reduce == "mean":
        total = scatter_sum(src, index, dim_size)
        count = scatter_sum(src.new_ones(index.shape[0]), index, dim_size).clamp_min(1)
        view = (dim_size,) + (1,) * (src.ndim - 1)
        return total / count.reshape(view)
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


def neighbor_mean(pos: Tensor, idx_i: Tensor, idx_j: Tensor) -> tuple[Tensor, Tensor]:
    """Mean sender position on each receiver, and the sender count.

    Atoms that never appear as a receiver keep a count of zero. Their mean
    is left at zero and must not be used as a centroid: that value is not
    the origin of the molecule.
    """
    n_atoms = pos.shape[0]
    mean = pos.new_zeros(n_atoms, 3)
    count = pos.new_zeros(n_atoms)
    if idx_i.numel() == 0:
        return mean, count
    total = scatter_sum(pos[idx_j], idx_i, n_atoms)
    count = scatter_sum(pos.new_ones(idx_i.shape[0]), idx_i, n_atoms)
    present = count > 0
    filled = total / count.clamp_min(1).unsqueeze(-1)
    mean = torch.where(present.unsqueeze(-1), filled, mean)
    return mean, count


def _require(condition: Tensor, message: str) -> None:
    if torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def require_same_graph(batch: Tensor | None, idx_i: Tensor, idx_j: Tensor) -> None:
    """Reject an edge that joins atoms from two molecules."""
    if batch is None or (not torch.compiler.is_compiling() and idx_i.numel() == 0):
        return
    if batch.ndim != 1:
        raise ValueError("batch_seg must be a 1D graph id per atom")
    _require(
        ((idx_i < batch.shape[0]) & (idx_j < batch.shape[0])).all(),
        "batch_seg must list one graph id per atom and cover every neighbor index",
    )
    _require((batch[idx_i] == batch[idx_j]).all(), "neighbor edges must stay inside one molecule")


def require_matching_distance(vij: Tensor, dij: Tensor, tol: float = 1e-4) -> None:
    """``Dij`` is the Euclidean length of ``vij``, including a periodic image."""
    if not torch.compiler.is_compiling() and vij.numel() == 0:
        return
    actual = torch.linalg.vector_norm(vij, dim=-1)
    given = dij.reshape(-1).to(dtype=actual.dtype)
    if given.shape != actual.shape:
        raise ValueError("Dij_sr and vij_sr must describe the same edges")
    scale = actual.detach().clamp_min(1.0)
    _require(~torch.any((given - actual).abs() > tol * scale), "Dij_sr must be the Euclidean norm of vij_sr")


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
    total = scatter_sum(pos, batch)
    count = scatter_sum(pos.new_ones(n_atoms), batch, total.shape[0]).clamp_min(1)
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
    mean, count = neighbor_mean(pos, idx_i, idx_j)
    displacement = pos - mean
    diff = unit_vector(displacement, eps)
    cross = unit_vector(torch.cross(pos, mean, dim=-1), eps)
    vertical = torch.cross(diff, cross, dim=-1)
    frame = torch.stack((diff, cross, vertical), dim=-1)
    # A missing neighborhood used to look like a centroid at the origin.
    # The identity does not depend on where that atom sits in the cell.
    degenerate = (count == 0) | (torch.linalg.vector_norm(displacement, dim=-1) < 1e-6)
    eye = torch.eye(3, device=pos.device, dtype=pos.dtype).expand(pos.shape[0], -1, -1)
    return torch.where(degenerate.view(-1, 1, 1), eye, frame)


def alphanet_edge_frame(
    pos: Tensor,
    vij: Tensor,
    dij: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """AlphaNet edge frame.

    ``e1`` is the unit ``vij``. ``Dij`` must be that length, so a stale
    distance cannot tilt the axis. ``e2`` is the unnormalized cross product
    of the receiver and sender relative to the receiver's neighbor centroid.
    The released model does not normalize ``e2`` or ``e3``; the paper writes
    both as unit vectors. This follows the released model. ``pos`` is the
    in-cell coordinate, while ``vij`` may already include the periodic image.
    Every receiver in ``idx_i`` has at least one neighbor, so the centroid
    is the mean of those senders.
    """
    require_matching_distance(vij, dij)
    edge_unit = unit_vector(vij, eps)
    mean, _ = neighbor_mean(pos, idx_i, idx_j)
    center = mean[idx_i]
    cross = torch.cross(pos[idx_i] - center, pos[idx_j] - center, dim=-1)
    vertical = torch.cross(edge_unit, cross, dim=-1)
    return torch.stack((edge_unit, cross, vertical), dim=-1), edge_unit
