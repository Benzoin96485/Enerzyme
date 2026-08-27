"""Geometry filters, covalent graphs, and heavy-atom RMSD."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from ase.data import atomic_masses, chemical_symbols, covalent_radii
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from .constants import CONNECT_SCALE, OVERLAP_SCALE

ArrayLike = np.ndarray | Sequence[float] | Sequence[Sequence[float]]


def as_int_numbers(numbers: Iterable[int]) -> np.ndarray:
    return np.asarray(list(numbers), dtype=np.int64)


def as_positions(positions: ArrayLike) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    return pos


def pairwise_distances(positions: ArrayLike) -> np.ndarray:
    pos = as_positions(positions)
    return cdist(pos, pos, metric="euclidean")


def covalent_cutoff_matrix(numbers: Iterable[int], scale: float) -> np.ndarray:
    z = as_int_numbers(numbers)
    rc = covalent_radii[z]
    cutoff = scale * (rc[:, None] + rc[None, :])
    np.fill_diagonal(cutoff, 0.0)
    return cutoff


def adjacency_matrix(
    numbers: Iterable[int],
    positions: ArrayLike,
    scale: float = CONNECT_SCALE,
) -> np.ndarray:
    dist = pairwise_distances(positions)
    cutoff = covalent_cutoff_matrix(numbers, scale)
    return (dist < cutoff) & (dist > 1.0e-8)


def n_connected_components(
    numbers: Iterable[int],
    positions: ArrayLike,
    scale: float = CONNECT_SCALE,
) -> int:
    adj = adjacency_matrix(numbers, positions, scale=scale)
    n_comp, _ = connected_components(adj.astype(np.int8), directed=False)
    return int(n_comp)


def has_atom_overlap(
    numbers: Iterable[int],
    positions: ArrayLike,
    scale: float = OVERLAP_SCALE,
) -> bool:
    dist = pairwise_distances(positions)
    cutoff = covalent_cutoff_matrix(numbers, scale)
    offdiag = ~np.eye(dist.shape[0], dtype=bool)
    return bool(np.any((dist < cutoff) & offdiag))


def radius_of_gyration(positions: ArrayLike, numbers: Iterable[int] | None = None) -> float:
    pos = as_positions(positions)
    if numbers is None:
        weights = np.ones(pos.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(atomic_masses[as_int_numbers(numbers)], dtype=np.float64)
    weights = np.maximum(weights, 1.0e-12)
    com = (weights[:, None] * pos).sum(axis=0) / weights.sum()
    delta = pos - com
    return float(np.sqrt(np.sum(weights * np.sum(delta * delta, axis=1)) / weights.sum()))


def heavy_mask(numbers: Iterable[int]) -> np.ndarray:
    return as_int_numbers(numbers) > 1


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.shape[0] == 0:
        return float("inf")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    cov = ac.T @ bc
    u, _s, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    aligned = ac @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - bc) ** 2, axis=1))))


def _match_by_element(za: np.ndarray, pa: np.ndarray, zb: np.ndarray, pb: np.ndarray) -> np.ndarray | None:
    """Return ``pb`` rows permuted to match ``za``/``pa``, or None if formulas differ."""
    if za.shape != zb.shape:
        return None
    order_b = np.full(za.shape[0], -1, dtype=np.int64)
    used = np.zeros(zb.shape[0], dtype=bool)
    for z in np.unique(za):
        ia = np.where(za == z)[0]
        ib = np.where(zb == z)[0]
        if ia.size != ib.size:
            return None
        if ia.size == 1:
            order_b[ia[0]] = ib[0]
            used[ib[0]] = True
            continue
        cost = cdist(pa[ia], pb[ib], metric="sqeuclidean")
        row, col = linear_sum_assignment(cost)
        order_b[ia[row]] = ib[col]
        used[ib[col]] = True
    if np.any(order_b < 0):
        return None
    return pb[order_b]


def heavy_atom_rmsd(
    numbers_a: Iterable[int],
    positions_a: ArrayLike,
    numbers_b: Iterable[int],
    positions_b: ArrayLike,
) -> float:
    """Hungarian-matched Kabsch RMSD over heavy atoms (all atoms if fewer than 3 heavy)."""
    za = as_int_numbers(numbers_a)
    zb = as_int_numbers(numbers_b)
    pa = as_positions(positions_a)
    pb = as_positions(positions_b)
    mask_a = heavy_mask(za)
    mask_b = heavy_mask(zb)
    if int(mask_a.sum()) >= 3 and int(mask_b.sum()) >= 3:
        za, pa = za[mask_a], pa[mask_a]
        zb, pb = zb[mask_b], pb[mask_b]
    matched = _match_by_element(za, pa, zb, pb)
    if matched is None:
        return float("inf")
    return _kabsch_rmsd(pa, matched)


def formula_from_numbers(numbers: Iterable[int]) -> str:
    z = as_int_numbers(numbers)
    counts: dict[str, int] = {}
    for atom_z in z:
        sym = chemical_symbols[int(atom_z)]
        counts[sym] = counts.get(sym, 0) + 1
    hill = []
    for sym in ("C", "H"):
        if sym in counts:
            n = counts.pop(sym)
            hill.append(f"{sym}{n}" if n != 1 else sym)
    for sym in sorted(counts):
        n = counts[sym]
        hill.append(f"{sym}{n}" if n != 1 else sym)
    return "".join(hill) or "empty"
