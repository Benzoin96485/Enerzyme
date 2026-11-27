"""Real Cartesian Gaunt tensors projected with the Lebedev rule on the sphere."""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

from ..basis.angular import spherical_harmonics
from .s2.lebedev import load_lebedev_rule

MAX_ANGULAR_DEGREE = 4
DegreeTriple = Tuple[int, int, int]


def enumerate_degree_triples(lmax: int) -> Tuple[DegreeTriple, ...]:
    """Non-scalar O(3)-even triples in lexicographic order."""
    if lmax < 0 or lmax > MAX_ANGULAR_DEGREE:
        raise ValueError(f"lmax must be in [0, {MAX_ANGULAR_DEGREE}], got {lmax}")
    triples = []
    for degree_1 in range(1, lmax + 1):
        for degree_2 in range(degree_1, lmax + 1):
            for degree_3 in range(degree_2, lmax + 1):
                if degree_3 > degree_1 + degree_2:
                    continue
                if (degree_1 + degree_2 + degree_3) % 2 != 0:
                    continue
                triples.append((degree_1, degree_2, degree_3))
    return tuple(triples)


def _symmetrize_equal_degrees(coupling: np.ndarray, degrees: DegreeTriple) -> np.ndarray:
    if degrees[0] == degrees[2]:
        permutations = tuple(itertools.permutations((0, 1, 2)))
    elif degrees[0] == degrees[1]:
        permutations = ((0, 1, 2), (1, 0, 2))
    elif degrees[1] == degrees[2]:
        permutations = ((0, 1, 2), (0, 2, 1))
    else:
        return coupling
    total = sum(np.transpose(coupling, axes) for axes in permutations)
    return total / len(permutations)


@lru_cache(maxsize=4)
def cached_couplings(lmax: int) -> Tuple[np.ndarray, ...]:
    """Unit-Frobenius real Cartesian Gaunt tensors, one per admissible triple.

    Lebedev weights in this repository sum to 1. The overall scale is removed
    by the Frobenius normalization, and the sign is chosen so the first
    significant entry is positive. The ``(1, 1, 2)`` block is not read by the
    readout; that triple uses the closed form ``-v^T Q v / sqrt(5)``.
    """
    triples = enumerate_degree_triples(lmax)
    if not triples:
        return triples
    points, weights = load_lebedev_rule(13)
    basis = spherical_harmonics(
        torch.as_tensor(points, dtype=torch.float64),
        lmax=lmax, layout="dpa4c", normalize_input=False,
    ).numpy()
    stored = []
    for degree_1, degree_2, degree_3 in triples:
        block_1 = basis[:, degree_1**2 : (degree_1 + 1) ** 2]
        block_2 = basis[:, degree_2**2 : (degree_2 + 1) ** 2]
        block_3 = basis[:, degree_3**2 : (degree_3 + 1) ** 2]
        coupling = np.einsum(
            "n,ni,nj,nk->ijk",
            weights,
            block_1,
            block_2,
            block_3,
            optimize=True,
        )
        coupling = _symmetrize_equal_degrees(coupling, (degree_1, degree_2, degree_3))
        norm = float(np.linalg.norm(coupling))
        if not math.isfinite(norm) or norm <= 1.0e-14:
            raise ValueError(
                f"degenerate bispectrum coupling for {(degree_1, degree_2, degree_3)}"
            )
        coupling = coupling / norm
        significant = np.flatnonzero(np.abs(coupling) > 1.0e-14)
        if significant.size > 0 and coupling.flat[significant[0]] < 0.0:
            coupling = -coupling
        stored.append(np.ascontiguousarray(coupling))
    return tuple(stored)


def coupling_tables(lmax: int) -> Tuple[DegreeTriple, ...]:
    """Build the Gaunt cache and return its ordered degree triples."""
    cached_couplings(lmax)
    return enumerate_degree_triples(lmax)
