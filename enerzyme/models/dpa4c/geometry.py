"""Real Cartesian harmonics and bispectrum layout for DPA4C.

The polynomials and the Gaunt tables follow Li et al., arXiv:2608.19041,
and the degree profile used by the DeePMD-kit descriptor. DeePMD-kit is
LGPL-3.0 and is not imported.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..mp.basis.angular import spherical_harmonics
from ..mp.equivariant.s2.lebedev import load_lebedev_rule


SUPPORTED_CHANNELS = (8, 16, 32, 64, 128)
SUPPORTED_LMAX = (2, 3, 4)
MAX_ANGULAR_DEGREE = 4
DegreeTriple = Tuple[int, int, int]


def require_profile(channels: int, lmax: int) -> Tuple[int, int]:
    """Validate the compiled channel width and angular degree."""
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError(f"channels must be an integer, got {type(channels).__name__}")
    if isinstance(lmax, bool) or not isinstance(lmax, int):
        raise TypeError(f"lmax must be an integer, got {type(lmax).__name__}")
    if channels not in SUPPORTED_CHANNELS:
        raise ValueError(f"channels must be one of {SUPPORTED_CHANNELS}, got {channels}")
    if lmax not in SUPPORTED_LMAX:
        raise ValueError(f"lmax must be one of {SUPPORTED_LMAX}, got {lmax}")
    return int(channels), int(lmax)


def derive_degree_channels(channels: int, lmax: int) -> List[int]:
    """Channel counts ``C_0, ..., C_L``.

    Degree one is ``max(4, 2^{ceil(log2(C_0) / 2)})``, degree two is half of
    that (at least 4), and every higher degree keeps one channel.
    """
    channels, lmax = require_profile(channels, lmax)
    exponent = channels.bit_length() - 1
    degree_one = max(4, 1 << ((exponent + 1) // 2))
    degree_two = max(4, degree_one >> 1)
    return [channels, degree_one, degree_two] + [1] * (lmax - 2)


def derive_bispectrum_ranks(degree_channels: Sequence[int]) -> List[int]:
    """Probe ranks ``K_1 = C_2``, ``K_2 = 2``, and ``K_ℓ = 1`` for ``ℓ ≥ 3``."""
    if len(degree_channels) not in (3, 4, 5):
        raise ValueError(
            "degree_channels must cover lmax 2, 3, or 4, "
            f"got {len(degree_channels)} entries"
        )
    return [int(degree_channels[2]), 2] + [1] * (len(degree_channels) - 3)


def degree_offsets(degree_channels: Sequence[int]) -> Tuple[int, ...]:
    """Flat moment offsets, one past the last degree."""
    offsets = [0]
    for degree, width in enumerate(degree_channels):
        offsets.append(offsets[-1] + (2 * degree + 1) * int(width))
    return tuple(offsets)


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


def bispectrum_feature_count(lmax: int, ranks: Sequence[int]) -> int:
    """Number of independent cubic probe contractions."""
    total = 0
    for degree_1, degree_2, degree_3 in enumerate_degree_triples(lmax):
        rank_1 = int(ranks[degree_1 - 1])
        rank_2 = int(ranks[degree_2 - 1])
        rank_3 = int(ranks[degree_3 - 1])
        if degree_1 == degree_3:
            total += rank_1 * (rank_1 + 1) * (rank_1 + 2) // 6
        elif degree_1 == degree_2:
            total += rank_1 * (rank_1 + 1) // 2 * rank_3
        elif degree_2 == degree_3:
            total += rank_1 * rank_2 * (rank_2 + 1) // 2
        else:
            total += rank_1 * rank_2 * rank_3
    return total


def invariant_width(channels: int, lmax: int) -> int:
    """Width of the calibrated descriptor, including both masses and ``X0``."""
    degree_channels = derive_degree_channels(channels, lmax)
    ranks = derive_bispectrum_ranks(degree_channels)
    grams = sum(width * (width + 1) // 2 for width in degree_channels[1:])
    return (
        2 * channels
        + 2
        + grams
        + bispectrum_feature_count(lmax, ranks)
        + ranks[0] * ranks[1]
    )


def cartesian_harmonics(direction: Tensor, lmax: int) -> Tensor:
    """DPA4C solid harmonics with Cartesian degree one and increasing m above it."""
    return spherical_harmonics(direction, lmax=lmax, layout="dpa4c", normalize_input=False)


def packed_l2_to_stf(packed: Tensor) -> Tensor:
    """Map normalized degree-two coefficients to symmetric traceless matrices.

    The map is an isometry from the Euclidean product on ``R^5`` to the
    Frobenius product.
    """
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    inv_sqrt_six = 1.0 / math.sqrt(6.0)
    q0, q1, q2, q3, q4 = packed.unbind(dim=-1)
    qxy = q0 * inv_sqrt_two
    qyz = q1 * inv_sqrt_two
    qxz = q3 * inv_sqrt_two
    qxx = -q2 * inv_sqrt_six + q4 * inv_sqrt_two
    qyy = -q2 * inv_sqrt_six - q4 * inv_sqrt_two
    qzz = 2.0 * q2 * inv_sqrt_six
    row0 = torch.stack((qxx, qxy, qxz), dim=-1)
    row1 = torch.stack((qxy, qyy, qyz), dim=-1)
    row2 = torch.stack((qxz, qyz, qzz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


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
def coupling_tables(lmax: int) -> Tuple[DegreeTriple, ...]:
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
    basis = cartesian_harmonics(
        torch.as_tensor(points, dtype=torch.float64),
        lmax,
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
    coupling_tables._cache[lmax] = stored  # type: ignore[attr-defined]
    return triples


def cached_couplings(lmax: int) -> Tuple[np.ndarray, ...]:
    """Return the tensors built alongside :func:`coupling_tables`."""
    triples = coupling_tables(lmax)
    stored = getattr(coupling_tables, "_cache", {}).get(lmax)
    if stored is None or len(stored) != len(triples):
        raise RuntimeError(f"missing Gaunt cache for lmax={lmax}")
    return tuple(stored)


# Attach a side cache without a second quadrature pass.
coupling_tables._cache = {}  # type: ignore[attr-defined]


def probe_entries(
    degrees: DegreeTriple,
    ranks: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Independent probe indices and isometric orbit scales.

    Equal-degree axes are represented by the nondecreasing index, scaled by
    the square root of the orbit size.
    """
    rank_1 = int(ranks[degrees[0] - 1])
    rank_2 = int(ranks[degrees[1] - 1])
    rank_3 = int(ranks[degrees[2] - 1])
    entries: List[Tuple[int, int, int]] = []
    scales: List[float] = []
    if degrees[0] == degrees[2]:
        for entry in itertools.combinations_with_replacement(range(rank_1), 3):
            entries.append(entry)
            counts = [entry.count(value) for value in set(entry)]
            multiplicity = math.factorial(3)
            for count in counts:
                multiplicity //= math.factorial(count)
            scales.append(math.sqrt(float(multiplicity)))
    elif degrees[0] == degrees[1]:
        for first, second in itertools.combinations_with_replacement(range(rank_1), 2):
            scale = 1.0 if first == second else math.sqrt(2.0)
            for third in range(rank_3):
                entries.append((first, second, third))
                scales.append(scale)
    elif degrees[1] == degrees[2]:
        for first in range(rank_1):
            for second, third in itertools.combinations_with_replacement(range(rank_2), 2):
                entries.append((first, second, third))
                scales.append(1.0 if second == third else math.sqrt(2.0))
    else:
        for entry in itertools.product(range(rank_1), range(rank_2), range(rank_3)):
            entries.append(entry)
            scales.append(1.0)
    index = np.asarray(
        [(first * rank_2 + second) * rank_3 + third for first, second, third in entries],
        dtype=np.int64,
    )
    return index, np.asarray(scales, dtype=np.float64)


def legendre(degree: int, value: Tensor) -> Tensor:
    """Legendre polynomial ``P_ℓ`` used by the addition theorem."""
    if degree == 0:
        return torch.ones_like(value)
    if degree == 1:
        return value
    if degree == 2:
        return 0.5 * (3.0 * value * value - 1.0)
    if degree == 3:
        return 0.5 * (5.0 * value**3 - 3.0 * value)
    if degree == 4:
        square = value * value
        return (35.0 * square * square - 30.0 * square + 3.0) / 8.0
    raise ValueError(f"legendre degree must be in [0, 4], got {degree}")
