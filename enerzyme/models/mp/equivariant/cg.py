"""Real e3nn-basis coupling coefficients and allowed O(3) product paths."""

from functools import lru_cache
from math import sqrt

import torch
from e3nn import o3
from e3nn.util import explicit_default_types


@lru_cache(maxsize=None)
def _unit_three_j(l1, l2, l3):
    return o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")


def wigner_3j(l1, l2, l3, dtype=None, device=None, *, normalized=True):
    """Return an independent real coefficient tensor in e3nn magnetic order.

    The default tensor has unit Frobenius norm. With `normalized=False`,
    its columns are orthonormal Clebsch–Gordan vectors for the output degree.
    """
    dtype, device = explicit_default_types(dtype, device)
    values = _unit_three_j(l1, l2, l3)
    if not normalized:
        values = values * sqrt(2 * l3 + 1)
    return values.to(dtype=dtype, device=device, copy=True)


@lru_cache(maxsize=None)
def wigner_6j(a, b, c, d, e, f):
    """Exact Racah recoupling coefficient, shared across layer constructions."""
    from sympy.physics.wigner import wigner_6j as exact_six_j

    return exact_six_j(a, b, c, d, e, f)


@lru_cache(maxsize=None)
def _paths(left, right, outputs):
    return tuple(
        (i, j, k)
        for i, (l1, p1) in enumerate(left)
        for j, (l2, p2) in enumerate(right)
        for k, (l3, p3) in enumerate(outputs)
        if abs(l1 - l2) <= l3 <= l1 + l2 and p1 * p2 == p3
    )


def coupling_paths(left, right, outputs):
    """Index triples in e3nn instruction order for sequences of (degree, parity)."""
    return _paths(tuple(left), tuple(right), tuple(outputs))


@lru_cache(maxsize=None)
def _gaunt(l1, l2, l3, convention):
    if convention == "dpa4c":
        from .gaunt import cached_couplings, enumerate_degree_triples

        degrees = (l1, l2, l3)
        triples = enumerate_degree_triples(l3)
        if degrees not in triples:
            raise ValueError(f"expected an ordered non-scalar DPA4C degree triple, got {degrees}")
        return torch.from_numpy(cached_couplings(l3)[triples.index(degrees)])
    if convention != "e3nn":
        raise ValueError(f"unknown Gaunt convention {convention!r}")
    from math import pi

    shape = (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    if not abs(l1 - l2) <= l3 <= l1 + l2 or (l1 + l2 + l3) % 2:
        return torch.zeros(shape, dtype=torch.float64)
    three_j = _unit_three_j(l1, l2, l3)
    return three_j * three_j[l1, l2, l3] * sqrt(shape[0] * shape[1] * shape[2] / (4 * pi))


def gaunt(l1, l2, l3, dtype=None, device=None, *, convention="e3nn"):
    """Triple integral of integral-normalized real e3nn spherical harmonics.

    The DPA4C convention gives unit-norm, sign-fixed Cartesian readout tensors.
    """
    dtype, device = explicit_default_types(dtype, device)
    return _gaunt(l1, l2, l3, convention).to(dtype=dtype, device=device, copy=True)
