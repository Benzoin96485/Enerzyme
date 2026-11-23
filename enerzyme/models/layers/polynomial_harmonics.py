"""Polynomial spherical harmonics shared by Geodite and ViSNet.

``l = 1`` is the Cartesian unit vector ``(x, y, z)``, not the e3nn ``(y, z, x)``
order. Higher degrees follow the e3nn component recursion used by IBM
``TensorInit`` and by ViSNet's ``Sphere``, without the constant ``l = 0``
channel. Geodite-MP uses ``lmax = 2``. ViSNet's directional features use
``lmax`` 1 or 2, and QuinNet's ``Sphere(l=2)`` is that same 8-channel prefix.
Degrees above 4 are rejected: a mismatched basis would still be equivariant
but would not be any of these architectures.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import Tensor


def degree_widths(lmax: int) -> List[int]:
    """Multiplicities ``2l + 1`` for ``l = 1 .. lmax``."""
    return [2 * degree + 1 for degree in range(1, int(lmax) + 1)]


def angular_dim(lmax: int) -> int:
    """Steerable width ``(lmax + 1)² - 1``."""
    return (int(lmax) + 1) ** 2 - 1


def split_degrees(tensor: Tensor, lmax: int, dim: int = 1) -> tuple[Tensor, ...]:
    return torch.split(tensor, degree_widths(lmax), dim=dim)


def polynomial_spherical_harmonics(unit_vectors: Tensor, lmax: int) -> Tensor:
    """Real harmonics of unit vectors. Shape ``[E, (lmax + 1)² - 1]``."""
    if unit_vectors.ndim != 2 or unit_vectors.shape[-1] != 3:
        raise ValueError(
            f"unit vectors must have shape [E, 3], got {tuple(unit_vectors.shape)}"
        )
    lmax = int(lmax)
    if lmax < 1 or lmax > 4:
        raise ValueError(f"lmax must be in 1..4, got {lmax}")
    x, y, z = unit_vectors.unbind(dim=-1)
    parts = [x, y, z]
    if lmax == 1:
        return torch.stack(parts, dim=-1)

    sqrt3 = math.sqrt(3.0)
    y2 = y.square()
    x2z2 = x.square() + z.square()
    sh20 = sqrt3 * x * z
    sh21 = sqrt3 * x * y
    sh22 = y2 - 0.5 * x2z2
    sh23 = sqrt3 * y * z
    sh24 = (sqrt3 / 2.0) * (z.square() - x.square())
    parts.extend((sh20, sh21, sh22, sh23, sh24))
    if lmax == 2:
        return torch.stack(parts, dim=-1)

    sqrt42 = math.sqrt(42.0)
    sqrt168 = math.sqrt(168.0)
    sqrt7 = math.sqrt(7.0)
    sh30 = (1.0 / 6.0) * sqrt42 * (sh20 * z + sh24 * x)
    sh31 = sqrt7 * sh20 * y
    sh32 = (1.0 / 8.0) * sqrt168 * (4.0 * y2 - x2z2) * x
    sh33 = 0.5 * sqrt7 * y * (2.0 * y2 - 3.0 * x2z2)
    sh34 = (1.0 / 8.0) * sqrt168 * z * (4.0 * y2 - x2z2)
    sh35 = sqrt7 * sh24 * y
    sh36 = (1.0 / 6.0) * sqrt42 * (sh24 * z - sh20 * x)
    parts.extend((sh30, sh31, sh32, sh33, sh34, sh35, sh36))
    if lmax == 3:
        return torch.stack(parts, dim=-1)

    sqrt2 = math.sqrt(2.0)
    sqrt6 = math.sqrt(6.0)
    sqrt14 = math.sqrt(14.0)
    sqrt21 = math.sqrt(21.0)
    sqrt210 = math.sqrt(210.0)
    sqrt105 = math.sqrt(105.0)
    sqrt70 = math.sqrt(70.0)
    sh40 = 0.75 * sqrt2 * (sh30 * z + sh36 * x)
    sh41 = 0.75 * sh30 * y + (3.0 / 8.0) * sqrt6 * sh31 * z + (3.0 / 8.0) * sqrt6 * sh35 * x
    sh42 = (
        -3.0 / 56.0 * sqrt14 * sh30 * z
        + (3.0 / 14.0) * sqrt21 * sh31 * y
        + (3.0 / 56.0) * sqrt210 * sh32 * z
        + (3.0 / 56.0) * sqrt210 * sh34 * x
        + (3.0 / 56.0) * sqrt14 * sh36 * x
    )
    sh43 = (
        -3.0 / 56.0 * sqrt42 * sh31 * z
        + (3.0 / 28.0) * sqrt105 * sh32 * y
        + (3.0 / 28.0) * sqrt70 * sh33 * x
        + (3.0 / 56.0) * sqrt42 * sh35 * x
    )
    sh44 = (
        -3.0 / 28.0 * sqrt42 * sh32 * x
        + (3.0 / 7.0) * sqrt7 * sh33 * y
        - 3.0 / 28.0 * sqrt42 * sh34 * z
    )
    sh45 = (
        -3.0 / 56.0 * sqrt42 * sh31 * x
        + (3.0 / 28.0) * sqrt70 * sh33 * z
        + (3.0 / 28.0) * sqrt105 * sh34 * y
        - 3.0 / 56.0 * sqrt42 * sh35 * z
    )
    sh46 = (
        -3.0 / 56.0 * sqrt14 * sh30 * x
        - 3.0 / 56.0 * sqrt210 * sh32 * x
        + (3.0 / 56.0) * sqrt210 * sh34 * z
        + (3.0 / 14.0) * sqrt21 * sh35 * y
        - 3.0 / 56.0 * sqrt14 * sh36 * z
    )
    sh47 = (
        -3.0 / 8.0 * sqrt6 * sh31 * x
        + (3.0 / 8.0) * sqrt6 * sh35 * z
        + 0.75 * sh36 * y
    )
    sh48 = 0.75 * sqrt2 * (-sh30 * x + sh36 * z)
    parts.extend((sh40, sh41, sh42, sh43, sh44, sh45, sh46, sh47, sh48))
    return torch.stack(parts, dim=-1)


# Geodite imported this name before the basis moved here.
geodite_spherical_harmonics = polynomial_spherical_harmonics
