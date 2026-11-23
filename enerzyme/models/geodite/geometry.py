"""Geodite edge spherical harmonics.

The basis itself lives in :mod:`enerzyme.models.layers.polynomial_harmonics`
so ViSNet can use the same ``l <= 2`` prefix. This module re-exports the
names Geodite already imported.
"""

from ..layers.polynomial_harmonics import (
    angular_dim,
    degree_widths,
    geodite_spherical_harmonics,
    polynomial_spherical_harmonics,
    split_degrees,
)

__all__ = [
    "angular_dim",
    "degree_widths",
    "geodite_spherical_harmonics",
    "polynomial_spherical_harmonics",
    "split_degrees",
]
