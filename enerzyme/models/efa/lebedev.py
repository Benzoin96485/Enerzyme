"""Lebedev quadrature for Euclidean Fast Attention.

Thin re-export of the shared tables in :mod:`enerzyme.models.so3.lebedev`
(e3x Apache-2.0 ``lebedev_grids.npz``). See ``enerzyme/models/efa/NOTICE``.
"""

from __future__ import annotations

from ..so3.lebedev import (  # noqa: F401
    LEBEDEV_FREQUENCY_LOOKUP,
    available_lebedev_nums,
    lebedev_quadrature,
    lebedev_tensors,
    recommend_max_frequency,
)

__all__ = [
    "LEBEDEV_FREQUENCY_LOOKUP",
    "available_lebedev_nums",
    "lebedev_quadrature",
    "lebedev_tensors",
    "recommend_max_frequency",
]
