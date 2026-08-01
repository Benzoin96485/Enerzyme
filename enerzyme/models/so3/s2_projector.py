"""Shared flat S² projector contract (lat–long ``SO3Grid`` and Lebedev)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class S2GridProjector(Protocol):
    """Project between packed SH coeffs and an S² discretization.

    Contract (flat layout)::

        to_grid:   (N, D, C) → (N, G, C)
        from_grid: (N, G, C) → (N, D, C)
    """

    def to_grid(self, x: Tensor) -> Tensor: ...

    def from_grid(self, grid: Tensor) -> Tensor: ...
