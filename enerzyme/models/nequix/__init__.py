"""Nequix: a compact E(3)-equivariant potential (Koker and Smidt, 2025).

The Core is the interaction stack. One-hot embedding, Bessel / polynomial
cutoff, readout, and forces are shared layers.
"""

from .core import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MP_CORE_PARAMS,
    PUBLISHED_MP_SCALE,
    PUBLISHED_MP_SHIFT,
    NequixCore,
)
from .species import MPTRJ_ATOMIC_NUMBERS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "MPTRJ_ATOMIC_NUMBERS",
    "NequixCore",
    "PUBLISHED_MP_CORE_PARAMS",
    "PUBLISHED_MP_SCALE",
    "PUBLISHED_MP_SHIFT",
]
