"""Cartesian Atomic Cluster Expansion (CACE).

Cheng, npj Computational Materials 10, 157 (2024), arXiv:2402.07472.

The Core is the map from a chemical embedding and a radial basis to invariant
cluster features. ``RandomAtomEmbedding``, ``BesselRBF``, and
``CACEEnergyReadout`` stay outside it. The reference repository is CC BY-NC
and is not a dependency.
"""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
from .presets import PAPER_PRESETS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PAPER_PRESETS",
]
