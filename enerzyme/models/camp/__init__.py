"""Cartesian Atomic Moment Potential (CAMP).

Wen, Huang, Dai, and Adhikari, npj Computational Materials 11, 128 (2025),
doi:10.1038/s41524-025-01623-4.

The Core is the map from a chemical embedding and a radius graph to the
scalar features of every message-passing layer. ``RandomAtomEmbedding`` and
``CAMPEnergyReadout`` stay outside it. The reference repository is not a
dependency.
"""

from .core import CAMPCore, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
from .presets import PAPER_PRESETS

__all__ = [
    "CAMPCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PAPER_PRESETS",
]
