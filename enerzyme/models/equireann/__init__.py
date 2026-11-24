"""Equivariant recursively embedded atom neural network.

Wu, Xia, Zhang, and Jiang, J. Phys. Chem. A 128, 11061 (2024). The Core extends
REANN: the same embedded-atom density, plus the neighbor's Cartesian orbital
passed at each coefficient update. Embedding and property heads stay outside.
"""

from .defaults import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_CUMULENE_CORE_PARAMS,
    PUBLISHED_CUMULENE_CUTOFF,
)

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_CUMULENE_CORE_PARAMS",
    "PUBLISHED_CUMULENE_CUTOFF",
]
