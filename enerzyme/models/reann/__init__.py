"""Recursively embedded atom neural network.

Zhang, Xia, and Jiang, J. Chem. Phys. 156, 114801 (2022). The Core is the
density iteration. Embedding, the energy head, and response tensors stay in
the shared layer stack.
"""

from .core import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MD17_CORE_PARAMS,
    PUBLISHED_MD17_CUTOFF,
    REANNCore,
)

__all__ = [
    "REANNCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_MD17_CORE_PARAMS",
    "PUBLISHED_MD17_CUTOFF",
]
