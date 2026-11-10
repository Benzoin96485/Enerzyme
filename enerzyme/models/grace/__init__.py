"""Graph Atomic Cluster Expansion (GRACE).

Lysogorskiy, Bochkarev, and Drautz, arXiv:2508.17936 (2025), and
Bochkarev, Lysogorskiy, and Drautz, Phys. Rev. X 14, 021036 (2024).

The Core is the map from a chemical embedding to invariant ACE densities.
``RandomAtomEmbedding`` and ``GRACEEnergyReadout`` (or ``SimpleReadout``)
stay outside it. The reference TensorPotential package is not a dependency.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, GRACECore
from .presets import PAPER_PRESETS

__all__ = [
    "GRACECore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PAPER_PRESETS",
]
