"""TensorNet: Cartesian tensor representations for molecular potentials.

Simeon and De Fabritiis, NeurIPS 2023 (arXiv:2306.06482). The Core is the
tensor embedding and interaction stack. The radial basis, species embedding,
cutoff window, and energy head stay in the shared layer stack.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, TensorNetCore

__all__ = [
    "TensorNetCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
