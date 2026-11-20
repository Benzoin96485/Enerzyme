"""TensorNet and TensorNet2 Cartesian tensor potentials.

TensorNet is Simeon and De Fabritiis, NeurIPS 2023 (arXiv:2306.06482).
TensorNet2 is Farr et al., arXiv:2601.00581: the same tensor stack plus
neutral charge equilibration inside the messages. The radial basis, species
embedding, cutoff window, energy head, Coulomb term, and ZBL stay in the
shared layer stack.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, TensorNetCore
from .tensornet2 import DEFAULT_BUILD_PARAMS_2, DEFAULT_LAYER_PARAMS_2, TensorNet2Core

__all__ = [
    "TensorNetCore",
    "TensorNet2Core",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "DEFAULT_BUILD_PARAMS_2",
    "DEFAULT_LAYER_PARAMS_2",
]
