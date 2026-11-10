"""EquFlash: E(3)-equivariant tensor-product message passing.

Samsung Electronics GGNN (https://github.com/SamsungDS/GGNN). The Core is the
convolution stack only. Embeddings, radial basis, and property heads are
shared layers.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, EquFlashCore

__all__ = [
    "EquFlashCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
