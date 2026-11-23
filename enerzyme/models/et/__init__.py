"""TorchMD-NET Equivariant Transformer.

Thölke and De Fabritiis, ICLR 2022 (arXiv:2202.02541). The radial basis, the
first species embedding, the cutoff window, and the scalar energy head stay
in the shared layer stack. The neighbor embedding and the attention stack
are the Core.
"""

from .core import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_QM9_BUILD_PARAMS,
    PUBLISHED_QM9_CORE_PARAMS,
    ETCore,
)

__all__ = [
    "ETCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_QM9_BUILD_PARAMS",
    "PUBLISHED_QM9_CORE_PARAMS",
]
