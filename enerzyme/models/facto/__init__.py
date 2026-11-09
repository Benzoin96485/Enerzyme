"""FACTO: fast atom-centered tensor operator (Liu et al., arXiv:2611.03276).

No upstream implementation is published. This package is the FP32 reference:
atom-centered scalar and tensor updates, with the energy head and ZBL
repulsion outside the Core. It does not include the paper's CUDA aggregation
kernels or BF16×3 matmul emulation.
"""

from .core import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    FACTOCore,
    PAPER_PRESETS,
)
from .interaction import RESIDUAL_SCALE

__all__ = [
    "FACTOCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PAPER_PRESETS",
    "RESIDUAL_SCALE",
]
