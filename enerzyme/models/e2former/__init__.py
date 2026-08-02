"""E2Former: efficient E(3)-equivariant transformer with Wigner-6j tensor products.

Li et al., NeurIPS 2025 Spotlight (arXiv:2501.19216).
E2Former-V2: Huang et al., arXiv:2601.16622 (SO2/EAAS + optional Triton).
Adapted from https://github.com/liyy2/E2Former and
https://github.com/IQuestLab/UBio-MolFM (MIT).
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, E2FormerCore
from .v2 import (
    DEFAULT_BUILD_PARAMS as V2_DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS as V2_DEFAULT_LAYER_PARAMS,
)

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "E2FormerCore",
    "V2_DEFAULT_BUILD_PARAMS",
    "V2_DEFAULT_LAYER_PARAMS",
]
