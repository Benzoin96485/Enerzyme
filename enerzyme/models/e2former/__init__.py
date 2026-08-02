"""E2Former: efficient E(3)-equivariant transformer with Wigner-6j tensor products.

Li et al., NeurIPS 2025 Spotlight (arXiv:2501.19216).
E2Former-LSR: Wang et al., arXiv:2601.03774 (fragment long-range).
Adapted from https://github.com/liyy2/E2Former and
https://github.com/IQuestLab/UBio-MolFM tag E2Former-LSR (MIT).
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, E2FormerCore
from .lsr import (
    DEFAULT_BUILD_PARAMS as LSR_DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS as LSR_DEFAULT_LAYER_PARAMS,
    E2FormerLSRCore,
)

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "E2FormerCore",
    "E2FormerLSRCore",
    "LSR_DEFAULT_BUILD_PARAMS",
    "LSR_DEFAULT_LAYER_PARAMS",
]
