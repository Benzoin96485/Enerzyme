"""E2Former: efficient E(3)-equivariant transformer with Wigner-6j tensor products.

Li et al., NeurIPS 2025 Spotlight (arXiv:2501.19216).
Adapted from https://github.com/liyy2/E2Former (MIT).
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, E2FormerCore

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "E2FormerCore",
]
