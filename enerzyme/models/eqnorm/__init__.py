"""EQNorm dual-branch E(3) Core.

Reference: https://github.com/yzchen08/eqnorm (MIT). The package is not
imported, and its checkpoints are not loaded.
"""

from .defaults import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MAX_CORE_PARAMS,
    PUBLISHED_MPTRJ_CORE_PARAMS,
    PUBLISHED_OMAT_CORE_PARAMS,
)

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_MAX_CORE_PARAMS",
    "PUBLISHED_MPTRJ_CORE_PARAMS",
    "PUBLISHED_OMAT_CORE_PARAMS",
]
