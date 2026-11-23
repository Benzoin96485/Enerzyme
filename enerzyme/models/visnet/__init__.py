"""ViSNet: equivariant vector-scalar interactive message passing.

Wang et al., Nature Communications 2024 (doi:10.1038/s41467-023-43720-2).
The radial basis, the first species embedding, the cutoff window, and the
scalar energy head stay in the shared layer stack. The neighbor embedding,
the edge state, and the ViS-MP stack are the Core.
"""

from .core import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MD17_BUILD_PARAMS,
    PUBLISHED_MD17_CORE_PARAMS,
    PUBLISHED_QM9_BUILD_PARAMS,
    PUBLISHED_QM9_CORE_PARAMS,
    ViSNetCore,
)

__all__ = [
    "ViSNetCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_MD17_BUILD_PARAMS",
    "PUBLISHED_MD17_CORE_PARAMS",
    "PUBLISHED_QM9_BUILD_PARAMS",
    "PUBLISHED_QM9_CORE_PARAMS",
]
