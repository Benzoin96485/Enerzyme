"""QuinNet: quintuple interactions on an equivariant transformer.

Wang et al., NeurIPS 2023. The radial basis, the first species embedding,
the cutoff window, and the scalar energy head stay in the shared layer
stack. The neighbor embedding, the edge embedding, and the interaction
stack are the Core.
"""

from .defaults import (
    DEFAULT_BUILD_PARAMS,
    DEFAULT_LAYER_PARAMS,
    PUBLISHED_MD17_BUILD_PARAMS,
    PUBLISHED_MD17_CORE_PARAMS,
)

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_MD17_BUILD_PARAMS",
    "PUBLISHED_MD17_CORE_PARAMS",
]
