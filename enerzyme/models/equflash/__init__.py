"""EquFlashV2: channelwise Clebsch-Gordan message passing.

Samsung Electronics' EquFlashV2 (https://github.com/SamsungDS/GGNN). The Core
is the interaction stack only. Embeddings, the radial window, and property
heads stay in the layer stack.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, EquFlashV2Core

__all__ = [
    "EquFlashV2Core",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
