"""EquFlash and EquFlashV2.

Both Cores share this package: :mod:`enerzyme.models.equflash.irreps`, the
``equflash`` Bessel flavor, and the XPLOR cutoff. EquFlash is the uvu
convolution in :mod:`enerzyme.models.equflash.core`. EquFlashV2 is the
channelwise metaformer in :mod:`enerzyme.models.equflash.core_v2`. The
package-level defaults belong to EquFlash; EquFlashV2 defaults stay on
``core_v2`` so each architecture loads its own layer stack.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, EquFlashCore
from .core_v2 import EquFlashV2Core

__all__ = [
    "EquFlashCore",
    "EquFlashV2Core",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
