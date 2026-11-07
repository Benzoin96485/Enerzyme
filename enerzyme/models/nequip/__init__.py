"""Native NequIP convolution stack.

The 2022 architecture is the Core in this package. The previous wrapper around
the external ``nequip`` package has been retired; install of that package is
not required.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, NequIPCore

__all__ = ["DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS", "NequIPCore"]
