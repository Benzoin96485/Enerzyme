"""Strictly local Allegro pair potential.

The Core is the tensor track only. Pair species embedding, the radial basis,
and the energy head are ordinary layer-stack modules.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, AllegroCore

__all__ = ["DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS", "AllegroCore"]
