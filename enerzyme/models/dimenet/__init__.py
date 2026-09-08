"""DimeNet (Gasteiger et al., ICLR 2020) — original directional MPNN.

Not DimeNet++ (bilinear SBF interaction, not Hadamard down-projection).
"""

from .core import DimeNetCore, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [
    "DimeNetCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
