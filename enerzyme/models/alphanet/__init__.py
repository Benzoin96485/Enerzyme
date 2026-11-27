"""AlphaNet: LEFTNet local frames, rotary phases, and a temporal state.

Yin et al., npj Computational Materials 2025 (arXiv:2501.07155). The Core
is the interaction stack. The species embedding, Bessel basis, energy head,
and conservative forces stay in the shared layer stack.
"""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
