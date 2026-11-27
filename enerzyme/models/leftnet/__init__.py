"""LEFTNet: local substructure encoding and frame transitions.

Du et al., NeurIPS 2023 (arXiv:2304.04757). The Core is the interaction
stack. The species embedding, radial basis, energy head, and conservative
forces stay in the shared layer stack.
"""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
