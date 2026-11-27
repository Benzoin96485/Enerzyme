"""Geodite: equivariant message passing without tensor products.

Reschützegger et al., J. Chem. Theory Comput. 2026
(https://doi.org/10.1021/acs.jctc.6c00712). The Core is the interaction
stack. Radial window, isolated-atom residual, readout, and screened ZBL
stay outside it.
"""

from .defaults import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
