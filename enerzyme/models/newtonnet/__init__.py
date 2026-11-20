"""NewtonNet: Newtonian message passing for energies and forces.

Haghighatlari et al., Digital Discovery 2022 (arXiv:2108.02913,
doi:10.1039/D2DD00008C). The Core is the interaction stack. The species
embedding, Bessel basis, energy head, and conservative forces stay in the
shared layer stack. ``EquivariantDipole`` and ``EquivariantPolarizability``
read the Cartesian latent pair.
"""

from .core import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS, NewtonNetCore
from .loss import LOSS_REGISTER

__all__ = [
    "NewtonNetCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "LOSS_REGISTER",
]
