"""BOTNet message-passing core.

Batatia et al., the design space of E(3)-equivariant atom-centred
interatomic potentials (Nat. Mach. Intell., doi:10.1038/s42256-024-00956-x,
arXiv:2205.06643). The reference implementation is
`gncs/botnet <https://github.com/gncs/botnet>`_.

Each interaction is one two-body Clebsch–Gordan convolution of the sender
features with real spherical harmonics. Stacking layers raises the body
order by one, and the site energy reads every layer. MACE keeps these
convolutions and inserts a symmetric product after each one; this Core does
not.

The Bessel × polynomial basis and the species embedding stay outside. The
per-layer energy head is :class:`~enerzyme.models.botnet.readout.BOTNetEnergyReadout`.
Species scale and shift are :class:`~enerzyme.models.layers.denormalize.AtomicAffineLayer`.
"""

from __future__ import annotations


from e3nn.o3 import Irreps



# CLI defaults in gncs/botnet (r_max 4 Å, 8 Bessel functions, polynomial
# order 6, 5 interactions, natural parity through l = 3, 32 channels).
# The checked-in stack is smaller. The 3BPA run in the README uses 80
# channels and both parities; later layers then grow extra irreps, which
# this Core zero-fills in the stacked latent.
PUBLISHED_CORE_PARAMS = {
    "max_ell": 3,
    "num_interactions": 5,
    "hidden_irreps": "32x0e+32x1o+32x2e+32x3o",
    "variant": "agnostic",
    "MLP_irreps": "16x0e",
    "radial_MLP": [64, 64, 64],
    "gate": "silu",
    "output_mode": "layers",
}
PUBLISHED_3BPA_HIDDEN_IRREPS = (
    "80x0e+80x0o+80x1o+80x1e+80x2e+80x2o+80x3o+80x3e"
)
PUBLISHED_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "cutoff_sr": 4.0,
    "polynomial_p": 6,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 16,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 4.0,
    "avg_num_neighbors": 1.0,
}
DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "mace", "trainable": False, "polynomial_p": 6},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "max_ell": 2,
            "num_interactions": 2,
            "variant": "agnostic",
            "output_mode": "layers",
            "MLP_irreps": "16x0e",
            "radial_MLP": [64, 64, 64],
            "gate": "silu",
        },
    },
    {
        "name": "BOTNetEnergyReadout",
        "params": {"MLP_irreps": "16x0e", "gate": "silu"},
    },
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
            "scales": {"Ea": {"values": 1.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def natural_hidden_irreps(num_channels: int, max_ell: int) -> str:
    """Even-scalar channels plus natural parity ``p = (-1)^l`` through ``max_ell``.

    A scalar embedding tensor-producted with ``Y_l`` produces exactly this set,
    and every later layer can produce it again, so the stacked latent has one
    irrep layout.
    """
    channels = int(num_channels)
    degree = int(max_ell)
    if channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if degree < 0:
        raise ValueError(f"max_ell must be >= 0, got {max_ell}")
    parts = []
    for ell in range(degree + 1):
        parity = "e" if ell % 2 == 0 else "o"
        parts.append(f"{channels}x{ell}{parity}")
    return str(Irreps("+".join(parts)).simplify())
