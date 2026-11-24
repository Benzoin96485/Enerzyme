"""Nequix Core.

Koker and Smidt, arXiv:2508.16067. A short stack of separable NequIP
convolutions with a species-independent residual, equivariant RMSNorm, and
a scalar final layer. One-hot species, the Bessel basis, the polynomial
cutoff, per-element energy shifts, and forces stay outside this module.

Reference code: https://github.com/atomicarchitects/nequix (MIT). The
published ``nequix-mp-1`` widths live in :data:`PUBLISHED_MP_CORE_PARAMS`.
The default widths are a small stack, so importing the architecture does not
compile the 700k-parameter tensor products. The neighbor normalization is the
published MPtrj mean, because the default species list and cutoff are the
MPtrj ones.
"""

from __future__ import annotations



from .species import MPTRJ_ATOMIC_NUMBERS


# Table A.1 and configs/nequix-mp-1.yml. avg_num_neighbors is the MPtrj mean.
PUBLISHED_MP_CORE_PARAMS = {
    "num_layers": 4,
    "lmax": 3,
    "hidden_irreps": "128x0e + 64x1o + 32x2e + 32x3o",
    "radial_mlp_size": 64,
    "radial_mlp_layers": 2,
    "mlp_init_scale": 4.0,
    "avg_num_neighbors": 57.413687022442645,
    "layer_norm": True,
}
PUBLISHED_MP_SCALE = 0.8066479563713074
PUBLISHED_MP_SHIFT = 0.16502578765761478


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": len(MPTRJ_ATOMIC_NUMBERS),
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "nequip",
    "atomic_numbers": list(MPTRJ_ATOMIC_NUMBERS),
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "equflash",
            "trainable": False,
            "apply_cutoff_fn": True,
        },
    },
    {"name": "OneHotAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 2,
            "hidden_irreps": "8x0e + 4x1o + 4x2e",
            "radial_mlp_size": 32,
            "radial_mlp_layers": 2,
            "mlp_init_scale": 4.0,
            "avg_num_neighbors": PUBLISHED_MP_CORE_PARAMS["avg_num_neighbors"],
            "layer_norm": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "order": "scale_then_shift",
            "scales": {"Ea": {"values": 1.0, "learnable": False}},
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
