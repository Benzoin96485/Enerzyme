"""EQNorm Core.

Chen, Fu, Yang, and Chen, https://github.com/yzchen08/eqnorm (MIT). A
dual-branch E(3) network: one branch keeps high-order tensor features, and
the other accumulates even scalars read out of that tensor state. The
scalar branch is the latent ``atom_feature``. One-hot species, the Bessel
basis, the polynomial window, the energy MLP, per-element shifts, and
``Fa = -dE/dR`` stay outside this module.

The checked-in widths are a small stack. Published MPtrj / OMat / max
settings are :data:`PUBLISHED_MPTRJ_CORE_PARAMS`,
:data:`PUBLISHED_OMAT_CORE_PARAMS`, and :data:`PUBLISHED_MAX_CORE_PARAMS`.
Reference checkpoints are not loaded.
"""

from __future__ import annotations





# eqnorm/model_settings/eqnorm-mptrj.yaml. avg_num_neighbors is MPtrj at 6 Å.
PUBLISHED_MPTRJ_CORE_PARAMS = {
    "num_layers": 4,
    "lmax": 3,
    "hidden_irreps": "128x0e+64x1o+32x2e+32x3o",
    "num_features": 128,
    "invariant_layers": 2,
    "invariant_neurons": 64,
    "avg_num_neighbors": 61.84,
    "normalization": "constant",
    "message_envelope": False,
    "polynomial_p": 6,
}
# eqnorm-omat.yaml: per-node degree normalization, no dataset neighbor mean.
PUBLISHED_OMAT_CORE_PARAMS = {
    **PUBLISHED_MPTRJ_CORE_PARAMS,
    "avg_num_neighbors": None,
    "normalization": "degree",
}
# eqnorm-max-mptrj.yaml: the same MPtrj stack with the envelope on the message.
PUBLISHED_MAX_CORE_PARAMS = {
    **PUBLISHED_MPTRJ_CORE_PARAMS,
    "message_envelope": True,
}


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 4,
    "num_rbf": 8,
    "max_Za": 8,
    "cutoff_sr": 6.0,
    "cutoff_fn": "nequip",
    "atomic_numbers": [1, 6, 7, 8],
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "equflash",
            "trainable": True,
            "apply_cutoff_fn": True,
        },
    },
    {"name": "OneHotAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 1,
            "hidden_irreps": "8x0e+4x1o",
            "num_features": 8,
            "invariant_layers": 2,
            "invariant_neurons": 16,
            "avg_num_neighbors": PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"],
            "normalization": "constant",
            "message_envelope": False,
            "polynomial_p": 6,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "variance_preserving",
            "use_bias": False,
            "nonlinearity": "silu",
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
