_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# examples/ViSNet-MD17.yml, ViSNet-rMD17.yml, and ViSNet-MD22.yml.
# max_z: 100 there is the embedding length (Z = 0..99). This stack follows
# the other Cores: max_Za is the largest atomic number, so the table has
# max_Za + 1 rows.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 32,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_MD17_BUILD_PARAMS = dict(DEFAULT_BUILD_PARAMS)

PUBLISHED_MD17_CORE_PARAMS = {
    "num_layers": 9,
    "num_heads": 8,
    "lmax": 2,
    "vertex_type": "none",
    "vecnorm_type": "none",
    "trainable_vecnorm": False,
    "neighbor_embedding": True,
    "include_self": True,
    "activation": "silu",
    "attn_activation": "silu",
}

# examples/ViSNet-QM9.yml. Atomref is a data shift, not a Core parameter.
PUBLISHED_QM9_BUILD_PARAMS = {
    "dim_embedding": 512,
    "num_rbf": 64,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_QM9_CORE_PARAMS = {
    **PUBLISHED_MD17_CORE_PARAMS,
    "vecnorm_type": "max_min",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": dict(PUBLISHED_MD17_CORE_PARAMS)},
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
