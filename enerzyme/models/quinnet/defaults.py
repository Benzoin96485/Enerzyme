_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# example/hparams_aspirin.yaml. max_z: 100 there is Embedding(100), so
# Z = 0..99. This stack follows the other Cores: max_Za is the largest
# atomic number, and the table has max_Za + 1 rows.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 32,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_MD17_BUILD_PARAMS = dict(DEFAULT_BUILD_PARAMS)

PUBLISHED_MD17_CORE_PARAMS = {
    "num_layers": 5,
    "num_heads": 8,
    "distance_influence": "both",
    "neighbor_embedding": True,
    "include_self": True,
    "activation": "silu",
    "attn_activation": "silu",
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
