_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# examples/ET-MD17.yaml. The class default in torchmd-net leaves
# vector_cutoff off; the published MD17 and QM9 runs turn it on.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 6,
            "num_heads": 8,
            "distance_influence": "both",
            "neighbor_embedding": True,
            "vector_cutoff": True,
            "include_self": True,
        },
    },
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

# examples/ET-QM9.yaml. Atomref is a data shift, not a Core parameter.
PUBLISHED_QM9_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 64,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_QM9_CORE_PARAMS = {
    "num_layers": 8,
    "num_heads": 8,
    "distance_influence": "both",
    "neighbor_embedding": True,
    "vector_cutoff": True,
    "include_self": True,
}
