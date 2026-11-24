DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"alpha": 1.0, "trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {"num_layers": 4, "fte_scalarization": "l2"},
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense"},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
