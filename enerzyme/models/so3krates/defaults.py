"""So3krates default stack and message normalization."""

# Typical MD17 / small-molecule average degree (shared ballpark with Equiformer).
_AVG_NUM_NEIGHBORS = 15.57930850982666

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 132,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BernsteinRBF",
        "params": {"cutoff_fn": "cosine"},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "degrees": [1, 2, 3],
            "num_features": 132,
            "num_heads": 4,
            "num_layers": 3,
            "message_normalization": "avg_num_neighbors",
            "initialize_ev_to_zeros": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
