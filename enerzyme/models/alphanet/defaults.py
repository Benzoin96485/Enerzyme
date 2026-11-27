DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 96,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "equflash", "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 3,
            "head": 16,
            "main_chi1": 24,
            "mp_chi1": 24,
            "chi2": 6,
            "hidden_channels_chi": 96,
            "has_dropout_flag": True,
            "has_norm_before_flag": True,
            "has_norm_after_flag": False,
            "reduce_mode": "sum",
            "fte_scalarization": "l2",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense"},
    },
    {
        "name": "AtomicAffine",
        "params": {
            "scales": {"Ea": {"values": 1, "learnable": True}},
            "shifts": {"Ea": {"values": 0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
