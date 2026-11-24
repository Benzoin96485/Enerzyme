"""CACE pipeline defaults."""

DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 3,
    "num_rbf": 6,
    "cutoff_sr": 5.5,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "default",
            "trainable": True,
            "apply_cutoff_fn": False,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "water"}},
    {"name": "CACEEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
