"""GRACE pipeline defaults."""

DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 128,
    "cutoff_sr": 6.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "1L"}},
    {"name": "GRACEEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
