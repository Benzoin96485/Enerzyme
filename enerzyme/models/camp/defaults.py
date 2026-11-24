"""CAMP pipeline defaults."""

DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 33,
    "cutoff_sr": 5.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "water"}},
    {"name": "CAMPEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
