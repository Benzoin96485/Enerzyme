"""REANN model and pipeline defaults."""

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
}

_READOUT = {
    "output_fields": ["Ea"],
    "head_type": "mlp",
    "hidden_dims": [128, 128],
    "activation_fn": "relu_like",
    "activation_params": {
        "dim_feature": 1,
        "initial_alpha": 1.0,
        "initial_beta": 1.0,
        "learnable": False,
    },
    "keep_feature": False,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "lmax": 2,
            "nwave": 8,
            "num_iterations": 1,
            "nblock": 1,
            "hidden_dims": [128, 128],
            "activation": "relu_like",
            "layer_norm": True,
            "dropout": 0.0,
            "elemental": False,
            "coeff_init": 0.1,
        },
    },
    {"name": "SimpleReadout", "params": dict(_READOUT)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

# MD17 column of Table I. ``elemental: true`` also needs ``atomic_numbers``
# for the species present in that molecule. The energy network is the shared
# readout, with the same hidden widths.
PUBLISHED_MD17_CORE_PARAMS = {
    "lmax": 2,
    "nwave": 20,
    "num_iterations": 1,
    "nblock": 1,
    "hidden_dims": [128, 128],
    "activation": "relu_like",
    "layer_norm": True,
    "dropout": 0.0,
    "elemental": True,
}
PUBLISHED_MD17_CUTOFF = 8.0
