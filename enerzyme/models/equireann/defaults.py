"""EQUIREANN model and pipeline defaults."""

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

_READOUT = {
    "output_fields": ["Ea"],
    "head_type": "mlp",
    "hidden_dims": [32, 32],
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
            "nwave": 20,
            "num_iterations": 2,
            "nblock": 1,
            "hidden_dims": [32, 32],
            "activation": "tanh_like",
            "layer_norm": False,
            "dropout": 0.0,
            "elemental": False,
            "coeff_init": 0.1,
        },
    },
    {"name": "SimpleReadout", "params": dict(_READOUT)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

# Cumulene section of the paper: Nφ = 8, L = 2, T = 2, 16×16 networks, rc = 6 Å.
# Eq. 1 uses Nφ for the contracted width. The NequIP timing paragraph also calls
# 8 the basis count, so nwave is set to 8 here. para/input_density instead uses
# nwave 20 and the default orbit formula; that is the checked-in example.
PUBLISHED_CUMULENE_CORE_PARAMS = {
    "lmax": 2,
    "nwave": 8,
    "n_orbit": 8,
    "num_iterations": 2,
    "nblock": 1,
    "hidden_dims": [16, 16],
    "activation": "tanh_like",
    "layer_norm": False,
    "dropout": 0.0,
    "elemental": False,
}
PUBLISHED_CUMULENE_CUTOFF = 6.0
