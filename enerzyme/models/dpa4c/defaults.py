"""DPA4C descriptor profiles and YAML defaults."""

ENVELOPE_EXPONENT = 5
DISTANCE_EPS = 1.0e-7
NORM_FLOOR = 0.25
# A geometric column this small on the calibration set has no magnitude to
# match to the embedding. Leave its scale at 1 instead of dividing by it.
CALIBRATION_RMS_FLOOR = 1.0e-12

# Grades in Li et al. and the DeePMD-kit descriptor docs. ``fitting_hidden``
# is the width of the external residual MLP, three layers in the released nets.
PUBLISHED_GRADES = {
    "nano": {"channels": 8, "lmax": 2, "radial_modes": 0, "fitting_hidden": 96, "fitting_layers": 3},
    "mini": {"channels": 32, "lmax": 2, "radial_modes": 0, "fitting_hidden": 192, "fitting_layers": 3},
    "neo": {"channels": 32, "lmax": 2, "radial_modes": 4, "fitting_hidden": 192, "fitting_layers": 3},
    "air": {"channels": 64, "lmax": 3, "radial_modes": 4, "fitting_hidden": 256, "fitting_layers": 3},
    "plus": {"channels": 128, "lmax": 3, "radial_modes": 4, "fitting_hidden": 384, "fitting_layers": 3},
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "channels": 8,
            "lmax": 2,
            "radial_modes": 0,
            "n_radial": 16,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "hidden_dims": [96, 96, 96],
            "identity_residual": True,
            "use_bias_out": True,
            "activation_fn": "swish",
            "activation_params": {"initial_beta": 1.0, "learnable": False},
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
