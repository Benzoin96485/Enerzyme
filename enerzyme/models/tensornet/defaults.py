"""Default layer stacks for TensorNet and TensorNet2."""

_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 4.5,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 2,
            "equivariance_group": "O(3)",
            "charge_scale": 0.1,
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


DEFAULT_BUILD_PARAMS_2 = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 4.5,
    "cutoff_fn": "cosine",
    "q_dim": 16,
}

# Stage weights 1, 2, 3 give the last charge block half of the Coulomb average
# (1+2+3). AceFF states that the last layer dominates; it does not publish the
# numeric weights, so this schedule is the Enerzyme default and can be replaced.
DEFAULT_LAYER_PARAMS_2 = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 2,
            "equivariance_group": "O(3)",
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
    {
        "name": "WeightedCoulombEnergy",
        "params": {
            "layer_weights": [1.0, 2.0, 3.0],
            "damping_cutoff": 4.6,
            "pairs": "all_to_all",
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
