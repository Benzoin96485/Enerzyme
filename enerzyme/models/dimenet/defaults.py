"""Default DimeNet and DimeNet++ layer stacks."""

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 6,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    "cutoff_fn": "polynomial",
}

_SWISH = {
    "dim_feature": 1,
    "initial_alpha": 1,
    "initial_beta": 1,
    "learnable": False,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "dimenet",
            "trainable": True,
            "envelope_exponent": 5,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_blocks": 6,
            "num_bilinear": 8,
            "num_spherical": 7,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "envelope_exponent": 5,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
        },
    },
    {
        "name": "HierachicalReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "num_hidden_layers": 3,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
            "initial_weight": "semi_orthogonal_glorot",
            "use_bias_out": False,
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


DEFAULT_BUILD_PARAMS_PP = dict(DEFAULT_BUILD_PARAMS)

DEFAULT_LAYER_PARAMS_PP = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "dimenet",
            "trainable": True,
            "envelope_exponent": 5,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_blocks": 4,
            "out_emb_size": 256,
            "int_emb_size": 64,
            "basis_emb_size": 8,
            "num_spherical": 7,
            "num_before_skip": 1,
            "num_after_skip": 2,
            "envelope_exponent": 5,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
        },
    },
    {
        "name": "HierachicalReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "num_hidden_layers": 3,
            "activation_fn": "swish",
            "activation_params": dict(_SWISH),
            "initial_weight": "semi_orthogonal_glorot",
            "use_bias_out": False,
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
