"""Default DPA4 layer stack and build parameters."""

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "channels": 32,
            "lmax": 2,
            "mmax": 1,
            "n_blocks": 2,
            "mixing_layers": 3,
            "n_focus": 1,
            "focus_dim": 0,
            "n_atten_head": 1,
            "radial_so2_rank": 1,
            "radial_so2_mode": "degree_channel",
            "ffn_so3_grid": True,
            "message_node_so3": False,
            "use_env_seed": True,
            "n_radial": 16,
            "basis_type": "bessel",
            "sandwich_norm": [False, True, True, False],
            "ffn_blocks": 1,
            "ffn_neurons": 96,
            "glu_activation": True,
            "lebedev_quadrature": True,
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
