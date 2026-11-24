PUBLISHED_CORE_PARAMS = {
    "n_dim": 128,
    "e_dim": 64,
    "a_dim": 32,
    "nlayers": 6,
    "e_rcut": 6.0,
    "e_rcut_smth": 5.0,
    "e_sel": 120,
    "a_rcut": 4.0,
    "a_rcut_smth": 3.5,
    "a_sel": 20,
    "axis_neuron": 4,
    "update_style": "res_residual",
    "update_residual": 0.1,
    "update_residual_init": "const",
    "fix_stat_std": 0.3,
    "smooth_edge_update": True,
    "activation": "silu",
}


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
            "e_dim": 16,
            "a_dim": 16,
            "nlayers": 2,
            "e_sel": 16,
            "a_sel": 8,
            "axis_neuron": 4,
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
