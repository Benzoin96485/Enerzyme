"""TACE preset defaults."""

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 48,
    "num_rbf": 8,
    "max_Za": 86,
    "cutoff_sr": 6.0,
    "cutoff_fn": "polynomial",
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "BesselRBF", "params": {"trainable": False}},
    {
        "name": "NuclearEmbedding",
        "params": {
            "element_embedding_cueq_config": None,
            "compute_cueq_config": None,
        },
    },
    {
        "name": "Core",
        "params": {
            "tensor_basis": "spherical",
            "num_layers": 2,
            "num_channel": 48,
            "Lmax": 2,
            "lmax": 3,
            "correlation": 2,
            "avg_num_neighbors": 8.0,
            "parity": False,
            "edge_embedding": "nonlinear",
            "edge_update": "element2",
            "scatter_norm": "avg_num_neighbors",
            "resnet_type": "BB",
            "resnet_linear_type": "aware",
            "use_first_resnet": False,
            "nonlinear": "sigmoid_gate",
            "radial_mlp": [64, 64, 64],
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense", "keep_feature": False},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
