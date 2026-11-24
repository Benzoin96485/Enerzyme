"""AllScAIP default build parameters and layer stack."""

# Defaults align with fairchem AllScAIP-style configs (Global / MolecularGraph / GNN / Regularization).
# Experimental Enerzyme adaptation (魔改): not a recommended primary production model yet.
DEFAULT_BUILD_PARAMS = {
    # Global
    "dim_embedding": 128,
    "num_layers": 4,
    "activation": "silu",
    "use_node_path": True,
    "use_residual_scaling": True,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    # Shared preprocessing / embeddings (Enerzyme pipeline)
    "num_rbf": 8,
    "max_Za": 86,
    "cutoff_sr": 5.0,
    "cutoff_fn": "polynomial",
    # Molecular graph
    "knn_k": 32,
    "knn_pad_size": None,
    "knn_soft": False,
    "knn_sigmoid_scale": 0.2,
    "knn_lse_scale": 0.1,
    "knn_use_low_mem": False,
    "use_envelope": True,
    "distance_function": "gaussian",
    "edge_distance_expansion_size": 8,
    "edge_direction_expansion_size": 3,
    "node_direction_expansion_size": 3,
    "preprocess_on_cpu": False,
    "pbc": False,
    "use_padding": False,
    "max_atoms": None,
    "max_batch_size": None,
    "single_system_no_padding": False,
    # GNN / attention
    "atten_num_heads": 4,
    "ffn_hidden_layer_multiplier": 2,
    "attn_num_freq": 32,
    "freequency_list": None,
    "use_freq_mask": True,
    "use_sincx_mask": True,
    # Regularization
    "normalization": "layer_norm",
    "mlp_dropout": 0.0,
    "atten_dropout": 0.0,
    # Equivariant LAE (TP + invariant scalars; disables SH frequency mask on Q/K)
    "equivariant_mode": False,
    "equivariant_tp_hidden": 32,
    "equivariant_tp_mul": 8,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "trainable": False,
        },
    },
    {
        "name": "NuclearEmbedding",
        "params": {
            "zero_init": False,
            "use_electron_config": True,
        },
    },
    {
        "name": "ElectronicEmbedding",
        "params": {
            "num_residual": 1,
            "attribute": "charge",
        },
    },
    {
        "name": "ElectronicEmbedding",
        "params": {
            "num_residual": 1,
            "attribute": "spin",
        },
    },
    {
        "name": "Core",
        "params": {},
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": {"Ea", "Qa"},
            "head_type": "residual_mlp",
            "num_residual": 2,
            "dim_embedding": 128,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {
                "Ea": {"values": 0, "learnable": True},
                "Qa": {"values": 0, "learnable": True},
            },
            "scales": {
                "Ea": {"values": 1, "learnable": True},
                "Qa": {"values": 1, "learnable": True},
            },
        },
    },
    {"name": "ChargeConservation"},
    {
        "name": "ElectrostaticEnergy",
        "params": {"flavor": "PhysNet", "cutoff_lr": None},
    },
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
