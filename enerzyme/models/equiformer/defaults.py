# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License)
# and adapted to Enerzyme BaseFFCore modular layer stack.
"""Equiformer Core: equivariant graph attention transformer latent."""




# MD17-style statistics (cutoff ~5 A); used for ScaledScatter normalization.
_AVG_DEGREE = 15.57930850982666
_AVG_NUM_NODES = 18.03065905448718

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 16,
    "max_Za": 86,
    "cutoff_sr": 5.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
    "irreps_node_embedding": "32x0e+16x1e+8x2e",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation", "params": {"cutoff_fn": "polynomial"}},
    {
        "name": "ExpNormalSmearing",
        "params": {"trainable": False},
    },
    {
        "name": "EquiformerNodeEmbedding",
        "params": {
            "irreps_node_embedding": "32x0e+16x1e+8x2e",
        },
    },
    {
        "name": "Core",
        "params": {
            "irreps_node_embedding": "32x0e+16x1e+8x2e",
            "irreps_feature": "64x0e",
            "irreps_sh": "1x0e+1x1e+1x2e",
            "irreps_head": "16x0e+8x1o+4x2e",
            "irreps_mlp_mid": "32x0e+16x1e+8x2e",
            "num_layers": 2,
            "num_heads": 2,
            "fc_neurons": [64, 64],
            "nonlinear_message": True,
            "use_attn_head": False,
            "alpha_drop": 0.0,
            "proj_drop": 0.0,
            "out_drop": 0.0,
            "drop_path_rate": 0.0,
            "avg_degree": _AVG_DEGREE,
            "avg_num_nodes": _AVG_NUM_NODES,
            "output_mode": "feature",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea", "Qa"],
            "head_type": "two_layer",
            "dim_embedding": 64,
            "activation_fn": "swish",
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
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
