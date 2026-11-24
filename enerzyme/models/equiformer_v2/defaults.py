"""EquiformerV2 Core (Liao et al., ICLR 2024).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs). Embedding / RBF stay in pre-core
layers; energy / force / charge heads stay in post-core layers.

Adapted from https://github.com/atomicarchitects/equiformer_v2 (MIT license).
Distinct from Equiformer V1 (``enerzyme.models.equiformer``) and paper eSCN.
"""

from __future__ import annotations




# Typical MD17 / small-molecule average degree (same ballpark as Equiformer V1).
_AVG_DEGREE = 15.57930850982666

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "polynomial",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "GaussianRBF", "params": {"flavor": "SchNet", "apply_cutoff_fn": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 64,
            "attn_hidden_channels": 64,
            "num_heads": 4,
            "attn_alpha_channels": 32,
            "attn_value_channels": 16,
            "ffn_hidden_channels": 128,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "edge_channels": 64,
            "norm_type": "rms_norm_sh",
            "use_atom_edge_embedding": True,
            "use_gate_act": False,
            "use_grid_mlp": False,
            "use_sep_s2_act": True,
            "use_attn_renorm": True,
            "alpha_drop": 0.0,
            "drop_path_rate": 0.0,
            "proj_drop": 0.0,
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
