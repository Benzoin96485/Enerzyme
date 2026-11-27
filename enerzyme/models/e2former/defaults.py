"""E2Former Core (Li et al., NeurIPS 2025 Spotlight).

Wigner-6j equivariant transformer Core: emits ``atom_feature`` (spherical
``l=0``) and ``atom_sphere_feature`` (full SH coeffs). Embeddings / RBF stay
in pre-core layers; energy / force heads stay in post-core layers.

Adapted from https://github.com/liyy2/E2Former (MIT license).
"""

from __future__ import annotations




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
            "irreps_node_embedding": "64x0e+64x1e+64x2e",
            "irreps_head": "16x0e+16x1e+16x2e",
            "num_layers": 2,
            "num_attn_heads": 4,
            "attn_scalar_head": 32,
            "ffn_hidden_channels": 128,
            "max_neighbors": 32,
            "attn_type": "first-order",
            "tp_type": "QK_alpha",
            "norm_layer": "rms_norm_sh",
            "alpha_drop": 0.0,
            "proj_drop": 0.0,
            "use_atom_edge_embedding": True,
            "use_gate_act": False,
            "use_grid_mlp": False,
            "use_sep_s2_act": True,
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
