"""EquiformerV3 Core (Liao et al., 2026, arXiv:2604.09130).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs). Embedding / RBF stay in pre-core
layers; energy / force / charge heads stay in post-core layers.

Adapted from https://github.com/atomicarchitects/equiformer_v3 (MIT license).
Distinct from EquiformerV2 (``enerzyme.models.equiformer_v2``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence



# Typical MD17 / small-molecule average degree (same ballpark as Equiformer V2).
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
            "ffn_hidden_channels": 256,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "edge_channels": 64,
            "norm_type": "merge_layer_norm",
            "use_atom_edge_embedding": True,
            "use_envelope": True,
            "attn_activation": "sep-merge_gates2_swiglu",
            "ffn_activation": "sep-merge_gates2_swiglu",
            "use_grid_mlp": True,
            "use_attn_renorm": True,
            "attn_grid_resolution": [8, 8],
            "ffn_grid_resolution": [8, 8],
            "alpha_drop": 0.0,
            "drop_path_rate": 0.0,
            "proj_drop": 0.0,
            "attn_weights_drop": 0.0,
            "ffn_drop": 0.0,
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


def _as_resolution_list(value: Optional[Sequence[int]], default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    value = list(value)
    if len(value) == 1:
        return [value[0], value[0]]
    if len(value) != 2:
        raise ValueError(f"grid resolution must be length 1 or 2, got {value}")
    return value
