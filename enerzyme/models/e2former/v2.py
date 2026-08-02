"""E2Former-V2 defaults (Huang et al., 2026, arXiv:2601.16622).

Reuses :class:`~enerzyme.models.e2former.core.E2FormerCore` with SO2 / EAAS
attention (:code:`attn_type: so2-first-order`) and optional Triton sparse
kernels (:code:`tp_type: QK_alpha+triton`). Embeddings and property heads stay
outside the Core. Distinct from E2Former-V1 (Wigner-6j ``first-order``).

Adapted from https://github.com/IQuestLab/UBio-MolFM (MIT).
"""

from __future__ import annotations

from .core import DEFAULT_BUILD_PARAMS as _V1_BUILD

DEFAULT_BUILD_PARAMS = dict(_V1_BUILD)

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "GaussianSmearing"},
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
            "attn_type": "so2-first-order",
            "tp_type": "QK_alpha+triton",
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
