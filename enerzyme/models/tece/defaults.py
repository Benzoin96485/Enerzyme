"""TECE Core — Edge Cluster Expansion with Radial Rotary Attention.

Paper: Xu et al., arXiv:2607.10664
Upstream: https://github.com/xvzemin/tace (MIT)

Feature-mode Core only: emits ``atom_feature`` for external ``SimpleReadout``.
Embeddings / energy-force heads / physics priors stay outside the Core.
"""

from __future__ import annotations




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
            "num_layers": 2,
            "num_channel": 48,
            "Lmax": 2,
            "lmax": 2,
            "mmax": 2,
            "correlation": 2,
            "avg_num_neighbors": 8.0,
            "edge_embedding": "nonlinear",
            "edge_update": "element2",
            "scatter_norm": "avg_num_neighbors",
            "resnet_type": "BB",
            "resnet_linear_type": "aware",
            "use_first_resnet": False,
            "nonlinear": "sigmoid_gate",
            "edge_nonlinear": "so2_sigmoid_gate",
            "radial_mlp": [64, 64],
            "use_so2_edge_ace": True,
            "use_graph_softmax": True,
            "use_radial_phase": True,
            "use_temperature": True,
            "num_head": 1,
            "so2_linear_type": "w1",
            "gate_m0": False,
            "wigner_type": "recursive",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense", "keep_feature": False},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
