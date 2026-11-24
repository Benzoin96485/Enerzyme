"""MatterSim potential: Graphormer-extended M3GNet.

Yang et al., arXiv:2405.04967. The preset extends M3GNet with the materials
Graphormer structural encoder. ``SphericalBesselRBF`` is the M3GNet bond
basis. ``GaussianRBF`` (SchNet width, no envelope) is the attention basis;
the Core renames its output to ``graphormer_rbf``. The potential head is
``SimpleReadout`` with ``head_type: gated``.

``microsoft/mattersim`` checkpoints (MatterSim-v1 M3GNet) do not load. The
public repository does not ship the Graphormer weights.
"""

_GATED = {
    "output_fields": ["Ea"],
    "head_type": "gated",
    "num_hidden_layers": 2,
    "activate_last": False,
    "use_bias": True,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 3,
    "max_l": 3,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "threebody_cutoff": 4.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "SphericalBesselRBF"},
    {
        "name": "GaussianRBF",
        "params": {
            "num_rbf": 16,
            "flavor": "SchNet",
            "apply_cutoff_fn": False,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_blocks": 3,
            "num_attn_layers": 2,
            "num_heads": 4,
            "graphormer_num_basis": 16,
            "attention_cutoff": 5.0,
            "use_centrality": True,
            "envelope_exponent": 3,
        },
    },
    {"name": "SimpleReadout", "params": dict(_GATED)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
