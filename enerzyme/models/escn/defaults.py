"""Native paper eSCN Core (Passaro & Zitnick, 2023).

Message-passing Core: emits ``atom_feature`` (spherical ``l=0``) and
``atom_sphere_feature`` (full SH coeffs, shape ``(N, (lmax+1)**2, C)``).
Message SO(2) may use ``mmax < lmax`` on edges; after ``rotate_inv`` node
features are always full degree. Default energy/forces use shared
``SimpleReadout`` + ``EnergyReduce`` + ``Force`` (requires differentiable edge
frames in ``so3.rotation``). Opt-in ``SphereSampleReadout`` consumes
``atom_sphere_feature`` for paper-style S² integration; optional
``vector_output_fields`` mirror the paper direct-force head.

Distinct from ``uma_qs`` and from native eSEN (``architecture: esen``).
"""

from __future__ import annotations




DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "polynomial",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"cutoff_fn": "polynomial", "trainable": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 64,
            "hidden_channels": 128,
            "edge_channels": 64,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
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
