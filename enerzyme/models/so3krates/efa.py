"""So3krates + Euclidean Fast Attention defaults (architecture: efa).

Local dual-stream So3krates Core with EFA on selected layers (MD17-style
``era_use_in_iterations=[0, 1]``, Lebedev 146, ``b_max=3π``).
"""

from __future__ import annotations

import math

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 132,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BernsteinRBF",
        "params": {"cutoff_fn": "cosine"},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "degrees": [1, 2, 3],
            "num_features": 132,
            "num_heads": 4,
            "num_layers": 3,
            "message_normalization": "avg_num_neighbors",
            "initialize_ev_to_zeros": True,
            "era_use_in_iterations": [0, 1],
            "era_lebedev_num": 146,
            "era_max_frequency": float(3 * math.pi),
            "era_max_length": 10.0,
            "era_qk_num_features": 32,
            "era_v_num_features": 16,
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
