"""MACE defaults. The Core is :func:`enerzyme.models.presets.irreps.mace`.

``output_mode="direct"`` keeps the per-layer energy and charge head in the
Core. ``output_mode="feature"`` returns the last interaction.
``output_mode="layers"`` returns every interaction as ``atom_feature`` of
shape ``[N, irreps, T]``, which
:class:`~enerzyme.models.mace.readout.MACEEnergyReadout` and
:class:`~enerzyme.models.mace.polar.PolarFieldLayer` consume.
"""

import torch

# Small stack. MACE-MP / MACE-POLAR-1-L use 512 channels, correlation 3,
# max_ell 3, and a 6 Å cutoff. Set avg_num_neighbors to the dataset mean.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "avg_num_neighbors": 8.0,
}
DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "mace", "trainable": False, "polynomial_p": 5},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "max_ell": 2,
            "additional_hidden_irreps": "32x1o",
            "correlation": 2,
            "num_interactions": 2,
            "output_mode": "layers",
            "keep_last_layer_irreps": True,
        },
    },
    {"name": "MACEEnergyReadout"},
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
            "scales": {"Ea": {"values": 1.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
GATE_FUNCTIONS = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}
