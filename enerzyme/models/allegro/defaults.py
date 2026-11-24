"""Allegro Core (Musaelian et al., Nat. Commun. 2023).

The Core maps a two-body scalar embedding and the short-range displacement
onto the latent pair features. Species tables, the Bessel × polynomial basis,
the pair-energy MLP, the directed-edge sum, per-element shifts, and
``Fa = -∇E`` stay outside.

Each layer tensor-products the current pair features with a sum of weighted
spherical harmonics over the center atom's neighbors. Scalar channels of
every layer are concatenated (the Allegro dense scalar track) and returned
as ``edge_feature``. One layer builds three-body features; each added layer
raises the body order by one. The receptive field never grows past
``cutoff_sr``.

The tensor product follows ``mir-group/allegro`` ``Allegro_Module`` (MIT).
Strided, Triton, and cuequivariance kernels are not used, and checkpoints
from that package are not loaded.
"""

from __future__ import annotations

from typing import Dict, Optional




# Widths from mir-group/allegro ``configs/tutorial.yaml``.
_PRESETS = {
    "tutorial": {
        "num_layers": 2,
        "l_max": 1,
        "parity": True,
        "num_scalar_features": 64,
        "num_tensor_features": 32,
        "scalar_embed_mlp_depth": 1,
        "scalar_embed_mlp_width": 64,
        "latent_mlp_depth": 1,
        "latent_mlp_width": 64,
        "nonlinearity": "silu",
        "tp_path_channel_coupling": True,
        "weight_individual_irreps": True,
    },
}

_DEFAULTS = dict(_PRESETS["tutorial"])


DEFAULT_BUILD_PARAMS = {
    "dim_edge": 64,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "avg_num_neighbors": 1.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
    },
    {"name": "PairChemicalRadial"},
    {"name": "Core"},
    {
        "name": "EdgeReadout",
        "params": {
            "output_fields": ["Eij"],
            "head_type": "variance_preserving",
            "hidden_layers_depth": 1,
            "nonlinearity": "silu",
            "use_bias": False,
        },
    },
    {"name": "EdgeEnergyReduce"},
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Ea": {"values": 0, "learnable": True}},
            "scales": {"Ea": {"values": 1, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _resolve(preset: Optional[str], overrides: Dict[str, object]) -> Dict[str, object]:
    resolved = dict(_DEFAULTS)
    if preset is not None:
        key = str(preset).lower()
        if key not in _PRESETS:
            known = ", ".join(_PRESETS)
            raise ValueError(f"unknown Allegro preset {preset!r}; expected {known}")
        resolved.update(_PRESETS[key])
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value
    return resolved
