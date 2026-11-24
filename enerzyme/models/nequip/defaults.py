"""NequIP Core (Batzner et al., Nat. Commun. 2022).

The Core is the equivariant convolution stack only. It reads a scalar species
embedding and a radial basis on the short-range graph, and returns the latent
``atom_feature``. Chemical embedding, the Bessel × polynomial edge embedding,
the scalar energy head, per-element shifts, and ``Fa = -∇E`` stay in the
layer stack.

The convolution matches current ``mir-group/nequip`` ``InteractionBlock`` and
``ConvNetLayer`` (and the 2022 paper): spherical harmonics of
``R_sender - R_receiver``, a radial MLP on the edge embedding, a Clebsch–Gordan
tensor product, a self-connection against the fixed species embedding, and a
gate. The first layer omits the self-connection unless ``sc_first`` is set, so
an isolated atom stays at zero and its reference energy belongs in
``AtomicAffine``. Checkpoints from the ``nequip`` package are not loaded.
"""

from __future__ import annotations

from typing import Dict, Optional




# Foundation-model sizes from mir-group/nequip ``NequIPGNNModel`` presets.
# ``type_embed_num_features`` stays ``dim_embedding`` on the layer stack (32
# in those presets). ``paper`` is the 2022 molecular setting: both parities,
# two hidden radial layers of width 64.
_PRESETS = {
    "S": {
        "num_layers": 2,
        "l_max": 1,
        "num_features": [128, 64],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "M": {
        "num_layers": 4,
        "l_max": 2,
        "num_features": [128, 64, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "L": {
        "num_layers": 6,
        "l_max": 3,
        "num_features": [128, 64, 32, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "XL": {
        "num_layers": 6,
        "l_max": 4,
        "num_features": [320, 96, 64, 32, 32],
        "parity": False,
        "radial_mlp_depth": 1,
        "radial_mlp_width": 128,
    },
    "paper": {
        "num_layers": 4,
        "l_max": 1,
        "num_features": 32,
        "parity": True,
        "radial_mlp_depth": 2,
        "radial_mlp_width": 64,
    },
}

_DEFAULTS = {
    "num_layers": 4,
    "l_max": 1,
    "num_features": 32,
    "parity": True,
    "radial_mlp_depth": 1,
    "radial_mlp_width": 128,
    "use_sc": True,
    "sc_first": False,
    "resnet": False,
    "resnet_first": False,
    "nonlinearity_type": "gate",
    "avg_num_neighbors": 1.0,
}


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 5.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 4,
            "l_max": 1,
            "parity": True,
            "num_features": 32,
            "radial_mlp_depth": 1,
            "radial_mlp_width": 128,
            "use_sc": True,
            "sc_first": False,
            "avg_num_neighbors": 1.0,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
        },
    },
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
        key = str(preset).upper() if str(preset).upper() in _PRESETS else str(preset).lower()
        if key == "PAPER":
            key = "paper"
        if key not in _PRESETS:
            known = ", ".join([*list(_PRESETS)[:-1], "paper"])
            raise ValueError(f"unknown NequIP preset {preset!r}; expected {known}")
        resolved.update(_PRESETS[key])
    for name, value in overrides.items():
        if value is not None:
            resolved[name] = value
    return resolved
