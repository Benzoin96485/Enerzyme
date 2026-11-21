"""ALIGNN interatomic potential.

Choudhary and DeCost, npj Computational Materials 7, 185 (2021). The
edge-gated update follows
`usnistgov/alignn <https://github.com/usnistgov/alignn>`_, including
``ALIGNNAtomWise``. That package is not imported, and its checkpoints are
not loaded.

The Core reads ``GaussianRBF`` with ``flavor: alignn`` and
``RandomAtomEmbedding``. The angle Gaussian on ``[-1, 1]`` is inside the
Core. The energy head is a linear ``SimpleReadout``. Summing it is the
released average-pool times the atom count, including the bias.
"""

from .core import ALIGNNCore


# alignn_atomwise defaults: 2 ALIGNN layers, 2 gated convolutions, width 64,
# 80 bond Gaussians, 40 angle Gaussians, LayerNorm, 8 Å cutoff.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 80,
    "num_angle_rbf": 40,
    "embedding_features": 64,
    "hidden_features": 64,
    "max_Za": 94,
    "cutoff_sr": 8.0,
    # Energies are eV and distances are Ångström. One Hartree is 27.211 eV.
    # One Bohr is 0.529 Å, so a later pair prior sees Ångström coordinates.
    "Hartree_in_E": 27.211386245988,
    "Bohr_in_R": 0.5291772105638411,
}

# Table 1 of the property-prediction paper. The released code still updates
# the atom graph before the line graph; ``norm: batch`` is that paper's
# normalization. The force field above uses LayerNorm and a narrower stack.
PUBLISHED_PROPERTY_CORE_PARAMS = {
    "alignn_layers": 4,
    "gcn_layers": 4,
    "hidden_features": 256,
    "embedding_features": 64,
    "num_angle_rbf": 40,
    "norm": "batch",
    "update_order": "code",
}

_ENERGY = {
    "output_fields": ["Ea"],
    "head_type": "dense",
    "use_bias": True,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "GaussianRBF",
        "params": {"flavor": "alignn", "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "alignn_layers": 2,
            "gcn_layers": 2,
            "norm": "layer",
            "update_order": "code",
            "multiply_envelope": False,
        },
    },
    {"name": "SimpleReadout", "params": dict(_ENERGY)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

__all__ = [
    "ALIGNNCore",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "PUBLISHED_PROPERTY_CORE_PARAMS",
]
