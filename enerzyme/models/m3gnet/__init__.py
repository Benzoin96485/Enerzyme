"""M3GNet interatomic potential.

Chen and Ong, Nature Computational Science 2, 718–728 (2022). The graph
convolution follows `materialyzeai/matgl <https://github.com/materialyzeai/matgl>`_
(BSD). That package is not imported, and its checkpoints are not loaded.

The Core reads ``SphericalBesselRBF`` and ``RandomAtomEmbedding``. The
potential head is ``SimpleReadout`` with ``head_type: gated``.
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
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_blocks": 3,
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
