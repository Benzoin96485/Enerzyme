"""GemNet-T, GemNet-Q, and GemNet-OC directional message passing.

``GemNetCore`` with ``triplets_only=True`` is GemNet-T. GemNet-Q sets
``triplets_only=False`` and uses a longer interaction cutoff for dihedral
quadruplets. ``GemNetOCCore`` adds the OC interaction hierarchy on the same
dense layers, bilinear, and geometry. Package defaults are GemNet-T.
GemNet-Q defaults are ``DEFAULT_BUILD_PARAMS_Q`` and ``DEFAULT_LAYER_PARAMS_Q``.
GemNet-OC defaults are ``DEFAULT_BUILD_PARAMS_OC`` and ``DEFAULT_LAYER_PARAMS_OC``.
"""



_SWISH = {
    "dim_feature": 1,
    "initial_alpha": 1,
    "initial_beta": 1,
    "learnable": False,
}

_READOUT = {
    "name": "HierachicalReadout",
    "params": {
        "output_fields": ["Ea"],
        "head_type": "dense",
        "use_bias": False,
        "initial_weight": "orthogonal",
        "keep_feature": False,
    },
}

_CORE_T = {
    "name": "Core",
    "params": {
        "num_blocks": 4,
        "num_spherical": 7,
        "emb_size_edge": 128,
        "emb_size_trip": 64,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_bil_trip": 64,
        "num_before_skip": 1,
        "num_after_skip": 1,
        "num_concat": 1,
        "num_atom": 2,
        "triplets_only": True,
        "envelope_exponent": 5,
        "output_mode": "feature",
    },
}

_CORE_Q = {
    "name": "Core",
    "params": {
        **_CORE_T["params"],
        "triplets_only": False,
        "emb_size_quad": 32,
        "emb_size_sbf": 32,
        "emb_size_bil_quad": 32,
        "emb_cutoff": 5.0,
        "int_cutoff": 10.0,
    },
}


def _stack(core: dict, basis_cutoff: float | None = None) -> list:
    bessel = {
        "name": "BesselRBF",
        "params": {
            "flavor": "gemnet",
            "trainable": True,
            "envelope_exponent": 5,
        },
    }
    if basis_cutoff is not None:
        bessel["params"]["basis_cutoff"] = basis_cutoff
    return [
        {"name": "RangeSeparation"},
        bessel,
        {"name": "RandomAtomEmbedding"},
        core,
        dict(_READOUT),
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 6,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
}

DEFAULT_LAYER_PARAMS = _stack(_CORE_T)

DEFAULT_BUILD_PARAMS_Q = {
    **DEFAULT_BUILD_PARAMS,
    "cutoff_sr": 10.0,
}

DEFAULT_LAYER_PARAMS_Q = _stack(_CORE_Q, basis_cutoff=5.0)

_READOUT_OC = {
    "name": "HierachicalReadout",
    "params": {
        "output_fields": ["Ea"],
        "head_type": "dense",
        "use_bias": False,
        "initial_weight": "orthogonal",
        "keep_feature": False,
    },
}

# OC20 base config (gemnet-oc.yml). Direct forces and the tag filter are not
# part of this stack; the Gaussian basis is the pre-core layer.
_CORE_OC = {
    "name": "Core",
    "params": {
        "num_blocks": 4,
        "num_spherical": 7,
        "emb_size_edge": 512,
        "emb_size_trip_in": 64,
        "emb_size_trip_out": 64,
        "emb_size_quad_in": 32,
        "emb_size_quad_out": 32,
        "emb_size_aint_in": 64,
        "emb_size_aint_out": 64,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_sbf": 32,
        "num_before_skip": 2,
        "num_after_skip": 2,
        "num_concat": 1,
        "num_atom": 3,
        "num_atom_emb_layers": 2,
        "num_output_afteratom": 3,
        "cutoff": 12.0,
        "cutoff_qint": 12.0,
        "cutoff_aeaint": 12.0,
        "cutoff_aint": 12.0,
        "max_neighbors": 30,
        "max_neighbors_qint": 8,
        "max_neighbors_aeaint": 20,
        "max_neighbors_aint": 1000,
        "quad_interaction": True,
        "atom_edge_interaction": True,
        "edge_atom_interaction": True,
        "atom_interaction": True,
        "envelope_exponent": 5,
        "output_mode": "feature",
    },
}


def _stack_oc() -> list:
    return [
        {"name": "RangeSeparation"},
        {
            "name": "GaussianRBF",
            "params": {
                "flavor": "gemnet-oc",
                "apply_cutoff_fn": False,
                "envelope_exponent": 5,
            },
        },
        {"name": "RandomAtomEmbedding"},
        _CORE_OC,
        dict(_READOUT_OC),
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


DEFAULT_BUILD_PARAMS_OC = {
    "dim_embedding": 256,
    "num_rbf": 128,
    "max_Za": 94,
    "cutoff_sr": 12.0,
    "Hartree_in_E": 1,
    "Bohr_in_R": 0.5291772108,
}

DEFAULT_LAYER_PARAMS_OC = _stack_oc()

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
    "DEFAULT_BUILD_PARAMS_Q",
    "DEFAULT_LAYER_PARAMS_Q",
    "DEFAULT_BUILD_PARAMS_OC",
    "DEFAULT_LAYER_PARAMS_OC",
]
