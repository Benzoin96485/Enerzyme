"""SpinConv rotation-invariant message passing.

Shuaibi et al., arXiv:2106.09575. The update follows the grid convolution
in `facebookresearch/fairchem <https://github.com/facebookresearch/fairchem/blob/ocp2.0/ocpmodels/models/spinconv.py>`_
(branch ``ocp2.0``). That module is not imported, and its checkpoints do
not load.

The Core reads ``GaussianRBF`` with ``flavor: spinconv`` and
``RandomAtomEmbedding``. The latitude–longitude projection, the circular
convolution, and the species mixture stay inside the Core. The energy head
is a linear ``SimpleReadout`` on the mixed node state.
"""



_READOUT = {
    "name": "SimpleReadout",
    "params": {
        "output_fields": ["Ea"],
        "head_type": "dense",
        "use_bias": True,
    },
}

# OC20 S2EF config (configs/s2ef/all/spinconv/spinconv_force.yml): messages
# of width 32, three updates, hidden width 256, 512 Gaussians with σ = 3Δ,
# a 16×12 grid, 40 neighbors, and a 6 Å cutoff. The energy-centric paper
# setting uses a 12×8 grid. The force-centric direct head is not in this stack.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 512,
    "hidden_channels": 32,
    "mid_hidden_channels": 256,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "basis_width_scalar": 3.0,
    # OC20 reports eV and Å. One Hartree is 27.211 eV.
    "Hartree_in_E": 27.211386245988,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "GaussianRBF",
        "params": {
            "flavor": "spinconv",
            "apply_cutoff_fn": False,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 3,
            "sphere_size_lat": 16,
            "sphere_size_long": 12,
            "max_num_neighbors": 40,
            "num_embedding_basis": 8,
            "group_channels": 8,
            "scale_distances": True,
            "distance_scale_max": 2.0,
            "residual_first": False,
        },
    },
    dict(_READOUT),
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
