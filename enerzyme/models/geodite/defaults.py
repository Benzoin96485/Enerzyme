DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 192,
    "num_rbf": 12,
    "max_Za": 100,
    "cutoff_sr": 6.0,
    # eV, so the screened ZBL prefactor is 14.3996 eV·Å. Use 1.0 for Hartree.
    "Hartree_in_E": 27.211386245988,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"trainable": False, "apply_cutoff_fn": False},
    },
    {
        "name": "RadialWindow",
        "params": {
            "envelope": "polynomial",
            "envelope_p": 6,
            "attenuation": "covalent",
            "trainable_attenuation": True,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core"},
    {"name": "IsolatedAtomResidual"},
    {
        "name": "HierachicalReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
            "initial_weight": "xavier_uniform",
        },
    },
    {"name": "ScreenedZBLRepulsionEnergy", "params": {"envelope_p": 5}},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
