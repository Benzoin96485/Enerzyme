"""SO3LR default build / layer stack (So3krates Core + universal pairwise FF).

SO3LR (Kabylda et al., JACS 2025) is a So3krates variant: the same
``So3kratesCore`` plus charge/spin embeds and ZBL / erf-Coulomb / TS–QDO
dispersion priors. Use ``architecture: so3lr`` in YAML / Enerzymette.
"""

from __future__ import annotations

# Pretrained SO3LR (so3lr params / paper SI): r_max=4.5, L≤4, H=128, T=3,
# σ=4.0, γ=1.2, avg_num_neighbors≈13.17, phys cutoff, charge+spin embeds.
# Energies in eV → Hartree_in_E ≈ 27.211 (kehalf = 0.5 * Bohr * Hartree).
_SO3LR_AVG_NUM_NEIGHBORS = 13.168995780096482

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    # Embeddings / readouts: full periodic table. TS–QDO free-atom α/C6 only
    # cover Z≤102 (see enerzyme.models.layers.dispersion.ts_qdo.MAX_ZA_TSQDO).
    "max_Za": 118,
    "cutoff_sr": 4.5,
    "cutoff_fn": "phys",
    # Long-range neighbor cutoff for elec / disp (MD default 12 Å; gas ≈1000).
    "cutoff_lr": 12.0,
    "Bohr_in_R": 0.5291772105638411,
    "Hartree_in_E": 27.211386245988,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation", "params": {"cutoff_fn": "phys"}},
    {"name": "BernsteinRBF", "params": {"cutoff_fn": "phys"}},
    {"name": "RandomAtomEmbedding"},
    {"name": "ChargeSpinEmbedding", "params": {"attribute": "charge"}},
    {"name": "ChargeSpinEmbedding", "params": {"attribute": "spin"}},
    {"name": "GatherAtomEmbedding", "params": {"scale_by_sqrt_count": True}},
    {
        "name": "Core",
        "params": {
            "degrees": [1, 2, 3, 4],
            "num_features": 128,
            "num_heads": 4,
            "num_layers": 3,
            "message_normalization": "avg_num_neighbors",
            "avg_num_neighbors": _SO3LR_AVG_NUM_NEIGHBORS,
            "initialize_ev_to_zeros": True,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
            "residual_mlp_1": True,
            "residual_mlp_2": False,
            "cutoff_fn": "phys",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea", "Qa"],
            "head_type": "dense",
            "keep_feature": True,
        },
    },
    # SO3LR partial charges ≡ Linear(x) + Emb(Za); AtomicAffine only shifts Qa (scale=1).
    {
        "name": "AtomicAffine",
        "params": {
            "shifts": {"Qa": {"values": 0.0, "learnable": True}},
            "scales": {"Qa": {"values": 1.0, "learnable": False}},
        },
    },
    {"name": "ChargeConservation"},
    {"name": "HirshfeldReadout"},
    {"name": "ZBLRepulsionEnergy", "params": {"switch_off": 1.5}},
    {
        "name": "ElectrostaticEnergy",
        "params": {
            "flavor": "SO3LR",
            "electrostatic_energy_scale": 4.0,
            "neighborlist_format_lr": "sparse",
        },
    },
    {
        "name": "TSQDODispersionEnergy",
        "params": {
            "dispersion_energy_scale": 1.2,
            "cutoff_lr_damping": 2.0,
            "neighborlist_format_lr": "sparse",
        },
    },
    {"name": "AtomicCharge2Dipole"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
