"""Fixed DC-UMA C-PCM candidate-pool constants.

Filter thresholds, label-matrix sizes, and the dielectric grids follow the PoC
draft. Sampling is deterministic given ``SEED``. This stage keeps closed-shell
singlets only (OMol ``s_squared`` ≤ ``CLOSED_SHELL_S2_MAX``).
"""

from __future__ import annotations

from typing import Mapping

SCHEMA_VERSION = "dc_uma_cpcm_manifest_v1"
SEED = 20260817

# Reports may only use this phrase. Structures may already have been seen in
# UMA pretraining; they are not an independent chemistry hold-out.
UNSEEN_CLAIM = "adapter-unseen parent"
UNSEEN_CLAIM_NOTE = (
    "Validation/test parents are held out from adapter training only. "
    "They may have been present in UMA-s-1p2 / OMol25 pretraining and must not "
    "be described as independent unseen chemistry."
)

ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl"})
ALLOWED_ATOMIC_NUMBERS = frozenset({1, 6, 7, 8, 9, 15, 16, 17})
MIN_ATOMS = 6
MAX_ATOMS = 60
ALLOWED_CHARGES = (-1, 0, 1)
SINGLET_SPIN = 1
# This stage is closed-shell only. OMol UKS rows that collapsed to S²≈0 are kept;
# broken-symmetry / open-shell singlets (S² above this cut) are dropped everywhere:
# candidate index, diagnostic jobs, and production labels.
CLOSED_SHELL_S2_MAX = 0.05
MAX_N_BASIS = 800
MAX_FORCE_EV_A = 5.0
CONNECT_SCALE = 1.3
OVERLAP_SCALE = 0.5
WL_ITERATIONS = 4
MAX_CONFS_FOR_RMSD = 24
NEAR_DUP_RMSD_A = 0.15
RESERVE_FRACTION = 0.20

# Charge mix among unique parents: 50% / 25% / 25%.
PARENT_CHARGE_QUOTA: Mapping[int, int] = {0: 92, 1: 46, -1: 46}
N_TRAIN_PARENTS = 112
N_VAL_PARENTS = 36
N_TEST_PARENTS = 36
N_UNIQUE_PARENTS = N_TRAIN_PARENTS + N_VAL_PARENTS + N_TEST_PARENTS  # 184
CONFS_PER_PARENT = 2
N_TRAIN_GEOMS = N_TRAIN_PARENTS * CONFS_PER_PARENT  # 224
N_VAL_GEOMS = N_VAL_PARENTS * CONFS_PER_PARENT  # 72
N_TEST_GEOMS = N_TEST_PARENTS * CONFS_PER_PARENT  # 72
N_PROBE_GEOMS = 32  # 16 train parents × 2 conformers
N_PROBE_PARENTS = N_PROBE_GEOMS // CONFS_PER_PARENT

TRAIN_EPSILONS = (1.5, 4.0, 16.0, 80.0)
PROBE_EPSILONS = (2.0, 8.0, 32.0)
EVAL_EPSILONS = (1.5, 2.0, 4.0, 8.0, 16.0, 32.0, 80.0)

N_TRAIN_LABELS = N_TRAIN_GEOMS * len(TRAIN_EPSILONS)  # 896
N_PROBE_LABELS = N_PROBE_GEOMS * len(PROBE_EPSILONS)  # 96
N_VAL_LABELS = N_VAL_GEOMS * len(EVAL_EPSILONS)  # 504
N_TEST_LABELS = N_TEST_GEOMS * len(EVAL_EPSILONS)  # 504
N_TOTAL_LABELS = N_TRAIN_LABELS + N_PROBE_LABELS + N_VAL_LABELS + N_TEST_LABELS  # 2000

# 16 train parents × 2 geoms × 4 ε = 128; 8 val parents × 2 geoms × 7 ε = 112.
N_DIAG_TRAIN_PARENTS = 16
N_DIAG_VAL_PARENTS = 8
N_DIAG_TRAIN_LABELS = N_DIAG_TRAIN_PARENTS * CONFS_PER_PARENT * len(TRAIN_EPSILONS)
N_DIAG_VAL_LABELS = N_DIAG_VAL_PARENTS * CONFS_PER_PARENT * len(EVAL_EPSILONS)
N_DIAG_LABELS = N_DIAG_TRAIN_LABELS + N_DIAG_VAL_LABELS  # 240
N_DIAG_JOBS = N_DIAG_LABELS
N_PILOT_GEOMS = 6
PILOT_EPSILONS = ("vacuum", 1.5, 7.3, 80.0)
N_PILOT_JOBS = N_PILOT_GEOMS * len(PILOT_EPSILONS)  # 24

# Three baselines × three seeds. Sub-evaluations: train ε, held-out ε, adapter-unseen parent.
DIAGNOSTIC_BASELINES = ("zero", "x_only", "stage_a")
DIAGNOSTIC_SEEDS = (SEED, SEED + 1, SEED + 2)
DIAGNOSTIC_EVAL_SLICES = ("train_epsilon", "held_out_epsilon", "held_out_parent")
DIAGNOSTIC_LR = 1.0e-3
DIAGNOSTIC_HUBER_DELTA = 0.01
DIAGNOSTIC_MAX_EPOCHS = 200
DIAGNOSTIC_PATIENCE = 20
DIAGNOSTIC_MICROBATCH = 1
DIAGNOSTIC_WEIGHT_DECAY = 0.0
# Aspirational residual MAE, not a hard gate before the label scale is measured.
ASPIRATIONAL_ENERGY_MAE_EV = 0.10
ASPIRATIONAL_FORCE_MAE_EV_A = 0.10
# Stage B 1-seed keep/drop vs Stage A on the same test split.
STAGE_B_CHARGED_ENERGY_IMPROVE = 0.10
STAGE_B_FORCE_DEGRADE_MAX = 0.05
LOWDIN_AUX_WEIGHT = 0.05

SIZE_BINS = ("6-15", "16-30", "31-60")
CHARGE_SPLIT = {
    0: {"train": 56, "validation": 18, "test": 18},
    1: {"train": 28, "validation": 9, "test": 9},
    -1: {"train": 28, "validation": 9, "test": 9},
}
CHARGE_RESERVE = {0: 18, 1: 9, -1: 9}
DIAG_TRAIN_BY_CHARGE = {0: 8, 1: 4, -1: 4}
DIAG_VAL_BY_CHARGE = {0: 4, 1: 2, -1: 2}

DEFAULT_TRAIN_4M = "/global/homes/w/weiliang/m4787_cfs/data/omol25/train_4M"
DEFAULT_EXPERIMENTS = "/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm"
KNOWN_DATA_IDS = (
    "ani2x",
    "biomolecules",
    "elytes",
    "geom_orca6",
    "metal_complexes",
    "orbnet_denali",
    "reactivity",
    "rgd",
    "spice",
    "trans1x",
)

# def2-TZVPD contracted function counts from fairchem.data.omol.orca.calc.BASIS_DICT.
DEF2_TZVPD_N_BASIS = {
    "H": 9,
    "He": 9,
    "Li": 17,
    "Be": 22,
    "B": 37,
    "C": 37,
    "N": 37,
    "O": 40,
    "F": 40,
    "Ne": 40,
    "Na": 35,
    "Mg": 35,
    "Al": 43,
    "Si": 43,
    "P": 43,
    "S": 46,
    "Cl": 46,
    "Ar": 46,
    "K": 36,
    "Ca": 36,
    "Sc": 48,
    "Ti": 48,
    "V": 48,
    "Cr": 48,
    "Mn": 48,
    "Fe": 48,
    "Co": 48,
    "Ni": 48,
    "Cu": 48,
    "Zn": 51,
    "Ga": 54,
    "Ge": 54,
    "As": 54,
    "Se": 57,
    "Br": 57,
    "Kr": 57,
    "Rb": 33,
    "Sr": 33,
    "Y": 40,
    "Zr": 40,
    "Nb": 40,
    "Mo": 40,
    "Tc": 40,
    "Ru": 40,
    "Rh": 40,
    "Pd": 40,
    "Ag": 40,
    "Cd": 40,
    "In": 56,
    "Sn": 56,
    "Sb": 56,
    "Te": 59,
    "I": 59,
    "Xe": 59,
    "Cs": 32,
    "Ba": 40,
    "La": 43,
    "Ce": 105,
    "Pr": 105,
    "Nd": 98,
    "Pm": 98,
    "Sm": 98,
    "Eu": 93,
    "Gd": 98,
    "Tb": 98,
    "Dy": 98,
    "Ho": 98,
    "Er": 101,
    "Tm": 101,
    "Yb": 96,
    "Lu": 96,
    "Hf": 43,
    "Ta": 43,
    "W": 43,
    "Re": 43,
    "Os": 43,
    "Ir": 43,
    "Pt": 43,
    "Au": 43,
    "Hg": 46,
    "Tl": 56,
    "Pb": 56,
    "Bi": 56,
    "Po": 59,
    "At": 59,
    "Rn": 59,
}
