"""EquFlashV2 Core.

Channelwise Clebsch-Gordan message passing with equivariant RMSNorm and a
bilinear (or gated) node update. Species embedding, the Bessel / XPLOR
radial window, species-wise energy calibration, and force heads stay outside
the Core.

The block layout follows Samsung Electronics' EquFlashV2
(https://github.com/SamsungDS/GGNN, CC BY-NC-SA 4.0). This module is an
Enerzyme reimplementation on e3nn. It does not depend on cuequivariance or
FlashTP. Published OMat / OAM checkpoints are not loaded here.

``conv_type`` ``full`` and ``efficient`` share one channelwise ``uvu``
convolution. Selecting ``efficient`` warns, because this port has no fused
kernel. The cuequivariance kernels differ in layout and fusion, not in the
Enerzyme data flow.
"""

from __future__ import annotations





DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cuton": 5.5,
    "cutoff_fn": "xplor",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "equflash",
            "trainable": False,
            "apply_cutoff_fn": True,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_layers": 2,
            "lmax": 2,
            "is_parity": True,
            "interaction_type": "equflash",
            "conv_type": "full",
            "sc_type": "identity",
            "nonlinear_type": "bilinear_gate",
            "nonlinear_scale": 1.0,
            "after_nonlinear": "linear",
            "feedforward_type": "feedforward",
            "ff_nonlinear_type": "bilinear_gate",
            "ff_nonlinear_scale": 1.0,
            "ff_sc_type": "identity",
            "radial_hidden": [32, 32],
            "denominator": 1.0,
            "train_denominator": False,
            "normalize_sph": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activate_hidden": False,
            "use_bias": False,
            "keep_feature": False,
        },
    },
    {
        "name": "AtomicAffine",
        "params": {
            "order": "scale_then_shift",
            "scales": {"Ea": {"values": 1.0, "learnable": False}},
            "shifts": {"Ea": {"values": 0.0, "learnable": True}},
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
