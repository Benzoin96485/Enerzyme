"""EquFlash Core (SamsungDS/GGNN).

E(3)-equivariant message passing built from Clebsch–Gordan tensor products.
The Core reads a scalar atom embedding and a radial basis and emits flat
irreps ``atom_feature``. Species embedding, the Bessel expansion, the XPLOR
window, and energy / force heads stay in the layer stack.

Reference: https://github.com/SamsungDS/GGNN (CC BY-NC-SA 4.0). This module
reimplements the interaction with Enerzyme's e3nn helpers. It does not vendor
that repository. The released checkpoints also offer cuequivariance and
FlashTP kernels; the numerics here are the e3nn uvu product.
"""

from __future__ import annotations





_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# Published EquFlash training config (GGNN ``configs/equflash/equflash.yml``):
# 256 channels, lmax 3, 10 convolutions, cutoff 6 Å with XPLOR from 5.5 Å,
# 8 Bessel functions. ``conv_denominator`` there is a dataset neighbor count
# (~72.5 on OMat). The default here is 1 so a new fit is not tied to that set.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "xplor",
    "cuton": 5.5,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "equflash", "trainable": True, "apply_cutoff_fn": True},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "lmax": 3,
            "num_layers": 10,
            "parity": True,
            "radial_hidden": [64, 64],
            "conv_denominator": 1.0,
            "train_denominator": False,
            "use_bias": False,
            "normalize_sph": True,
            "feed_forward": False,
            "act_radial": "silu",
            "act_scalar_even": "silu",
            "act_scalar_odd": "tanh",
            "act_gate_even": "silu",
            "act_gate_odd": "tanh",
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
