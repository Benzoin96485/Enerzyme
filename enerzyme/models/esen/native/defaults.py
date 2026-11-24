"""Native eSEN Core (Fu et al., 2025, arXiv:2502.12147).

Equivariant Smooth Energy Network. The Core maps an external scalar atom
embedding and radial basis to spherical latents. Nodewise layers stay in
the spherical-harmonic basis (SiLU gate), and a polynomial envelope makes
edge messages vanish smoothly at the cutoff. Forces stay outside the Core
(``EnergyReduce`` + ``Force``), so they are energy-conserving.

Distinct from ``uma_qs`` / ``UMAWrapperQS`` in the parent package, which
wrap a fairchem UMA checkpoint. This module does not import fairchem and
does not load those checkpoints.

Reuses :class:`~enerzyme.models.mp.equivariant.so2.ops.SO2_Convolution`,
:class:`~enerzyme.models.mp.equivariant.sph.linear.SO3Linear`,
:class:`~enerzyme.models.mp.equivariant.s2.activation.GateActivation`,
:func:`~enerzyme.models.mp.equivariant.sph.layer_norm.get_normalization_layer`,
:class:`~enerzyme.models.mp.nn.radial_mlp.RadialFunction`, and
:func:`~enerzyme.models.mp.basis.envelope.polynomial_envelope`.
"""

from __future__ import annotations




# fairchem eSEN divides aggregated messages by 5 (about sqrt of a typical
# neighbor count). It is a constant, not a dataset statistic.
_RESCALE = 5.0

# Released widths. The checked-in stack stays small; swap these onto the Core
# when reproducing a published model. Radial basis and cutoff stay in
# build_params / the RBF layer (Gaussian, 10 functions, 6 Å on MPTrj;
# 64 Gaussians for the 30M OAM model; Bessel, 10 functions, 5 Å on SPICE).
PUBLISHED_MP_3M_CORE_PARAMS = {
    "sphere_channels": 128,
    "hidden_channels": 128,
    "edge_channels": 128,
    "lmax": 2,
    "mmax": 2,
    "num_layers": 2,
    "norm_type": "rms_norm_sh",
    "act_type": "gate",
    "mlp_type": "spectral",
    "use_envelope": True,
    "rescale_factor": _RESCALE,
}
PUBLISHED_SPICE_6M_CORE_PARAMS = {
    **PUBLISHED_MP_3M_CORE_PARAMS,
    "num_layers": 4,
}
PUBLISHED_30M_CORE_PARAMS = {
    **PUBLISHED_MP_3M_CORE_PARAMS,
    "lmax": 3,
    "mmax": 2,
    "num_layers": 10,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 16,
    "num_rbf": 8,
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": None,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "GaussianRBF",
        "params": {"flavor": "esen", "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "sphere_channels": 16,
            "hidden_channels": 16,
            "edge_channels": 16,
            "lmax": 2,
            "mmax": 2,
            "num_layers": 2,
            "norm_type": "rms_norm_sh",
            "act_type": "gate",
            "mlp_type": "spectral",
            "use_envelope": True,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "num_hidden_layers": 2,
            "use_bias_out": True,
            "activation_fn": "swish",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
