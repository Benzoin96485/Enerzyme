"""Shared SO(3) / SO(2) primitives for equivariant GNNs (eSCN, EquiformerV2, So3krates, …).

These modules are adapted from fairchem v1 eSCN (Passaro & Zitnick, *Reducing SO(3)
Convolutions to SO(2)*, 2023) with EquiformerV2 extensions (``get_rotate_inv_rescale``,
grid kwargs, ``SO3_LinearV2``). So3krates additionally uses closed-form
``RealSphericalHarmonics`` / ``spherical_harmonics`` (tesseral layouts) and
``L0Contraction`` (mlff / So3krates-torch). SpookyNet uses ``layout='spookynet_d'``.
EquiformerV2 SO(2) convolution lives in ``so2_ops`` alongside the eSCN-style
``SO2Block`` / ``SO2Conv`` (different parameterization — not interchangeable).

They are **not** used by the Meta UMA wrappers under ``enerzyme.models.esen``, which
keep the fairchem ``eSCNMD*`` checkpoint path.
"""

from .activation import GateActivation, S2Activation, SeparableS2Activation
from .coefficient_mapping import CoefficientMapping
from .drop import EquivariantDropoutArraySphericalHarmonics
from .embedding import SO3_Embedding
from .grid import SO3_Grid
from .l0_contraction import L0Contraction, load_cgmatrix
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_l_to_all_m_expand_index,
    get_normalization_layer,
)
from .linear import SO3_LinearV2
from .rotation import SO3_Rotation, init_edge_rot_mat
from .so2_conv import SO2Block, SO2Conv
from .so2_ops import SO2_Convolution, SO2_m_Convolution
from .spherical_harmonics import RealSphericalHarmonics, spherical_harmonics

# SphereSampleReadout subclasses layers.BaseFFLayer. Keep it lazy so importing
# ``enerzyme.models.so3`` does not pull ``layers`` (which re-exports the readout
# for YAML discovery) and create a circular import.

__all__ = [
    "CoefficientMapping",
    "EquivariantDropoutArraySphericalHarmonics",
    "EquivariantLayerNormArray",
    "EquivariantLayerNormArraySphericalHarmonics",
    "EquivariantRMSNormArraySphericalHarmonicsV2",
    "GateActivation",
    "L0Contraction",
    "RealSphericalHarmonics",
    "S2Activation",
    "SO2_Convolution",
    "SO2_m_Convolution",
    "SO2Block",
    "SO2Conv",
    "SO3_Embedding",
    "SO3_Grid",
    "SO3_LinearV2",
    "SO3_Rotation",
    "SeparableS2Activation",
    "SphereSampleReadout",
    "calc_sphere_points",
    "get_l_to_all_m_expand_index",
    "get_normalization_layer",
    "init_edge_rot_mat",
    "load_cgmatrix",
    "spherical_harmonics",
]


def __getattr__(name: str):
    if name in {"SphereSampleReadout", "calc_sphere_points"}:
        from .sphere_sample_readout import SphereSampleReadout, calc_sphere_points

        mapping = {
            "SphereSampleReadout": SphereSampleReadout,
            "calc_sphere_points": calc_sphere_points,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
