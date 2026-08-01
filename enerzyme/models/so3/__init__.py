"""Shared SO(3) / SO(2) primitives for equivariant GNNs (eSCN, EquiformerV2, EquiformerV3, So3krates, DPA4, …).

These modules are adapted from fairchem v1 eSCN (Passaro & Zitnick, *Reducing SO(3)
Convolutions to SO(2)*, 2023) with EquiformerV2 extensions (``get_rotate_inv_rescale``,
grid kwargs, ``SO3_LinearV2``). So3krates additionally uses closed-form
``RealSphericalHarmonics`` / ``spherical_harmonics`` (tesseral layouts) and
``L0Contraction`` (mlff / So3krates-torch). SpookyNet uses ``layout='spookynet_d'``.
EquiformerV2 SO(2) convolution lives in ``so2_ops`` alongside the eSCN-style
``SO2Block`` / ``SO2Conv`` (different parameterization — not interchangeable).
DPA4 / EMFA contributes packed/m-major indexing, focus-stream gated activations,
``SO3FocusLinear``, ``FocusSO2Linear``, degree-balanced ``EquivariantDegreeRMSNorm``,
``BesselC3RadialBasis``, quaternion edge frames (:mod:`wigner_quaternion`) that
share the e3nn/``Jd`` Wigner-D backend (:mod:`wigner_jd`), flat lat–long
:class:`SO3Grid` (eSCN / EquiformerV2 / V3) plus Lebedev
:class:`S2LebedevProjector` behind :class:`S2GridProjector`, and shared Lebedev
tables (also used by EFA for points).

They are **not** used by the Meta UMA wrappers under ``enerzyme.models.esen``, which
keep the fairchem ``eSCNMD*`` checkpoint path.
"""

from .activation import GateActivation, S2Activation, SeparableS2Activation

from .activation_v3 import (
    get_activation,
    check_activation_name,
    has_scalars,
    add_dropout,
    prepare_activation_forward_param,
    LinearSwiGLU,
    LinearSquare,
    SwiGLU,
    SeparableGateS2Activation_SwiGLU_Merge,
)
from .envelope import C3CutoffEnvelope, PolynomialEnvelope
from .gated import FocusLinear, SO3GatedActivation
from .indexing import (
    build_gie_zonal_index,
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim,
    get_so3_dim_of_lmax,
    m_major_reduced_dim,
    map_degree_idx,
    project_D_to_m,
    project_Dt_from_m,
    so3_packed_index,
)
from .lebedev import (
    LEBEDEV_FREQUENCY_LOOKUP,
    LEBEDEV_PRECISION_TO_NPOINTS,
    S2LebedevProjector,
    available_lebedev_nums,
    available_lebedev_precisions,
    lebedev_quadrature,
    lebedev_tensors,
    load_lebedev_rule,
    recommend_max_frequency,
    resolve_lebedev_precision,
)
from .softmax import GraphSoftmax, SoftCap, segment_envelope_gated_softmax
from .so2_focus import FocusSO2Linear
from .so2_ops import SO2Linear, SO2MLinear
from .linear import SO3FocusLinear, SO3Linear
from .layer_norm import (
    EquivariantDegreeRMSNorm,
    EquivariantMergeLayerNorm,
    EquivariantSeparableLayerNorm,
    RMSNorm,
)
from .radial import BesselC3RadialBasis, RadialMLP
from .rotation_fused import SO3RotationFused, CoefficientMappingModule
from .coefficient_mapping import CoefficientMapping
from .drop import EquivariantDropoutArraySphericalHarmonics, EquivariantDegreeDropout
from .embedding import SO3_Embedding
from .grid import SO3Grid, SO3_Grid, SO3GridResolved, build_so3_grid_table
from .s2_projector import S2GridProjector
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
from .wigner_jd import (
    max_wigner_lmax,
    rotation_matrix_to_euler,
    wigner_D,
    wigner_from_rotation_matrix,
)
from .wigner_quaternion import (
    WignerDCalculator,
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_rotation_matrix,
    quaternion_z_rotation,
    safe_norm,
)

# SphereSampleReadout subclasses layers.BaseFFLayer. Keep it lazy so importing
# ``enerzyme.models.so3`` does not pull ``layers`` (which re-exports the readout
# for YAML discovery) and create a circular import.

__all__ = [
    "CoefficientMapping",
    "EquivariantDropoutArraySphericalHarmonics",
    "EquivariantDegreeDropout",
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
    "SO3Grid",
    "SO3GridResolved",
    "S2GridProjector",
    "build_so3_grid_table",
    "SO3_LinearV2",
    "SO3_Rotation",
    "SeparableS2Activation",
    "SphereSampleReadout",
    "calc_sphere_points",
    "get_l_to_all_m_expand_index",
    "get_normalization_layer",
    "init_edge_rot_mat",
    "max_wigner_lmax",
    "rotation_matrix_to_euler",
    "wigner_D",
    "wigner_from_rotation_matrix",
    "load_cgmatrix",
    "spherical_harmonics",
    "PolynomialEnvelope",
    "C3CutoffEnvelope",
    "LEBEDEV_PRECISION_TO_NPOINTS",
    "LEBEDEV_FREQUENCY_LOOKUP",
    "S2LebedevProjector",
    "available_lebedev_nums",
    "available_lebedev_precisions",
    "lebedev_quadrature",
    "lebedev_tensors",
    "load_lebedev_rule",
    "recommend_max_frequency",
    "resolve_lebedev_precision",
    "GraphSoftmax",
    "SoftCap",
    "segment_envelope_gated_softmax",
    "SO2Linear",
    "SO2MLinear",
    "FocusSO2Linear",
    "SO3Linear",
    "SO3FocusLinear",
    "EquivariantDegreeRMSNorm",
    "EquivariantMergeLayerNorm",
    "EquivariantSeparableLayerNorm",
    "RMSNorm",
    "FocusLinear",
    "SO3GatedActivation",
    "BesselC3RadialBasis",
    "RadialMLP",
    "build_gie_zonal_index",
    "build_m_major_index",
    "build_m_major_l_index",
    "build_rotate_inv_rescale",
    "get_so3_dim",
    "get_so3_dim_of_lmax",
    "m_major_reduced_dim",
    "map_degree_idx",
    "project_D_to_m",
    "project_Dt_from_m",
    "so3_packed_index",
    "WignerDCalculator",
    "build_edge_quaternion",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_to_rotation_matrix",
    "quaternion_z_rotation",
    "safe_norm",
    "SO3RotationFused",
    "CoefficientMappingModule",
    "get_activation",
    "check_activation_name",
    "has_scalars",
    "add_dropout",
    "prepare_activation_forward_param",
    "LinearSwiGLU",
    "SwiGLU",
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
