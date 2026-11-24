"""Shared SO(3) / SO(2) primitives for equivariant GNNs (eSCN, EquiformerV2, EquiformerV3, So3krates, DPA4, TECE, …).

These modules are adapted from fairchem v1 eSCN (Passaro & Zitnick, *Reducing SO(3)
Convolutions to SO(2)*, 2023) with EquiformerV2 extensions (``get_rotate_inv_rescale``,
grid kwargs, ``SO3_LinearV2``). So3krates additionally uses closed-form
``RealSphericalHarmonics`` / ``spherical_harmonics`` (tesseral layouts) and
``L0Contraction`` (mlff / So3krates-torch). SpookyNet uses ``layout='spookynet_d'``.
EquiformerV2 SO(2) convolution lives in ``so2.ops`` alongside the eSCN-style
``SO2Block`` / ``SO2Conv`` (different parameterization — not interchangeable).
DPA4 / EMFA contributes packed/m-major indexing, focus-stream gated activations,
``SO3FocusLinear``, ``FocusSO2Linear``, degree-balanced ``EquivariantDegreeRMSNorm``,
``BesselC3RadialBasis``, quaternion edge frames (:mod:`rotation.wigner_quaternion`) that
share the e3nn/``Jd`` Wigner-D backend (:mod:`rotation.wigner_jd`), flat lat–long
:class:`SO3Grid` (eSCN / EquiformerV2 / V3) plus Lebedev
:class:`S2LebedevProjector` behind :class:`S2GridProjector`, and shared Lebedev
tables (also used by EFA for points).

TECE (Xu et al., arXiv:2607.10664) adds recursive/direct :class:`WignerD`,
:class:`uvSO2Linear`, :class:`SO2Gate`, :class:`ComplexProductBasis`, and
:class:`LayoutTransform` for edge-frame ECE + Radial Rotary Attention.

Native eSEN (``architecture: esen``) uses these SO(2) convolutions, the SiLU
gate, ``SO3Linear``, and the equivariant norms. The Meta UMA wrappers in the
same ``esen`` package (``uma_qs``) do not: they keep the fairchem checkpoint path.

Names resolve lazily so that importing one submodule (for example
``mp.equivariant.e3nn``) does not import every backend.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "GateActivation": ".s2.activation",
    "S2Activation": ".s2.activation",
    "SeparableS2Activation": ".s2.activation",
    "get_activation": ".s2.activation_v3",
    "check_activation_name": ".s2.activation_v3",
    "has_scalars": ".s2.activation_v3",
    "add_dropout": ".s2.activation_v3",
    "prepare_activation_forward_param": ".s2.activation_v3",
    "LinearSwiGLU": ".s2.activation_v3",
    "LinearSquare": ".s2.activation_v3",
    "SwiGLU": ".s2.activation_v3",
    "SeparableGateS2Activation_SwiGLU_Merge": ".s2.activation_v3",
    "C3CutoffEnvelope": "..basis.envelope",
    "DimeNetEnvelope": "..basis.envelope",
    "PolynomialEnvelope": "..basis.envelope",
    "FocusLinear": ".sph.gated",
    "SO3GatedActivation": ".sph.gated",
    "build_gie_zonal_index": ".indexing",
    "build_m_major_index": ".indexing",
    "build_m_major_l_index": ".indexing",
    "build_rotate_inv_rescale": ".indexing",
    "get_so3_dim": ".indexing",
    "get_so3_dim_of_lmax": ".indexing",
    "m_major_reduced_dim": ".indexing",
    "map_degree_idx": ".indexing",
    "project_D_to_m": ".indexing",
    "project_Dt_from_m": ".indexing",
    "so3_packed_index": ".indexing",
    "LEBEDEV_FREQUENCY_LOOKUP": ".s2.lebedev",
    "LEBEDEV_PRECISION_TO_NPOINTS": ".s2.lebedev",
    "S2LebedevProjector": ".s2.lebedev",
    "available_lebedev_nums": ".s2.lebedev",
    "available_lebedev_precisions": ".s2.lebedev",
    "lebedev_quadrature": ".s2.lebedev",
    "lebedev_tensors": ".s2.lebedev",
    "load_lebedev_rule": ".s2.lebedev",
    "recommend_max_frequency": ".s2.lebedev",
    "resolve_lebedev_precision": ".s2.lebedev",
    "GraphSoftmax": "..ops.softmax",
    "SoftCap": "..ops.softmax",
    "segment_envelope_gated_softmax": "..ops.softmax",
    "FocusSO2Linear": ".so2.focus",
    "SO2Linear": ".so2.ops",
    "SO2MLinear": ".so2.ops",
    "uvSO2Linear": ".so2.ops",
    "uvSO2MLinear": ".so2.ops",
    "SO2_Convolution": ".so2.ops",
    "SO2_m_Convolution": ".so2.ops",
    "SO2Gate": ".so2.gate",
    "so2_expand_index": ".so2.utils",
    "so3_expand_index": ".so2.utils",
    "num_so2_components": ".so2.utils",
    "ComplexProductBasis": ".so2.product",
    "UUUSo2TensorProduct": ".so2.product",
    "LayoutTransform": ".layout",
    "WignerD": ".rotation.wigner_cg",
    "init_edge_rot_mat_quaternion": ".rotation.wigner_cg",
    "SO3FocusLinear": ".sph.linear",
    "SO3Linear": ".sph.linear",
    "SO3_LinearV2": ".sph.linear",
    "EquivariantDegreeRMSNorm": ".sph.layer_norm",
    "EquivariantMergeLayerNorm": ".sph.layer_norm",
    "EquivariantSeparableLayerNorm": ".sph.layer_norm",
    "RMSNorm": ".sph.layer_norm",
    "EquivariantLayerNormArray": ".sph.layer_norm",
    "EquivariantLayerNormArraySphericalHarmonics": ".sph.layer_norm",
    "EquivariantRMSNormArraySphericalHarmonicsV2": ".sph.layer_norm",
    "get_l_to_all_m_expand_index": ".sph.layer_norm",
    "get_normalization_layer": ".sph.layer_norm",
    "BesselC3RadialBasis": "..basis.radial",
    "RadialMLP": "..nn.radial_mlp",
    "SO3RotationFused": ".rotation.fused",
    "CoefficientMappingModule": ".rotation.fused",
    "CoefficientMapping": ".so2.coefficient_mapping",
    "EquivariantDropoutArraySphericalHarmonics": ".sph.drop",
    "EquivariantDegreeDropout": ".sph.drop",
    "SO3_Embedding": ".sph.embedding",
    "SO3Grid": ".s2.grid",
    "SO3_Grid": ".s2.grid",
    "SO3GridResolved": ".s2.grid",
    "build_so3_grid_table": ".s2.grid",
    "S2GridProjector": ".s2.projector",
    "L0Contraction": ".sph.l0_contraction",
    "load_cgmatrix": ".sph.l0_contraction",
    "edge_rotation": ".rotation",
    "SO3_Rotation": ".rotation.so3_rotation",
    "init_edge_rot_mat": ".rotation.so3_rotation",
    "SO2Block": ".so2.conv",
    "SO2Conv": ".so2.conv",
    "RealSphericalHarmonics": "..basis.angular",
    "spherical_harmonics": "..basis.angular",
    "max_wigner_lmax": ".rotation.wigner_jd",
    "rotation_matrix_to_euler": ".rotation.wigner_jd",
    "wigner_D": ".rotation.wigner_jd",
    "wigner_from_rotation_matrix": ".rotation.wigner_jd",
    "WignerDCalculator": ".rotation.wigner_quaternion",
    "build_edge_quaternion": ".rotation.wigner_quaternion",
    "quaternion_multiply": ".rotation.wigner_quaternion",
    "quaternion_normalize": ".rotation.wigner_quaternion",
    "quaternion_to_rotation_matrix": ".rotation.wigner_quaternion",
    "quaternion_z_rotation": ".rotation.wigner_quaternion",
    "safe_norm": ".rotation.wigner_quaternion",
    "SphereSampleReadout": ".s2.sample_readout",
    "calc_sphere_points": ".s2.sample_readout",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
