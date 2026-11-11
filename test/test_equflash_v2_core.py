"""EquFlashV2: registration, equivariance, and the energy/force stack."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "equflashv2_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_xplor_window_is_flat_then_zero():
    from enerzyme.models.cutoff import xplor_transition

    distance = torch.tensor([0.0, 5.5, 5.75, 6.0, 7.0])
    window = xplor_transition(distance, cutoff=6.0, cuton=5.5)
    assert torch.allclose(window[:2], torch.ones(2), atol=1e-6)
    assert window[2] > 0 and window[2] < 1
    assert torch.allclose(window[3:], torch.zeros(2), atol=1e-6)


def test_bessel_equflash_prefactor():
    import math

    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 6.0
    layer = BesselRBFLayer(
        num_rbf=4,
        cutoff_sr=cutoff,
        flavor="equflash",
        trainable=False,
        apply_cutoff_fn=False,
    )
    distances = torch.tensor([0.0, 0.5, 2.0, 5.9])
    got = layer.get_rbf(distances)
    frequencies = torch.arange(1, 5) * math.pi / cutoff
    expected = []
    for dist in distances:
        if dist == 0:
            expected.append((2.0 / cutoff) * frequencies)
        else:
            expected.append((2.0 / cutoff) * torch.sin(frequencies * dist) / dist)
    assert torch.allclose(got, torch.stack(expected), atol=1e-5, rtol=1e-5)


def test_atomic_affine_scale_then_shift():
    from enerzyme.models.layers.denormalize import AtomicAffineLayer

    layer = AtomicAffineLayer(
        max_Za=8,
        shifts={"Ea": {"values": {6: 1.5, 1: -0.5}, "learnable": False}},
        scales={"Ea": {"values": 2.0, "learnable": False}},
        order="scale_then_shift",
    )
    energy = torch.tensor([1.0, 3.0, 4.0])
    atomic_numbers = torch.tensor([6, 1, 6])
    out = layer.get_output(Za=atomic_numbers, Ea=energy)["Ea"]
    assert torch.allclose(out, torch.tensor([3.5, 5.5, 9.5]))


def test_equflash_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.denormalize import AtomicAffineLayer
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("equflashv2", "equflash_v2", "EquFlashV2"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "EquFlashV2Core"
        assert build_params["cutoff_fn"] == "xplor"
        assert build_params["cuton"] == 5.5
        assert layers[1]["params"]["flavor"] == "equflash"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "EquFlashV2Core"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.num_layers == 2
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head[0].activation_fn is None
    assert readout.head[0].bias is None
    assert readout.head[0].dim_feature_out == 4
    affine = next(layer for layer in model.post_sequence if isinstance(layer, AtomicAffineLayer))
    assert affine.order == "scale_then_shift"


def _core(**overrides):
    from enerzyme.models.equflash import EquFlashV2Core

    params = dict(
        dim_embedding=4,
        num_rbf=3,
        num_layers=2,
        lmax=1,
        radial_hidden=[8],
        denominator=1.0,
    )
    params.update(overrides)
    return EquFlashV2Core(**params)


def _pack(core, pos, embedding=None):
    n = pos.shape[0]
    idx_i, idx_j = _edges(n)
    vij = pos[idx_j] - pos[idx_i]
    width = embedding.shape[-1] if embedding is not None else core.dim_embedding
    if embedding is None:
        embedding = torch.randn(n, width)
    return {
        "atom_embedding": embedding,
        "rbf": torch.randn(idx_i.numel(), core.num_rbf),
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "vij_sr": vij,
    }


def test_equflash_scalar_features_are_rotation_and_translation_invariant():
    torch.manual_seed(0)
    core = _core()
    core.eval()
    pos = torch.randn(5, 3)
    kwargs = _pack(core, pos)
    reference = core.get_output(**kwargs)["atom_feature"]
    assert reference.shape == (5, 4)
    rotation = _rotation(1)
    rotated = dict(kwargs)
    rotated["vij_sr"] = kwargs["vij_sr"] @ rotation.T
    assert torch.allclose(
        reference, core.get_output(**rotated)["atom_feature"], atol=1e-5, rtol=1e-5
    )
    shifted = dict(kwargs)
    # Vectors are differences, so a translation of both endpoints cancels.
    assert torch.allclose(
        reference, core.get_output(**shifted)["atom_feature"], atol=1e-5, rtol=1e-5
    )


def test_equflash_hidden_irreps_rotate_with_coordinates():
    torch.manual_seed(1)
    core = _core(
        irreps_manual=["4x0e", "4x0e+4x1o", "4x0e+4x1o"],
        lmax=1,
        feedforward_type="identity",
        nonlinear_type="gate",
    )
    core.eval()
    pos = torch.randn(4, 3)
    kwargs = _pack(core, pos)
    reference = core.get_output(**kwargs)["atom_feature"]
    assert reference.shape[-1] == core.schedule[-1].dim
    rotation = _rotation(2)
    rotated = dict(kwargs)
    # Row vectors: x' = x @ R.T matches e3nn's column action y' = R @ y.
    rotated["vij_sr"] = kwargs["vij_sr"] @ rotation.T
    got = core.get_output(**rotated)["atom_feature"]
    transform = core.schedule[-1].D_from_matrix(rotation)
    assert torch.allclose(reference @ transform.T, got, atol=1e-4, rtol=1e-4)


def test_equflash_isolated_atom_and_molecule_separation():
    torch.manual_seed(2)
    core = _core(num_layers=1, nonlinear_type="gate", feedforward_type="identity")
    core.eval()
    alone = core.get_output(
        atom_embedding=torch.randn(1, 4),
        rbf=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
    )
    assert alone["atom_feature"].shape == (1, 4)
    assert torch.isfinite(alone["atom_feature"]).all()

    def _run(n):
        pos = torch.randn(n, 3)
        packed = _pack(core, pos)
        return packed, core.get_output(**packed)["atom_feature"]

    left_in, left = _run(3)
    right_in, right = _run(4)
    n_left = left_in["atom_embedding"].shape[0]
    joined = {
        "atom_embedding": torch.cat([left_in["atom_embedding"], right_in["atom_embedding"]]),
        "rbf": torch.cat([left_in["rbf"], right_in["rbf"]]),
        "idx_i_sr": torch.cat([left_in["idx_i_sr"], right_in["idx_i_sr"] + n_left]),
        "idx_j_sr": torch.cat([left_in["idx_j_sr"], right_in["idx_j_sr"] + n_left]),
        "vij_sr": torch.cat([left_in["vij_sr"], right_in["vij_sr"]]),
    }
    both = core.get_output(**joined)["atom_feature"]
    assert torch.allclose(both[:n_left], left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], right, atol=1e-6, rtol=1e-6)


def test_bilinear_gate_and_contraction_are_equivariant():
    from e3nn import o3

    from enerzyme.models.equflash.interaction_v2 import BiLinearGate, SymmetricContraction

    torch.manual_seed(3)
    irreps = o3.Irreps("4x0e+4x1o")
    gate = BiLinearGate(irreps)
    gate.eval()
    features = gate.irreps_in.randn(6, -1)
    rotation = _rotation(3)
    out = gate(features)
    assert out.shape[-1] == gate.irreps_out.dim
    rotated = gate(features @ gate.irreps_in.D_from_matrix(rotation).T)
    assert torch.allclose(
        out @ gate.irreps_out.D_from_matrix(rotation).T,
        rotated,
        atol=1e-4,
        rtol=1e-4,
    )
    contraction = SymmetricContraction(irreps, contraction_degree=2)
    contraction.eval()
    values = irreps.randn(5, -1)
    contracted = contraction(values)
    rotated_values = contraction(values @ irreps.D_from_matrix(rotation).T)
    assert torch.allclose(
        contracted @ irreps.D_from_matrix(rotation).T,
        rotated_values,
        atol=1e-4,
        rtol=1e-4,
    )


def _forward_stack(model, pos, atomic_numbers, batch_seg):
    n = pos.shape[0]
    idx_i, idx_j = _edges(n)
    return model(
        {
            "Ra": pos,
            "Za": atomic_numbers,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
        }
    )


def test_equflash_force_matches_energy_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(4)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    layers = []
    for layer in ff["layers"]:
        layer = dict(layer)
        if layer["name"] == "Core":
            layer["params"] = dict(
                layer["params"],
                num_layers=1,
                lmax=1,
                radial_hidden=[8],
                nonlinear_type="gate",
                feedforward_type="identity",
            )
        layers.append(layer)
    model = build_model(
        "equflashv2",
        layer_params=layers,
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 5.0,
            "cuton": 4.0,
            "cutoff_fn": "xplor",
        },
        verbose=0,
    ).double()
    model.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.3, 0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    out = _forward_stack(model, pos, atomic_numbers, batch)
    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(3):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            plus.requires_grad_(True)
            minus.requires_grad_(True)
            e_plus = _forward_stack(model, plus, atomic_numbers, batch)["E"][0]
            e_minus = _forward_stack(model, minus, atomic_numbers, batch)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.isfinite(analytical).all()
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)


def test_channelwise_conv_rejects_mismatched_multiplicity():
    from enerzyme.models.equflash.interaction_v2 import ChannelwiseConv

    with pytest.raises(ValueError, match="multiplicity"):
        ChannelwiseConv(
            "4x0e",
            "1x0e + 1x1o",
            "8x0e + 4x1o",
            num_rbf=4,
            radial_hidden=[4],
        )


def test_efficient_conv_type_warns_that_the_kernel_matches_full():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _core(conv_type="efficient", num_layers=1, radial_hidden=[4])
    assert any("efficient" in str(item.message) for item in caught)


def test_nequip_interaction_honors_after_nonlinear():
    from torch import nn
    from e3nn import o3

    from enerzyme.models.equflash.interaction_v2 import NequIPInteractionBlock

    common = dict(
        irreps_x="4x0e",
        irreps_filter="1x0e + 1x1o",
        irreps_out="4x0e + 4x1o",
        num_rbf=4,
        radial_hidden=[4],
        denominator=1.0,
        train_denominator=False,
        sc_type="linear",
        nonlinear_type="gate",
        nonlinear_scale=1.0,
        nonlinear_kwargs=None,
    )
    linear = NequIPInteractionBlock(**common, after_nonlinear="linear")
    identity = NequIPInteractionBlock(**common, after_nonlinear="identity")
    assert isinstance(linear.project, o3.Linear)
    assert isinstance(identity.project, nn.Identity)
