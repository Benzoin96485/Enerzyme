"""MACE layer features, the external energy head, and MACE-POLAR-1."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]


def _edges(n: int):
    left, right = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                left.append(i)
                right.append(j)
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    return rotation


def _tiny_core(**overrides):
    from enerzyme.models.presets.irreps import mace

    params = dict(
        max_Za=10,
        max_ell=1,
        dim_embedding=8,
        num_rbf=4,
        additional_hidden_irreps="8x1o",
        correlation=2,
        num_interactions=2,
        avg_num_neighbors=2.0,
        MLP_irreps="4x0e",
        radial_MLP=[8, 8],
        gate="silu",
        output_mode="layers",
    )
    params.update(overrides)
    return mace(**params)


def _forward_inputs(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    idx_i, idx_j = _edges(n)
    return dict(
        Za=torch.tensor([6, 1, 8, 1][:n], dtype=torch.long),
        vij_sr=torch.randn(idx_i.numel(), 3, generator=generator),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4, generator=generator),
        atom_embedding=torch.randn(n, 8, generator=generator),
    )


def test_bessel_mace_flavor_matches_envelope():
    from enerzyme.models.mp.basis.envelope import polynomial_envelope
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 5.0
    order = 5
    n_basis = 4
    layer = BesselRBFLayer(
        n_basis, cutoff, flavor="mace", trainable=False, polynomial_p=order
    )
    distance = torch.tensor([0.0, 0.4, 2.5, 4.9, 5.0, 6.0])
    values = layer.get_rbf(distance)
    frequencies = math.pi / cutoff * torch.arange(1, n_basis + 1, dtype=distance.dtype)
    argument = distance.view(-1, 1) * frequencies.view(1, -1)
    safe = distance.view(-1, 1).clamp_min(1e-12)
    quotient = torch.where(
        distance.view(-1, 1) == 0,
        frequencies.view(1, -1),
        torch.sin(argument) / safe,
    )
    envelope = polynomial_envelope(distance.view(-1, 1), cutoff, order)
    manual = math.sqrt(2.0 / cutoff) * quotient * envelope
    assert torch.allclose(values, manual, atol=1e-6)
    assert torch.allclose(values[-2:], torch.zeros_like(values[-2:]), atol=1e-6)
    assert values[0, 0] > 0


def test_layers_mode_stacks_full_irreps_and_direct_mode_stays():
    torch.manual_seed(0)
    core = _tiny_core()
    out = core.get_output(**_forward_inputs(4))
    assert out["atom_feature"].shape == (4, core.hidden_irreps.dim, 2)
    assert core.feature_irreps == str(core.hidden_irreps)
    assert core.num_output_blocks == 2
    assert core.dim_feature_out == o3.Irreps(core.feature_irreps).count("0e")

    direct = _tiny_core(output_mode="direct", keep_last_layer_irreps=False)
    direct_out = direct.get_output(**_forward_inputs(4))
    assert set(direct_out) == {"Ea", "Qa"}
    assert direct_out["Ea"].shape == (4,)
    (direct_out["Ea"].sum() + direct_out["Qa"].sum()).backward()


def test_mace_stack_energy_is_invariant_and_forces_match():
    from enerzyme.models.ff import build_model

    torch.manual_seed(1)
    model = build_model("mace", layer_params=_mace_layers(), build_params=_build(), verbose=0)
    model = model.double()
    model.eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.2, 0.9, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    out = _run(model, positions, atomic_numbers, batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (3, 3)
    rotation = _rotation(2).double()
    rotated_positions = (positions.detach() @ rotation).requires_grad_(True)
    rotated = _run(model, rotated_positions, atomic_numbers, batch)
    assert torch.allclose(rotated["E"], out["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(rotated["Fa"], out["Fa"] @ rotation, atol=1e-6, rtol=1e-6)

    _assert_forces(model, positions, atomic_numbers, batch, out["Fa"])


def test_polar_conserves_charge_and_spin_per_molecule():
    from enerzyme.models.mace.polar import PolarFieldLayer

    torch.manual_seed(2)
    layer = PolarFieldLayer(
        max_Za=10,
        feature_irreps="4x0e+4x1o",
        num_output_blocks=2,
        num_update=1,
        mlp_hidden=8,
        nonlocal_hidden=8,
        readout_irreps="8x0e+4x1o",
        fukui_irreps="4x0e",
        field_smearing_widths=[1.5, 2.0],
    ).double()
    n = 4
    features = torch.randn(n, o3.Irreps("4x0e+4x1o").dim, 2, dtype=torch.float64)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [6.0, 0.0, 0.0], [7.1, -0.2, 0.4]],
        dtype=torch.float64,
    )
    atomic_numbers = torch.tensor([6, 1, 8, 1])
    batch = torch.tensor([0, 0, 1, 1])
    charge = torch.tensor([1.0, -2.0], dtype=torch.float64)
    spin = torch.tensor([1.0, 0.0], dtype=torch.float64)
    out = layer.get_output(
        atom_feature=features,
        Za=atomic_numbers,
        Ra=positions,
        batch_seg=batch,
        Q=charge,
        S=spin,
    )
    for graph in range(2):
        mask = batch == graph
        assert torch.allclose(out["Qa"][mask].sum(), charge[graph], atol=1e-6)
        assert torch.allclose(out["Sa"][mask].sum(), spin[graph], atol=1e-6)
    per_atom_charge = torch.tensor([1.0, 1.0, -2.0, -2.0], dtype=torch.float64)
    repeated = layer.get_output(
        atom_feature=features,
        Za=atomic_numbers,
        Ra=positions,
        batch_seg=batch,
        Q=per_atom_charge,
        S=torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(repeated["Qa"][:2].sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
    neutral = layer.get_output(
        atom_feature=features, Za=atomic_numbers, Ra=positions, batch_seg=batch
    )
    assert torch.allclose(neutral["Qa"].view(2, 2).sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-6)
    try:
        layer.get_output(
            atom_feature=features,
            Za=atomic_numbers,
            Ra=positions,
            batch_seg=batch,
            pbc=torch.ones(2, 3, dtype=torch.bool),
        )
    except NotImplementedError:
        pass
    else:
        raise AssertionError("periodic batches should be rejected")


def test_polar_field_monopole_follows_rotated_features():
    """Monopole PolarField is invariant once the node features rotate as irreps.

    Row vectors use ``positions @ R``. The matching feature map is ``f @ D(R)``.
    """
    from enerzyme.models.mace.polar import PolarFieldLayer

    torch.manual_seed(0)
    irreps = o3.Irreps("4x0e+4x1o")
    layer = PolarFieldLayer(
        max_Za=10,
        feature_irreps=str(irreps),
        num_output_blocks=2,
        density_max_l=0,
        field_max_l=0,
        field_smearing_widths=[1.5],
        num_update=1,
        mlp_hidden=8,
        nonlocal_hidden=8,
        readout_irreps="8x0e+4x1o",
        fukui_irreps="4x0e",
    ).double()
    layer.eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.15, 0.15, -0.05], [0.1, 1.05, 0.35]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1])
    batch = torch.zeros(3, dtype=torch.long)
    charge = torch.tensor([1.0], dtype=torch.float64)
    spin = torch.tensor([1.0], dtype=torch.float64)
    features = torch.randn(3, irreps.dim, 2, dtype=torch.float64)

    def energy(pos, feat):
        out = layer.get_output(
            atom_feature=feat, Za=atomic_numbers, Ra=pos, batch_seg=batch, Q=charge, S=spin
        )
        return (out["E_coulomb_a"] + out["E_nonlocal_a"]).sum(), out

    reference, out = energy(positions, features)
    forces = -torch.autograd.grad(reference, positions)[0]
    rotation = _rotation(5).double()
    representation = irreps.D_from_matrix(rotation)
    rotated_features = torch.einsum("nti,ij->ntj", features.transpose(1, 2), representation)
    rotated_features = rotated_features.transpose(1, 2)
    rotated_positions = (positions.detach() @ rotation).requires_grad_(True)
    rotated_energy, rotated = energy(rotated_positions, rotated_features)
    rotated_forces = -torch.autograd.grad(rotated_energy, rotated_positions)[0]
    assert torch.allclose(rotated["Qa"], out["Qa"], atol=1e-8)
    assert torch.allclose(rotated["Sa"], out["Sa"], atol=1e-8)
    assert torch.allclose(rotated_energy, reference, atol=1e-8)
    assert torch.allclose(rotated_forces, forces @ rotation, atol=1e-6)

    parity = -torch.eye(3, dtype=torch.float64)
    parity_features = torch.einsum(
        "nti,ij->ntj", features.transpose(1, 2), irreps.D_from_matrix(parity)
    ).transpose(1, 2)
    inverted_positions = (-positions.detach()).requires_grad_(True)
    inverted_energy, inverted = energy(inverted_positions, parity_features)
    inverted_forces = -torch.autograd.grad(inverted_energy, inverted_positions)[0]
    assert torch.allclose(inverted_energy, reference, atol=1e-8)
    assert torch.allclose(inverted_forces, -forces, atol=1e-6)
    assert torch.allclose(inverted["Qa"].sum(), charge, atol=1e-8)


def test_mace_polar_dipole_forces_match_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(4)
    model = build_model(
        "mace-polar-1",
        layer_params=_polar_layers(),
        build_params=_build(),
        verbose=0,
    ).double()
    model.eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.15, 0.15, -0.05], [0.1, 1.05, 0.35]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1])
    batch = torch.zeros(3, dtype=torch.long)
    charge = torch.tensor([0.0], dtype=torch.float64)
    spin = torch.tensor([0.0], dtype=torch.float64)
    out = _run(model, positions, atomic_numbers, batch, charge, spin)
    assert torch.allclose(out["Qa"].sum(), charge[0], atol=1e-6)
    assert out["spin_multipole"].shape[-2:] == (2, 4)
    assert torch.isfinite(out["E_coulomb_a"]).all()
    _assert_forces(
        model, positions, atomic_numbers, batch, out["Fa"], charge, spin, atol=2e-4
    )


def test_example_yaml_names_match_the_defaults():
    from enerzyme.models.mace.defaults import DEFAULT_LAYER_PARAMS
    from enerzyme.models.mace.polar import DEFAULT_LAYER_PARAMS_POLAR

    mace = yaml.safe_load((ROOT / "enerzyme/config/mace_layers_example.yaml").read_text())
    polar = yaml.safe_load((ROOT / "enerzyme/config/mace_polar_layers_example.yaml").read_text())
    mace_layers = mace["Modelhub"]["internal_FFs"]["FF01"]["layers"]
    polar_layers = polar["Modelhub"]["internal_FFs"]["FF01"]["layers"]
    assert [layer["name"] for layer in mace_layers] == [layer["name"] for layer in DEFAULT_LAYER_PARAMS]
    assert [layer["name"] for layer in polar_layers] == [
        layer["name"] for layer in DEFAULT_LAYER_PARAMS_POLAR
    ]
    assert polar["Modelhub"]["internal_FFs"]["FF01"]["architecture"] == "mace_polar"


def _build():
    return {
        "dim_embedding": 8,
        "num_rbf": 4,
        "max_Za": 10,
        "cutoff_sr": 5.0,
        "avg_num_neighbors": 2.0,
    }


def _mace_layers():
    return [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"flavor": "mace", "trainable": False, "polynomial_p": 5}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "max_ell": 1,
                "additional_hidden_irreps": "8x1o",
                "correlation": 2,
                "num_interactions": 2,
                "output_mode": "layers",
                "keep_last_layer_irreps": True,
                "radial_MLP": [8, 8],
            },
        },
        {"name": "MACEEnergyReadout", "params": {"MLP_irreps": "4x0e"}},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


def _polar_layers(**polar_params):
    layers = _mace_layers()
    params = {
        "num_update": 1,
        "mlp_hidden": 8,
        "nonlocal_hidden": 8,
        "readout_irreps": "8x0e+4x1o",
        "fukui_irreps": "4x0e",
        "field_smearing_widths": [1.5, 2.0],
    }
    params.update(polar_params)
    layers.insert(-2, {"name": "PolarField", "params": params})
    return layers


def _assert_forces(model, positions, atomic_numbers, batch, analytical, charge=None, spin=None, atol=1e-4):
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(positions.shape[0]):
        for coord in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            plus.requires_grad_(True)
            minus.requires_grad_(True)
            e_plus = _run(model, plus, atomic_numbers, batch, charge, spin)["E"].sum()
            e_minus = _run(model, minus, atomic_numbers, batch, charge, spin)["E"].sum()
            numerical[atom, coord] = -(e_plus - e_minus) / (2 * step)
    assert torch.allclose(analytical, numerical, atol=atol, rtol=1e-4)


def _run(model, positions, atomic_numbers, batch, charge=None, spin=None):
    idx_i, idx_j = _edges(positions.shape[0])
    payload = {
        "Ra": positions,
        "Za": atomic_numbers,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "batch_seg": batch,
    }
    if charge is not None:
        payload["Q"] = charge
    if spin is not None:
        payload["S"] = spin
    return model(payload)
