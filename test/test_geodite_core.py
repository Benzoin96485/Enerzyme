"""Geodite Core: registration, equivariance, energy conservation, batch isolation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "geodite_layers_example.yaml"


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _edges(positions: torch.Tensor, cutoff: float):
    number = positions.shape[0]
    receivers, senders = [], []
    for i in range(number):
        for j in range(number):
            if i == j:
                continue
            if torch.linalg.vector_norm(positions[j] - positions[i]) < cutoff:
                receivers.append(i)
                senders.append(j)
    if not receivers:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    return torch.tensor(receivers), torch.tensor(senders)


def test_registration_defaults_and_yaml():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.geodite.interaction import parse_edge_updates
    from enerzyme.models.layers.isolated import IsolatedAtomResidualLayer
    from enerzyme.models.layers.radial_window import RadialWindowLayer
    from enerzyme.models.layers.readout import HierachicalReadout
    from enerzyme.models.layers.screened_zbl import ScreenedZBLRepulsionEnergyLayer

    for name in ("geodite", "Geodite"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "geodite"
        assert build_params["dim_embedding"] == 192
        assert build_params["num_rbf"] == 12
        assert build_params["cutoff_sr"] == 6.0
        assert build_params["Hartree_in_E"] == 27.211386245988
        assert layers[2]["name"] == "RadialWindow"
        assert layers[2]["params"]["attenuation"] == "covalent"
        assert layers[4]["name"] == "Core"
        assert "params" not in layers[4]
        assert layers[5]["name"] == "IsolatedAtomResidual"
        assert layers[6]["name"] == "HierachicalReadout"
        assert layers[7]["name"] == "ScreenedZBLRepulsionEnergy"

    spec = parse_edge_updates("linw_mlpa_ln")
    assert spec.widen and spec.activate_last and spec.reject
    assert spec.project == 1 and spec.project_norm == 1 and spec.gate == ""

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    torch.manual_seed(0)
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.architecture == "geodite"
    assert model.dim_embedding == 16
    assert model.num_layers == 2
    assert model.lmax == 2
    assert model.feature_irreps == "16x0e"
    assert model.blocks[0].message.update_edges is True
    assert model.blocks[-1].message.update_edges is False
    names = [type(layer).__name__ for layer in list(model.pre_sequence) + list(model.post_sequence)]
    assert "RadialWindowLayer" in names
    assert "IsolatedAtomResidualLayer" in names
    assert "HierachicalReadout" in names
    assert "ScreenedZBLRepulsionEnergyLayer" in names
    residual = next(layer for layer in model.post_sequence if isinstance(layer, IsolatedAtomResidualLayer))
    assert residual.dim_feature_out == 16
    assert residual.num_output_blocks == 2
    readout = next(layer for layer in model.post_sequence if isinstance(layer, HierachicalReadout))
    assert readout.num_blocks == 2
    assert readout.heads[0].bias is None
    window = next(layer for layer in model.pre_sequence if isinstance(layer, RadialWindowLayer))
    assert window.attenuation == "covalent"
    zbl = next(layer for layer in model.post_sequence if isinstance(layer, ScreenedZBLRepulsionEnergyLayer))
    assert zbl.envelope_p == 5

    try:
        from enerzyme.models.presets.cartesian_vector import geodite

        geodite(dim_embedding=16, num_heads=5)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("indivisible heads should fail")


def test_published_width_forward():
    from enerzyme.models.presets.cartesian_vector import geodite

    torch.manual_seed(1)
    core = geodite().eval()
    assert core.dim_embedding == 192
    assert core.num_layers == 4
    assert core.lmax == 2
    assert core.attn_dropout == 0.2
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.5, 0.3]])
    atomic_numbers = torch.tensor([8, 1, 1])
    embedding = torch.randn(3, 192)
    radial = torch.randn(6, 12)
    out = core.get_output(
        atom_embedding=embedding,
        rbf=radial,
        idx_i_sr=torch.tensor([0, 0, 1, 1, 2, 2]),
        idx_j_sr=torch.tensor([1, 2, 0, 2, 0, 1]),
        vij_sr=torch.randn(6, 3),
        Za=atomic_numbers,
    )
    assert out["atom_feature"].shape == (3, 192, 4)
    assert out["isolated_atom_feature"].shape == (3, 192, 4)
    assert out["atom_steerable_feature"].shape == (3, 8, 192)
    assert torch.isfinite(out["atom_feature"]).all()
    del positions


def test_molecules_do_not_exchange_messages():
    from enerzyme.models.presets.cartesian_vector import geodite

    torch.manual_seed(2)
    core = geodite(
        dim_embedding=16, num_rbf=4, max_Za=10, num_layers=2, num_heads=4, attn_dropout=0.0
    ).eval()
    table = torch.randn(11, 16)
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.3, 0.1]])
    right_pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, -0.2, 0.3]])
    left_z = torch.tensor([6, 1, 1])
    right_z = torch.tensor([8, 1])

    def run(positions, atomic_numbers, radial):
        idx_i, idx_j = _edges(positions, 6.0)
        displacement = positions[idx_j] - positions[idx_i]
        out = core.get_output(
            table[atomic_numbers], radial, idx_i, idx_j, displacement, atomic_numbers
        )
        return out["atom_feature"], idx_i, idx_j

    left_edges = _edges(left_pos, 6.0)
    right_edges = _edges(right_pos, 6.0)
    left_radial = torch.randn(left_edges[0].shape[0], 4)
    right_radial = torch.randn(right_edges[0].shape[0], 4)
    left, _, _ = run(left_pos, left_z, left_radial)
    right, _, _ = run(right_pos, right_z, right_radial)
    joined_pos = torch.cat([left_pos, right_pos + torch.tensor([30.0, -10.0, 4.0])])
    receivers = torch.cat([left_edges[0], right_edges[0] + left_pos.shape[0]])
    senders = torch.cat([left_edges[1], right_edges[1] + left_pos.shape[0]])
    both = core.get_output(
        table[torch.cat([left_z, right_z])],
        torch.cat([left_radial, right_radial]),
        receivers,
        senders,
        joined_pos[senders] - joined_pos[receivers],
        torch.cat([left_z, right_z]),
    )
    assert torch.allclose(both["atom_feature"][:3], left, atol=1e-5, rtol=1e-5)
    assert torch.allclose(both["atom_feature"][3:], right, atol=1e-5, rtol=1e-5)


def _stack(**overrides):
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False, "apply_cutoff_fn": False}},
        {
            "name": "RadialWindow",
            "params": {
                "envelope": "polynomial",
                "envelope_p": 6,
                "attenuation": "covalent",
                "trainable_attenuation": True,
            },
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_layers": 2,
                "lmax": 2,
                "num_heads": 4,
                "attn_dropout": 0.0,
            },
        },
        {"name": "IsolatedAtomResidual"},
        {
            "name": "HierachicalReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "use_bias": False,
                "initial_weight": "xavier_uniform",
            },
        },
        {"name": "ScreenedZBLRepulsionEnergy", "params": {"envelope_p": 5}},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    build = {
        "dim_embedding": 16,
        "num_rbf": 4,
        "max_Za": 10,
        "cutoff_sr": 6.0,
        "Hartree_in_E": 27.211386245988,
        "Bohr_in_R": 0.5291772105638411,
    }
    build.update(overrides)
    return build_model("geodite", layer_params=layers, build_params=build, verbose=0)


def _forward(model, positions, atomic_numbers, batch):
    number = positions.shape[0]
    receivers, senders = [], []
    for i in range(number):
        for j in range(number):
            if i != j and torch.linalg.vector_norm(positions[j] - positions[i]) < 6.0:
                receivers.append(i)
                senders.append(j)
    return model(
        {
            "Ra": positions,
            "Za": atomic_numbers,
            "idx_i": torch.tensor(receivers, dtype=torch.long),
            "idx_j": torch.tensor(senders, dtype=torch.long),
            "batch_seg": batch,
        }
    )


def test_isolated_atom_network_energy_is_zero_and_short_range_repulsion_is_positive():
    torch.manual_seed(4)
    model = _stack().double().eval()
    lonely = torch.tensor([[0.2, -0.1, 0.4]], dtype=torch.float64)
    atomic_number = torch.tensor([6])
    empty = torch.zeros(0, dtype=torch.long)
    # Stop before Force: with no edges the energy does not depend on coordinates.
    data = {
        "Ra": lonely,
        "Za": atomic_number,
        "idx_i": empty,
        "idx_j": empty,
        "batch_seg": torch.zeros(1, dtype=torch.long),
    }
    for layer in model.pre_sequence:
        data = layer(data)
    data = model._forward(data)
    for layer in model.post_sequence:
        if type(layer).__name__ == "ForceLayer":
            break
        data = layer(data)
    assert torch.allclose(data["E"], torch.zeros(1, dtype=torch.float64), atol=1e-8)
    assert torch.count_nonzero(data["Ea"]) == 0

    close = torch.tensor([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    packed = _forward(model, close, torch.tensor([1, 1]), torch.zeros(2, dtype=torch.long))
    assert packed["E_zbl_a"].sum() > 0
    assert torch.isfinite(packed["Fa"]).all()


def test_stack_is_invariant_and_forces_are_conservative():
    torch.manual_seed(5)
    model = _stack().double().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.35, 0.15, 0.0], [0.2, 1.4, 0.25], [2.2, 0.4, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 6])
    batch = torch.zeros(4, dtype=torch.long)
    base = _forward(model, positions, atomic_numbers, batch)
    assert base["E"].shape == (1,)
    assert base["Fa"].shape == (4, 3)
    assert torch.isfinite(base["Fa"]).all()

    shift = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    moved = _forward(model, (positions.detach() + shift).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(base["Fa"], moved["Fa"], atol=1e-8, rtol=1e-8)

    rotation = _rotation(6).double()
    rotated_pos = (positions.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, atomic_numbers, batch)
    assert torch.allclose(base["E"], rotated["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    inverted_pos = (-positions.detach()).requires_grad_(True)
    inverted = _forward(model, inverted_pos, atomic_numbers, batch)
    assert torch.allclose(base["E"], inverted["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(-base["Fa"], inverted["Fa"], atol=1e-6, rtol=1e-6)

    analytical = base["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), atomic_numbers, batch)["E"]
            e_minus = _forward(model, minus.requires_grad_(True), atomic_numbers, batch)["E"]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=2e-5, rtol=1e-5)


def test_two_molecules_keep_separate_total_energies():
    torch.manual_seed(7)
    model = _stack().double().eval()
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.1, 1.3, 0.2],
            [30.0, -4.0, 2.0],
            [31.4, -3.8, 2.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([6, 1, 1, 8, 1])
    batch = torch.tensor([0, 0, 0, 1, 1])
    both = _forward(model, positions, atomic_numbers, batch)
    left = _forward(
        model,
        positions[:3].detach().requires_grad_(True),
        atomic_numbers[:3],
        torch.zeros(3, dtype=torch.long),
    )
    right = _forward(
        model,
        positions[3:].detach().requires_grad_(True),
        atomic_numbers[3:],
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(both["E"][0], left["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(both["E"][1], right["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(both["Ea"][:3], left["Ea"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(both["Ea"][3:], right["Ea"], atol=1e-8, rtol=1e-8)
