"""GRACE Core: presets, equivariance, message passing, energy conservation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "grace_layers_example.yaml"


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _edges(positions: torch.Tensor, cutoff: float):
    receivers, senders = [], []
    number = positions.shape[0]
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


def _core_inputs(positions, atomic_numbers, embedding, cutoff):
    idx_i, idx_j = _edges(positions, cutoff)
    if idx_i.numel() == 0:
        return {
            "atom_embedding": embedding,
            "Za": atomic_numbers,
            "Dij_sr": positions.new_zeros(0),
            "vij_sr": positions.new_zeros(0, 3),
            "idx_i_sr": idx_i,
            "idx_j_sr": idx_j,
        }
    displacement = positions[idx_j] - positions[idx_i]
    return {
        "atom_embedding": embedding,
        "Za": atomic_numbers,
        "Dij_sr": torch.linalg.vector_norm(displacement, dim=-1),
        "vij_sr": displacement,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }


def _tiny(preset="custom", **overrides):
    from enerzyme.models.grace import GRACECore

    params = dict(max_Za=10, dim_embedding=8, cutoff_sr=5.0, preset=preset)
    params.update(overrides)
    return GRACECore(**params)


def test_registration_presets_and_yaml():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.grace import PAPER_PRESETS
    from enerzyme.models.grace.readout import GRACEEnergyReadout

    for name in ("grace", "GRACE"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "GRACECore"
        assert build_params["cutoff_sr"] == 6.0
        assert build_params["dim_embedding"] == 128
        assert layers[1]["name"] == "RandomAtomEmbedding"
        assert layers[2]["params"]["preset"] == "1L"
        assert layers[3]["name"] == "GRACEEnergyReadout"

    paper = PAPER_PRESETS["1L"]
    assert paper["basis"] == "chebyshev"
    assert paper["n_rad_base"] == 8
    assert paper["embedding_size"] == 128
    assert paper["cutoff"] == 6.0
    assert paper["n_mlp_dens"] == 12
    assert [order.lmax for order in paper["layers"][0]["orders"]] == [4, 4, 0, 0]
    assert [order.channels for order in paper["layers"][0]["orders"]] == [32, 32, 32, 32]
    medium = PAPER_PRESETS["1L-M"]
    assert medium["n_rad_base"] == 10 and medium["n_mlp_dens"] == 16
    large = PAPER_PRESETS["1L-L"]
    assert [order.channels for order in large["layers"][0]["orders"]] == [32, 64, 64, 64]
    two = PAPER_PRESETS["2L"]
    assert [order.lmax for order in two["layers"][0]["orders"]] == [4, 4, 1, 0]
    assert [order.lmax for order in two["layers"][1]["orders"]] == [4, 4, 0, 0]
    assert two["layers"][0]["indicator_lmax"] == 1
    two_large = PAPER_PRESETS["2L-L"]
    assert [order.lmax for order in two_large["layers"][0]["orders"]] == [4, 4, 3, 1]
    assert two_large["layers"][0]["indicator_lmax"] == 3
    assert PAPER_PRESETS["2L-latest"] is two_large
    latest = PAPER_PRESETS["1L-latest"]
    assert latest["n_rad_base"] == 10
    assert [order.channels for order in latest["layers"][0]["orders"]][1] == 64
    fs = PAPER_PRESETS["FS"]
    assert fs["basis"] == "sbessel" and fs["readout"] == "finnis_sinclair"
    assert fs["embedding_size"] == 64 and fs["cutoff"] == 7.0
    assert fs["radial_hidden"] == 0
    assert [order.channels for order in fs["layers"][0]["orders"]] == [20, 15, 10, 5]
    assert [order.lmax for order in fs["layers"][0]["orders"]] == [5, 3, 0, 0]
    assert fs["fs_terms"][1] == (1.0, 0.5)

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], max_Za=12),
        verbose=0,
    )
    assert model.__class__.__name__ == "GRACECore"
    assert model.preset == "custom"
    assert model.n_densities == 3
    assert model.feature_irreps == "3x0e"
    assert model.num_layers == 1
    readout = next(layer for layer in model.post_sequence if isinstance(layer, GRACEEnergyReadout))
    assert readout.head == "linear_mlp"
    assert readout.mlp is not None

    try:
        _tiny(preset="nope")
    except ValueError as exc:
        assert "preset" in str(exc).lower() or "Unknown" in str(exc)
    else:
        raise AssertionError("unknown preset should fail")


def test_custom_forward_shapes_and_isolated_atoms_depend_only_on_species():
    torch.manual_seed(0)
    core = _tiny().eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.3, 0.2]])
    atomic_numbers = torch.tensor([6, 1, 1])
    embedding = torch.randn(3, 8)
    out = core.get_output(**_core_inputs(positions, atomic_numbers, embedding, 5.0))
    assert out["atom_feature"].shape == (3, 3)
    assert torch.isfinite(out["atom_feature"]).all()

    far = torch.tensor([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 1.0]])
    separated = core.get_output(**_core_inputs(far, atomic_numbers, embedding, 5.0))
    alone = []
    for index in range(3):
        alone.append(
            core.get_output(
                **_core_inputs(
                    far[index : index + 1],
                    atomic_numbers[index : index + 1],
                    embedding[index : index + 1],
                    5.0,
                )
            )["atom_feature"]
        )
    assert torch.allclose(separated["atom_feature"], torch.cat(alone, dim=0), atol=1e-6, rtol=1e-6)
    # No neighbors: the basis is zero, so the density is the species bias.
    with torch.no_grad():
        core.density.linear.bias.zero_()
        core.density.linear.bias[6, 0] = 0.3
        core.density.linear.bias[1, 1] = -0.2
    carbon = core.get_output(
        **_core_inputs(far[:1], atomic_numbers[:1], embedding[:1], 5.0)
    )["atom_feature"]
    hydrogen = core.get_output(
        **_core_inputs(far[1:2], atomic_numbers[1:2], embedding[1:2] * 3, 5.0)
    )["atom_feature"]
    assert torch.allclose(carbon, torch.tensor([[0.3, 0.0, 0.0]]), atol=1e-6)
    assert torch.allclose(hydrogen, torch.tensor([[0.0, -0.2, 0.0]]), atol=1e-6)


def test_molecules_do_not_exchange_messages_and_species_enter_the_basis():
    torch.manual_seed(1)
    core = _tiny().eval()
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]])
    right_pos = torch.tensor([[0.0, 0.0, 0.0], [1.4, -0.2, 0.3]])
    left_z = torch.tensor([6, 1])
    right_z = torch.tensor([8, 1])
    left_h = torch.randn(2, 8)
    right_h = torch.randn(2, 8)
    left = core.get_output(**_core_inputs(left_pos, left_z, left_h, 5.0))["atom_feature"]
    right = core.get_output(**_core_inputs(right_pos, right_z, right_h, 5.0))["atom_feature"]
    joined_pos = torch.cat([left_pos, right_pos + torch.tensor([25.0, -4.0, 3.0])], dim=0)
    joined = core.get_output(
        **_core_inputs(
            joined_pos,
            torch.cat([left_z, right_z]),
            torch.cat([left_h, right_h]),
            5.0,
        )
    )["atom_feature"]
    assert torch.allclose(joined[:2], left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(joined[2:], right, atol=1e-6, rtol=1e-6)

    swapped = core.get_output(
        **_core_inputs(left_pos, torch.tensor([8, 1]), left_h, 5.0)
    )["atom_feature"]
    assert not torch.allclose(swapped, left, atol=1e-5, rtol=1e-5)


def test_second_layer_message_changes_the_densities():
    torch.manual_seed(2)
    core = _tiny(preset="custom-2L").eval()
    assert core.num_layers == 2
    assert core.indicator_lmax[0] == 1
    assert core.layers[0].indicator is not None
    assert core.layers[1].atomic.edge_product is not None
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.15, 0.0], [-0.2, 1.25, 0.3]])
    atomic_numbers = torch.tensor([6, 1, 8])
    embedding = torch.randn(3, 8)
    kwargs = _core_inputs(positions, atomic_numbers, embedding, 5.0)
    baseline = core.get_output(**kwargs)["atom_feature"]
    with torch.no_grad():
        for parameter in core.layers[0].indicator.parameters():
            parameter.zero_()
    silenced = core.get_output(**kwargs)["atom_feature"]
    assert not torch.allclose(baseline, silenced, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(silenced).all()


def test_densities_are_invariant_under_rotation_and_translation():
    torch.manual_seed(3)
    core = _tiny().eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.2, 0.4], [2.0, 0.3, 1.1]])
    atomic_numbers = torch.tensor([8, 1, 1, 6])
    embedding = torch.randn(4, 8)
    base = core.get_output(**_core_inputs(positions, atomic_numbers, embedding, 5.0))["atom_feature"]
    shift = torch.tensor([2.0, -1.0, 0.4])
    moved = core.get_output(
        **_core_inputs(positions + shift, atomic_numbers, embedding, 5.0)
    )["atom_feature"]
    assert torch.allclose(base, moved, atol=1e-6, rtol=1e-6)
    rotation = _rotation(4)
    rotated = core.get_output(
        **_core_inputs(positions @ rotation, atomic_numbers, embedding, 5.0)
    )["atom_feature"]
    assert torch.allclose(base, rotated, atol=1e-5, rtol=1e-5)
    inverted = core.get_output(
        **_core_inputs(-positions, atomic_numbers, embedding, 5.0)
    )["atom_feature"]
    assert torch.allclose(base, inverted, atol=1e-5, rtol=1e-5)


def _stack(preset="custom"):
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"preset": preset}},
        {"name": "GRACEEnergyReadout"},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    return build_model(
        "grace",
        layer_params=layers,
        build_params={
            "max_Za": 10,
            "dim_embedding": 8,
            "cutoff_sr": 5.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    )


def _forward(model, positions, atomic_numbers, batch):
    number = positions.shape[0]
    receivers, senders = [], []
    for i in range(number):
        for j in range(number):
            if i != j:
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


def test_stack_is_invariant_and_forces_are_conservative():
    torch.manual_seed(4)
    model = _stack().double().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.25, 0.1, 0.0], [0.15, 1.3, 0.2], [2.1, 0.3, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 6])
    batch = torch.zeros(4, dtype=torch.long)
    base = _forward(model, positions, atomic_numbers, batch)
    assert base["E"].shape == (1,)
    assert base["Fa"].shape == (4, 3)
    assert base["atom_feature"].shape == (4, 3)
    assert torch.isfinite(base["Fa"]).all()

    shift = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    moved = _forward(model, (positions.detach() + shift).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(base["Fa"], moved["Fa"], atol=1e-7, rtol=1e-7)

    rotation = _rotation(5).double()
    rotated = _forward(model, (positions.detach() @ rotation).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], rotated["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    inverted = _forward(model, (-positions.detach()).requires_grad_(True), atomic_numbers, batch)
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
    assert torch.allclose(analytical, numerical, atol=2e-5, rtol=1e-4)


def test_two_molecules_keep_separate_total_energies_and_simple_readout_fits():
    from enerzyme.models.ff import build_model

    torch.manual_seed(5)
    model = _stack(preset="custom-2L").double().eval()
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.15, 0.05, 0.0],
            [0.1, 1.2, 0.1],
            [20.0, 0.0, 0.0],
            [21.3, -0.2, 0.2],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([6, 1, 1, 8, 1])
    batch = torch.tensor([0, 0, 0, 1, 1])
    together = _forward(model, positions, atomic_numbers, batch)
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
    assert torch.allclose(together["E"][0], left["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(together["E"][1], right["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(together["Fa"][:3], left["Fa"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(together["Fa"][3:], right["Fa"], atol=1e-7, rtol=1e-7)

    simple = build_model(
        "grace",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"preset": "custom"}},
            {"name": "SimpleReadout", "params": {"output_fields": ["Ea"], "head_type": "dense"}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
        build_params={"max_Za": 8, "dim_embedding": 8, "cutoff_sr": 5.0, "cutoff_fn": "cosine"},
        verbose=0,
    )
    output = _forward(
        simple,
        positions[:3].detach().float().requires_grad_(True),
        atomic_numbers[:3],
        torch.zeros(3, dtype=torch.long),
    )
    assert output["E"].shape == (1,)
    assert output["Fa"].shape == (3, 3)
    assert torch.isfinite(output["E"]).all()
