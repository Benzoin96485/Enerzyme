"""CAMP Core: presets, invariance, batching, and energy conservation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "camp_layers_example.yaml"


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


def _inputs(positions, embedding, atomic_numbers, cutoff):
    idx_i, idx_j = _edges(positions, cutoff)
    if idx_i.numel() == 0:
        distance = positions.new_zeros(0)
        vectors = positions.new_zeros(0, 3)
    else:
        vectors = positions[idx_j] - positions[idx_i]
        distance = torch.linalg.vector_norm(vectors, dim=-1)
    return {
        "atom_embedding": embedding,
        "Za": atomic_numbers,
        "Dij_sr": distance,
        "vij_sr": vectors,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }


def _tiny(**overrides):
    from enerzyme.models.presets.cartesian_tensor import camp

    params = dict(
        dim_embedding=2,
        max_Za=8,
        cutoff_sr=5.0,
        preset="custom",
        max_v=1,
        num_layers=2,
        max_chebyshev_degree=2,
        radial_mlp_hidden_layers=0,
        output_mlp_hidden_layers=0,
        avg_num_neighbors=1.0,
        atomic_numbers=[1, 6, 8],
    )
    params.update(overrides)
    return camp(**params)


def test_registration_presets_and_yaml():
    from enerzyme.models.camp import PAPER_PRESETS
    from enerzyme.models.camp.readout import CAMPEnergyReadout
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.atom_embedding import RandomAtomEmbedding

    for name in ("camp", "CAMP"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "camp"
        assert build_params["dim_embedding"] == 33
        assert build_params["cutoff_sr"] == 5.0
        assert layers[0]["name"] == "RangeSeparation"
        assert layers[1]["name"] == "RandomAtomEmbedding"
        assert layers[2]["params"]["preset"] == "water"
        assert layers[3]["name"] == "CAMPEnergyReadout"

    water = PAPER_PRESETS["water"]
    assert water["max_u"] == 32 and water["dim_embedding"] == 33
    assert water["max_v"] == 2 and water["num_layers"] == 3 and water["cutoff_sr"] == 5.0
    assert PAPER_PRESETS["lips-10"]["cutoff_sr"] == 6.0
    assert PAPER_PRESETS["lips-10"]["max_v"] == 3 and PAPER_PRESETS["lips-10"]["dim_embedding"] == 17
    assert PAPER_PRESETS["lips-2500"]["max_u"] == 64 and PAPER_PRESETS["lips-2500"]["num_layers"] == 3
    assert PAPER_PRESETS["aspirin"]["max_u"] == 48 and PAPER_PRESETS["ethanol"]["cutoff_sr"] == 6.0
    assert PAPER_PRESETS["salicylic"]["cutoff_sr"] == 5.0
    assert PAPER_PRESETS["uracil"]["num_layers"] == 3 and PAPER_PRESETS["graphene"]["max_v"] == 3
    assert PAPER_PRESETS["water"]["max_chebyshev_degree"] == 8
    assert PAPER_PRESETS["water"]["envelope_p"] == 6

    published = _tiny(
        dim_embedding=33,
        preset="water",
        max_v=None,
        num_layers=None,
        max_chebyshev_degree=None,
        atomic_numbers=[1, 8],
    )
    assert published.max_v == 2 and published.num_layers == 3
    assert published.max_chebyshev_degree == 8 and published.n_species == 2
    assert published.dim_feature_out == 33 and published.num_output_blocks == 3
    assert published.feature_irreps == "33x0e"
    assert published.blocks[0].atomic.radial.coeff.shape == (2, 2, 33, 9)
    assert published.blocks[0].atomic.radial.coeff is not published.blocks[1].atomic.radial.coeff

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"]),
        verbose=0,
    )
    assert model.__class__.__name__ == "ComposableCore"
    assert model.max_v == 2 and model.num_layers == 2 and model.n_species == 3
    assert any(isinstance(layer, RandomAtomEmbedding) for layer in model.pre_sequence)
    readout = next(layer for layer in model.post_sequence if isinstance(layer, CAMPEnergyReadout))
    assert len(readout.heads) == 2
    assert readout.heads[0].out_features == 1

    try:
        _tiny(preset="nope")
    except ValueError as exc:
        assert "preset" in str(exc).lower()
    else:
        raise AssertionError("unknown preset should fail")


def test_isolated_atoms_have_zero_features_and_unknown_elements_fail():
    torch.manual_seed(0)
    core = _tiny().double()
    positions = torch.tensor([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=torch.float64)
    embedding = torch.randn(2, 2, dtype=torch.float64)
    atomic_numbers = torch.tensor([8, 1])
    out = core.get_output(**_inputs(positions, embedding, atomic_numbers, 5.0))
    assert out["atom_feature"].shape == (2, 2, 2)
    assert torch.allclose(out["atom_feature"], torch.zeros_like(out["atom_feature"]))

    try:
        core.get_output(
            **_inputs(positions, embedding, torch.tensor([7, 1]), 5.0)
        )
    except ValueError as exc:
        assert "7" in str(exc)
    else:
        raise AssertionError("Z=7 is outside the species table")


def test_scalar_features_are_invariant_and_ignore_edge_order():
    torch.manual_seed(1)
    core = _tiny(max_v=2).double()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.2, 1.2, 0.15]],
        dtype=torch.float64,
    )
    embedding = torch.randn(3, 2, dtype=torch.float64)
    atomic_numbers = torch.tensor([8, 1, 1])
    base = core.get_output(**_inputs(positions, embedding, atomic_numbers, 5.0))["atom_feature"]
    rotation = _rotation(2).double()
    rotated = core.get_output(
        **_inputs(positions @ rotation, embedding, atomic_numbers, 5.0)
    )["atom_feature"]
    inverted = core.get_output(
        **_inputs(-positions, embedding, atomic_numbers, 5.0)
    )["atom_feature"]
    assert torch.allclose(base, rotated, atol=1e-8, rtol=1e-8)
    assert torch.allclose(base, inverted, atol=1e-8, rtol=1e-8)

    data = _inputs(positions, embedding, atomic_numbers, 5.0)
    order = torch.tensor([2, 0, 4, 1, 3, 5])
    shuffled = dict(data)
    shuffled["Dij_sr"] = data["Dij_sr"][order]
    shuffled["vij_sr"] = data["vij_sr"][order]
    shuffled["idx_i_sr"] = data["idx_i_sr"][order]
    shuffled["idx_j_sr"] = data["idx_j_sr"][order]
    again = core.get_output(**shuffled)["atom_feature"]
    assert torch.allclose(base, again, atol=1e-8, rtol=1e-8)


def _stack():
    from enerzyme.models.ff import build_model

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    return build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"]),
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
        [[0.0, 0.0, 0.0], [1.15, 0.05, 0.0], [0.1, 1.2, 0.15], [1.9, 0.2, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 6])
    batch = torch.zeros(4, dtype=torch.long)
    base = _forward(model, positions, atomic_numbers, batch)
    assert base["E"].shape == (1,)
    assert base["Fa"].shape == (4, 3)
    assert base["atom_feature"].shape == (4, 4, 2)
    assert torch.isfinite(base["Fa"]).all()

    shift = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    moved = _forward(model, (positions.detach() + shift).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(base["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(5).double()
    rotated = _forward(
        model, (positions.detach() @ rotation).requires_grad_(True), atomic_numbers, batch
    )
    assert torch.allclose(base["E"], rotated["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-5, rtol=1e-5)

    inverted = _forward(model, (-positions.detach()).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], inverted["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(-base["Fa"], inverted["Fa"], atol=1e-5, rtol=1e-5)

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
    assert torch.allclose(analytical, numerical, atol=2e-4, rtol=1e-4)


def test_two_molecules_stay_separate_and_simple_readout_reads_the_last_layer():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = _stack().double().eval()
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.05],
            [0.05, 1.15, 0.1],
            [20.0, 0.0, 0.0],
            [21.2, -0.1, 0.2],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 6, 1])
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
    assert torch.allclose(together["Fa"][:3], left["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(together["Fa"][3:], right["Fa"], atol=1e-6, rtol=1e-6)

    simple = build_model(
        "camp",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "preset": "custom",
                    "max_v": 1,
                    "num_layers": 2,
                    "max_chebyshev_degree": 2,
                    "radial_mlp_hidden_layers": 0,
                    "atomic_numbers": [1, 6, 8],
                },
            },
            {"name": "SimpleReadout", "params": {"output_fields": ["Ea"], "head_type": "dense"}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
        build_params={"max_Za": 8, "dim_embedding": 2, "cutoff_sr": 5.0},
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


def test_rank_zero_reduces_to_scalar_messages():
    torch.manual_seed(7)
    core = _tiny(max_v=0, num_layers=1).double()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]], dtype=torch.float64)
    embedding = torch.randn(2, 2, dtype=torch.float64)
    atomic_numbers = torch.tensor([6, 1])
    out = core.get_output(**_inputs(positions, embedding, atomic_numbers, 5.0))
    assert out["atom_feature"].shape == (2, 2, 1)
    assert torch.isfinite(out["atom_feature"]).all()
    far = positions.clone()
    far[1, 0] = 12.0
    isolated = core.get_output(**_inputs(far, embedding, atomic_numbers, 5.0))
    assert torch.allclose(isolated["atom_feature"], torch.zeros_like(isolated["atom_feature"]))
