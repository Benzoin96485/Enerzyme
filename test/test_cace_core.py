"""CACE Core: presets, invariance, messages, and energy conservation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "cace_layers_example.yaml"


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


def _inputs(positions, embedding, cutoff, rbf=None, window=None):
    idx_i, idx_j = _edges(positions, cutoff)
    if idx_i.numel() == 0:
        distance = positions.new_zeros(0)
        vectors = positions.new_zeros(0, 3)
    else:
        vectors = positions[idx_j] - positions[idx_i]
        distance = torch.linalg.vector_norm(vectors, dim=-1)
    if rbf is None:
        rbf = positions.new_ones(idx_i.shape[0], 1)
    data = {
        "atom_embedding": embedding,
        "rbf": rbf,
        "Dij_sr": distance,
        "vij_sr": vectors,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }
    if window is not None:
        data["cutoff_values_sr"] = window
    return data


def _tiny(**overrides):
    from enerzyme.models.cace import CACECore

    params = dict(
        dim_embedding=2,
        cutoff_sr=5.0,
        num_rbf=3,
        preset="custom",
        lmax=2,
        max_nu=2,
        n_radial=2,
        num_message_passing=0,
    )
    params.update(overrides)
    return CACECore(**params)


def test_registration_presets_and_yaml():
    from enerzyme.models.cace import PAPER_PRESETS
    from enerzyme.models.cace.readout import CACEEnergyReadout
    from enerzyme.models.ff import build_model, get_ff_core

    for name in ("cace", "CACE"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "CACECore"
        assert build_params["cutoff_sr"] == 5.5
        assert build_params["dim_embedding"] == 3
        assert build_params["num_rbf"] == 6
        assert layers[1]["name"] == "BesselRBF"
        assert layers[1]["params"]["apply_cutoff_fn"] is False
        assert layers[2]["name"] == "RandomAtomEmbedding"
        assert layers[3]["params"]["preset"] == "water"
        assert layers[4]["name"] == "CACEEnergyReadout"

    water = PAPER_PRESETS["water"]
    assert water["lmax"] == 3 and water["max_nu"] == 3 and water["n_radial"] == 12
    assert water["num_message_passing"] == 1 and water["messages"] == "b"
    assert water["avg_num_neighbors"] == 1.0 and water["envelope_p"] == 6
    assert water["dim_embedding"] == 3 and water["cutoff"] == 5.5 and water["num_rbf"] == 6
    assert water["readout_hidden"] == (32, 16)
    assert water["shared_channels"] is False and water["shared_l"] is False
    local = PAPER_PRESETS["water-local"]
    assert local["num_message_passing"] == 0
    ethanol = PAPER_PRESETS["ethanol"]
    assert ethanol["messages"] == "full" and ethanol["envelope_p"] == 5
    assert ethanol["cutoff"] == 4.0 and ethanol["avg_num_neighbors"] == 10.0
    bpa = PAPER_PRESETS["bpa"]
    assert bpa["num_rbf"] == 8 and bpa["envelope_p"] == 2 and bpa["cutoff"] == 6.0
    hea = PAPER_PRESETS["hea"]
    assert hea["n_radial"] == 8 and hea["dim_embedding"] == 4
    assert hea["readout_hidden"] == (24, 12) and hea["cutoff"] == 4.5

    published = _tiny(
        dim_embedding=3,
        cutoff_sr=5.5,
        num_rbf=6,
        preset="water",
        lmax=None,
        max_nu=None,
        n_radial=None,
        num_message_passing=None,
    )
    assert published.n_invariants == 6
    assert published.dim_feature_out == 2 * 12 * 6 * 9
    assert published.messages == frozenset({"b"})
    hea_core = _tiny(
        dim_embedding=4,
        cutoff_sr=4.5,
        num_rbf=6,
        preset="hea",
        lmax=None,
        max_nu=None,
        n_radial=None,
        num_message_passing=None,
    )
    assert hea_core.dim_feature_out == 2 * 8 * 6 * 16
    assert hea_core.readout_hidden == (24, 12)

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"]),
        verbose=0,
    )
    assert model.__class__.__name__ == "CACECore"
    assert model.lmax == 2 and model.max_nu == 3 and model.n_radial == 2
    assert model.num_message_passing == 1
    assert model.dim_feature_out == 64
    assert model.feature_irreps == "64x0e"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, CACEEnergyReadout))
    assert readout.linear is not None and readout.mlp is not None

    try:
        _tiny(preset="nope")
    except ValueError as exc:
        assert "preset" in str(exc).lower()
    else:
        raise AssertionError("unknown preset should fail")


def test_local_features_match_the_pair_contraction_and_ignore_distant_edges():
    torch.manual_seed(0)
    core = _tiny(
        dim_embedding=1,
        num_rbf=1,
        n_radial=1,
        lmax=2,
        max_nu=2,
        num_message_passing=0,
        preset=None,
    ).double()
    with torch.no_grad():
        core.mix.weight.fill_(1.0)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    embedding = torch.ones(3, 1, dtype=torch.float64)
    window = torch.ones(6, dtype=torch.float64)
    out = core.get_output(**_inputs(positions, embedding, 5.0, window=window))
    # Three directed edges touch the center from the two neighbors, and each
    # neighbor sees the center plus the other neighbor.
    assert out["atom_feature"].shape == (3, 1 * 3 * 1)
    center = out["atom_feature"][0]
    # Identity mix, unit edge type, unit radial, window 1.
    # ν=2, L=1: 2 + 2 (u·v) = 2 for perpendicular unit vectors.
    assert torch.allclose(center[1], torch.tensor(2.0, dtype=torch.float64), atol=1e-6)

    far = positions.clone()
    far[2] = torch.tensor([0.0, 8.0, 0.0], dtype=torch.float64)
    near = core.get_output(**_inputs(positions[:2], embedding[:2], 5.0))
    widened = core.get_output(**_inputs(far, embedding, 5.0))
    assert torch.allclose(widened["atom_feature"][:2], near["atom_feature"], atol=1e-6, rtol=1e-6)


def test_isolated_atoms_are_zero_and_embeddings_enter_only_through_edges():
    torch.manual_seed(1)
    core = _tiny().eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3]])
    embedding = torch.randn(3, 2)
    out = core.get_output(**_inputs(positions, embedding, 5.0, rbf=torch.rand(6, 3)))
    assert out["atom_feature"].shape == (3, core.dim_feature_out)
    assert torch.isfinite(out["atom_feature"]).all()

    alone = core.get_output(**_inputs(positions[:1], embedding[:1], 5.0, rbf=torch.rand(0, 3)))
    assert torch.count_nonzero(alone["atom_feature"]) == 0
    other = core.get_output(
        **_inputs(positions[:1], embedding[:1] + 4.0, 5.0, rbf=torch.rand(0, 3))
    )
    assert torch.count_nonzero(other["atom_feature"]) == 0

    swapped = core.get_output(
        **_inputs(positions, torch.randn(3, 2), 5.0, rbf=torch.rand(6, 3))
    )
    assert not torch.allclose(swapped["atom_feature"], out["atom_feature"], atol=1e-5)


def test_molecules_do_not_exchange_messages():
    torch.manual_seed(2)
    core = _tiny(num_message_passing=1, messages="full", preset=None, lmax=2, max_nu=2).eval()
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0]])
    right_pos = torch.tensor([[0.0, 0.0, 0.0], [1.3, -0.2, 0.2]])
    left_h = torch.randn(2, 2)
    right_h = torch.randn(2, 2)
    left = core.get_output(**_inputs(left_pos, left_h, 5.0, rbf=torch.rand(2, 3)))["atom_feature"]
    right = core.get_output(**_inputs(right_pos, right_h, 5.0, rbf=torch.rand(2, 3)))["atom_feature"]
    joined_pos = torch.cat([left_pos, right_pos + torch.tensor([25.0, -4.0, 3.0])], dim=0)
    joined_h = torch.cat([left_h, right_h])
    n_left = _edges(left_pos, 5.0)[0].numel()
    n_right = _edges(right_pos, 5.0)[0].numel()
    rbf = torch.cat([torch.rand(n_left, 3), torch.rand(n_right, 3)], dim=0)
    # Rebuild with the same per-molecule bases by evaluating the joined graph
    # from scratch with one rbf; isolation is about geometry, so use a fresh
    # shared evaluation of each piece and of the joined system with consistent rbf.
    del rbf
    joined = core.get_output(
        **_inputs(joined_pos, joined_h, 5.0, rbf=torch.ones(_edges(joined_pos, 5.0)[0].numel(), 3))
    )["atom_feature"]
    left_ones = core.get_output(
        **_inputs(left_pos, left_h, 5.0, rbf=torch.ones(n_left, 3))
    )["atom_feature"]
    right_ones = core.get_output(
        **_inputs(right_pos, right_h, 5.0, rbf=torch.ones(n_right, 3))
    )["atom_feature"]
    assert torch.allclose(joined[:2], left_ones, atol=1e-5, rtol=1e-5)
    assert torch.allclose(joined[2:], right_ones, atol=1e-5, rtol=1e-5)
    assert left.shape == left_ones.shape and right.shape == right_ones.shape


def test_scalar_message_changes_the_second_block():
    torch.manual_seed(3)
    core = _tiny(num_message_passing=1, messages="b", avg_num_neighbors=1.0).eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.15, 0.0], [-0.2, 1.15, 0.25]])
    embedding = torch.randn(3, 2)
    rbf = torch.rand(6, 3)
    kwargs = _inputs(positions, embedding, 5.0, rbf=rbf)
    baseline = core.get_output(**kwargs)["atom_feature"]
    width = baseline.shape[-1] // 2
    with torch.no_grad():
        core.layers[0].gate.h.weight.zero_()
        core.layers[0].gate.h.bias.zero_()
    silenced = core.get_output(**kwargs)["atom_feature"]
    assert torch.allclose(silenced[:, width:], torch.zeros_like(silenced[:, width:]))
    assert not torch.allclose(baseline[:, width:], silenced[:, width:], atol=1e-6)
    with torch.no_grad():
        core.layers[0].gate.h.bias.fill_(1.0)
    repeated = core.get_output(**kwargs)["atom_feature"]
    assert torch.isfinite(repeated).all()
    assert not torch.allclose(repeated[:, width:], silenced[:, width:], atol=1e-6)


def test_features_are_invariant_under_rotation_translation_and_inversion():
    torch.manual_seed(4)
    core = _tiny(num_message_passing=1, messages="full", preset=None, lmax=2, max_nu=3).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.2, 0.4], [2.0, 0.3, 1.1]]
    )
    embedding = torch.randn(4, 2)
    n_edges = _edges(positions, 5.0)[0].numel()
    rbf = torch.rand(n_edges, 3)

    def features(coords):
        # The radial basis is recomputed by the caller; pass ones so geometry
        # is the only change. A learned Bessel would move with distance, which
        # a pure rotation does not.
        edges = _edges(coords, 5.0)[0].numel()
        return core.get_output(
            **_inputs(coords, embedding, 5.0, rbf=torch.ones(edges, 3))
        )["atom_feature"]

    base = features(positions)
    assert torch.allclose(base, features(positions + torch.tensor([2.0, -1.0, 0.4])))
    rotation = _rotation(5)
    assert torch.allclose(base, features(positions @ rotation), atol=1e-5, rtol=1e-5)
    assert torch.allclose(base, features(-positions), atol=1e-5, rtol=1e-5)
    assert rbf.shape[0] == n_edges


def _stack():
    from enerzyme.models.ff import build_model

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    return build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], max_Za=12),
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
    torch.manual_seed(6)
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
    assert base["atom_feature"].shape == (4, 64)
    assert torch.isfinite(base["Fa"]).all()

    shift = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    moved = _forward(model, (positions.detach() + shift).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(base["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(7).double()
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


def test_two_molecules_keep_separate_energies_and_simple_readout_fits():
    from enerzyme.models.ff import build_model

    torch.manual_seed(8)
    model = _stack().double().eval()
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
    assert torch.allclose(together["Fa"][:3], left["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(together["Fa"][3:], right["Fa"], atol=1e-6, rtol=1e-6)

    simple = build_model(
        "cace",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "BesselRBF", "params": {"trainable": False, "apply_cutoff_fn": False}},
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "preset": "custom",
                    "num_message_passing": 0,
                    "lmax": 1,
                    "max_nu": 2,
                    "n_radial": 2,
                },
            },
            {"name": "SimpleReadout", "params": {"output_fields": ["Ea"], "head_type": "dense"}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
        build_params={"max_Za": 8, "dim_embedding": 2, "num_rbf": 3, "cutoff_sr": 5.0},
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
