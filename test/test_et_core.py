"""ET Core: registration, equivariance, energy, and conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "et_layers_example.yaml"


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


def _rotate_channels(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ncf,cd->ndf", vector, rotation)


def _graph(n: int, features: int, num_rbf: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3, generator=generator)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    return {
        "atom_embedding": torch.randn(n, features, generator=generator),
        "rbf": torch.rand(idx_i.numel(), num_rbf, generator=generator),
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "vij_sr": vij,
        "Dij_sr": dist,
        "cutoff_values_sr": torch.rand(idx_i.numel(), generator=generator).clamp(0.2, 1),
        "Za": torch.randint(1, 8, (n,), generator=generator),
    }


def _tiny_layers(dipole: bool = False):
    layers = [
        {"name": "RangeSeparation"},
        {"name": "ExpNormalSmearing", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_layers": 2,
                "num_heads": 2,
                "neighbor_embedding": True,
                "vector_cutoff": True,
                "include_self": True,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "two_layer",
                "pyramidal": True,
                "activation_fn": "swish",
                "activation_params": {
                    "dim_feature": 1,
                    "initial_alpha": 1.0,
                    "initial_beta": 1.0,
                    "learnable": False,
                },
                "keep_feature": False,
            },
        },
    ]
    if dipole:
        layers.append(
            {"name": "EquivariantDipole", "params": {"n_layers": 1, "correct_charges": True}}
        )
    layers.extend([{"name": "EnergyReduce"}, {"name": "Force"}])
    return layers


def _forward(model, pos, za, batch_seg, offsets=None, edges=None):
    n = pos.shape[0]
    if edges is None:
        idx_i, idx_j = _edges(n)
    else:
        idx_i, idx_j = edges
    net = {
        "Ra": pos,
        "Za": za,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "batch_seg": batch_seg,
    }
    if offsets is not None:
        net["offsets"] = offsets
    return model(net)


def test_et_registration_and_yaml_build():
    from enerzyme.models.et import PUBLISHED_QM9_CORE_PARAMS
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("et", "ET", "torchmd-et", "torchmd_et", "equivariant-transformer"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "ETCore"
        assert build_params["num_rbf"] == 32
        assert build_params["cutoff_sr"] == 5.0
        assert build_params["cutoff_fn"] == "cosine"
        assert layers[1]["name"] == "ExpNormalSmearing"
        assert layers[3]["params"]["num_layers"] == 6
        assert layers[3]["params"]["vector_cutoff"] is True
    assert PUBLISHED_QM9_CORE_PARAMS["num_layers"] == 8

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "ETCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    assert model.num_heads == 8
    assert model._radial_holder[0].__class__.__name__ == "ExpNormalSmearing"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "two_layer"
    assert readout.head[0].dim_feature_out == 4


def test_unknown_width_and_cutoff_mismatch_raise():
    from enerzyme.models.et import ETCore
    from enerzyme.models.ff import build_model

    try:
        ETCore(dim_embedding=6, num_rbf=3, num_heads=4)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected a bad head count to fail")

    try:
        build_model(
            "et",
            layer_params=[
                {"name": "ExpNormalSmearing", "params": {"cutoff_sr": 5.0}},
                {"name": "Core", "params": {"num_heads": 2}},
            ],
            build_params={
                "dim_embedding": 4,
                "num_rbf": 4,
                "cutoff_sr": 4.5,
                "max_Za": 8,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "cutoff" in str(exc)
    else:
        raise AssertionError("expected a radial cutoff mismatch to fail")


def test_features_rotation_parity_translation_and_batch():
    from enerzyme.models.et import ETCore
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    torch.manual_seed(3)
    n, features, num_rbf = 4, 8, 4
    core = ETCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_layers=2,
        num_heads=2,
        cutoff_sr=5.0,
        max_Za=10,
    )
    radial = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=5.0, trainable=False)
    core._radial_holder = [radial]
    core.eval()
    graph = _graph(n, features, num_rbf, seed=3)
    out = core.get_output(**graph)
    assert out["atom_feature"].shape == (n, features)
    assert out["atom_vector_feature"].shape == (n, 3, features)

    rotation = _rotation(1)
    rotated = dict(graph)
    rotated["vij_sr"] = graph["vij_sr"] @ rotation
    rotated_out = core.get_output(**rotated)
    assert torch.allclose(out["atom_feature"], rotated_out["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _rotate_channels(out["atom_vector_feature"], rotation),
        rotated_out["atom_vector_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
    reflected = dict(graph)
    reflected["vij_sr"] = -graph["vij_sr"]
    reflected_out = core.get_output(**reflected)
    assert torch.allclose(out["atom_feature"], reflected_out["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        -out["atom_vector_feature"], reflected_out["atom_vector_feature"], atol=1e-5, rtol=1e-5
    )
    shifted = dict(graph)
    shifted["vij_sr"] = graph["vij_sr"]
    assert torch.allclose(
        out["atom_feature"], core.get_output(**shifted)["atom_feature"], atol=1e-6
    )

    order = torch.randperm(graph["idx_i_sr"].numel())
    permuted = dict(graph)
    for key in ("rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "cutoff_values_sr"):
        permuted[key] = graph[key][order]
    assert torch.allclose(
        out["atom_feature"], core.get_output(**permuted)["atom_feature"], atol=1e-5, rtol=1e-5
    )

    left = _graph(3, features, num_rbf, seed=4)
    right = _graph(2, features, num_rbf, seed=5)
    separate_l = core.get_output(**left)["atom_feature"]
    separate_r = core.get_output(**right)["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    joined = {
        "atom_embedding": torch.cat((left["atom_embedding"], right["atom_embedding"])),
        "rbf": torch.cat((left["rbf"], right["rbf"])),
        "idx_i_sr": torch.cat((left["idx_i_sr"], right["idx_i_sr"] + n_left)),
        "idx_j_sr": torch.cat((left["idx_j_sr"], right["idx_j_sr"] + n_left)),
        "vij_sr": torch.cat((left["vij_sr"], right["vij_sr"])),
        "Dij_sr": torch.cat((left["Dij_sr"], right["Dij_sr"])),
        "cutoff_values_sr": torch.cat((left["cutoff_values_sr"], right["cutoff_values_sr"])),
        "Za": torch.cat((left["Za"], right["Za"])),
    }
    both = core.get_output(**joined)["atom_feature"]
    assert torch.allclose(both[:n_left], separate_l, atol=1e-5, rtol=1e-5)
    assert torch.allclose(both[n_left:], separate_r, atol=1e-5, rtol=1e-5)


def test_neighbor_embedding_self_pairs_and_isolated_atom():
    from enerzyme.models.et import ETCore
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    torch.manual_seed(6)
    n, features, num_rbf = 3, 8, 4
    radial = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=5.0, trainable=False)
    graph = _graph(n, features, num_rbf, seed=6)
    with_neighbors = ETCore(
        dim_embedding=features, num_rbf=num_rbf, num_layers=2, num_heads=2, max_Za=10
    )
    with_neighbors._radial_holder = [radial]
    without = ETCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_layers=2,
        num_heads=2,
        max_Za=10,
        neighbor_embedding=False,
    )
    without._radial_holder = [radial]
    without.load_state_dict(with_neighbors.state_dict(), strict=False)
    plain = with_neighbors.get_output(**graph)["atom_feature"]
    ablated = without.get_output(**graph)["atom_feature"]
    assert not torch.allclose(plain, ablated, atol=1e-5, rtol=1e-5)

    other_z = dict(graph)
    other_z["Za"] = torch.tensor([2, 7, 3])
    assert not torch.allclose(
        plain, with_neighbors.get_output(**other_z)["atom_feature"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        ablated, without.get_output(**other_z)["atom_feature"], atol=1e-6, rtol=1e-6
    )

    no_self = ETCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_layers=2,
        num_heads=2,
        max_Za=10,
        include_self=False,
        neighbor_embedding=False,
    )
    no_self.load_state_dict(without.state_dict())
    assert not torch.allclose(
        without.get_output(**graph)["atom_feature"],
        no_self.get_output(**graph)["atom_feature"],
        atol=1e-5,
        rtol=1e-5,
    )

    alone = with_neighbors.get_output(
        atom_embedding=torch.randn(1, features),
        rbf=torch.zeros(0, num_rbf),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
        Za=torch.tensor([6]),
    )
    assert alone["atom_feature"].shape == (1, features)
    assert torch.count_nonzero(alone["atom_vector_feature"]) == 0
    assert torch.isfinite(alone["atom_feature"]).all()


def test_stack_energy_force_dipole_and_image():
    from enerzyme.models.ff import build_model

    torch.manual_seed(7)
    model = build_model(
        "et",
        layer_params=_tiny_layers(dipole=True),
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    ).double()
    model.eval()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.15, 0.05],
            [0.1, 0.85, 0.2],
            [0.2, 0.1, 0.95],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([8, 1, 6, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    out = _forward(model, pos, za, batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert out["M2"].shape == (1, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shifted = (pos.detach() + torch.tensor([2.0, -1.0, 0.4])).requires_grad_(True)
    moved = _forward(model, shifted, za, batch)
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(9).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, za, batch)
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["M2"] @ rotation, rotated["M2"], atol=1e-6, rtol=1e-6)

    inverted = _forward(model, (-pos.detach()).requires_grad_(True), za, batch)
    assert torch.allclose(out["E"], inverted["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(-out["Fa"], inverted["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(-out["M2"], inverted["M2"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), za, batch)["E"][0]
            e_minus = _forward(model, minus.requires_grad_(True), za, batch)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)

    far = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    species = torch.tensor([6, 8], dtype=torch.long)
    pair = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
    )
    isolated = _forward(model, far, species, torch.zeros(2, dtype=torch.long), edges=pair)
    image = _forward(
        model,
        far,
        species,
        torch.zeros(2, dtype=torch.long),
        edges=pair,
        offsets=torch.tensor([[-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert torch.isfinite(isolated["E"]).all()
    assert not torch.allclose(isolated["E"], image["E"], atol=1e-6, rtol=1e-6)
    alone = _forward(
        model,
        far[:1].detach().requires_grad_(True),
        species[:1],
        torch.zeros(1, dtype=torch.long),
        edges=(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)),
    )
    assert torch.allclose(alone["Fa"], torch.zeros_like(alone["Fa"]), atol=1e-8)
