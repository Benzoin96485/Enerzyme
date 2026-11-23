"""FreeCG Core: registration, equivariance, energy, and conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from e3nn.o3 import Irrep

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "freecg_layers_example.yaml"


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


def _rotate_degree(features: torch.Tensor, rotation: torch.Tensor, degree: int) -> torch.Tensor:
    parity = -1 if degree % 2 else 1
    matrix = Irrep(degree, parity).D_from_matrix(rotation).to(dtype=features.dtype)
    return torch.einsum("naf,ad->ndf", features, matrix)


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
                "lmax": 2,
                "vertex_type": "edge",
                "vecnorm_type": "rms",
                "neighbor_embedding": True,
                "include_self": True,
                "num_groups": 2,
                "shuffle_scale": 1.5,
                "attention_enhancer": "linear",
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


def _attach(core, num_rbf: int):
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    radial = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=5.0, trainable=False)
    core._radial_holder = [radial]
    core.eval()
    return core


def test_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.freecg import (
        PUBLISHED_MD17_CORE_PARAMS,
        PUBLISHED_QM9_BUILD_PARAMS,
        PUBLISHED_QM9_CORE_PARAMS,
    )
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("freecg", "FreeCG", "free-cg", "free_cg"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "FreeCGCore"
        assert build_params["num_rbf"] == 32
        assert build_params["dim_embedding"] == 256
        assert build_params["cutoff_sr"] == 5.0
        assert layers[3]["params"]["num_groups"] == 8
        assert layers[3]["params"]["shuffle_scale"] == 1.5
        assert layers[3]["params"]["lmax"] == 2
    assert PUBLISHED_MD17_CORE_PARAMS["num_groups"] == 8
    assert PUBLISHED_MD17_CORE_PARAMS["attention_enhancer"] == "linear"
    assert PUBLISHED_QM9_CORE_PARAMS["vecnorm_type"] == "max_min"
    assert PUBLISHED_QM9_BUILD_PARAMS["dim_embedding"] == 512
    assert PUBLISHED_QM9_BUILD_PARAMS["dim_embedding"] % PUBLISHED_QM9_CORE_PARAMS["num_groups"] == 0

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    assert ff["architecture"] == "freecg"
    assert ff["layers"][3]["params"]["num_groups"] == 8
    model = build_model(
        ff["architecture"],
        layer_params=_tiny_layers(),
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "FreeCGCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    assert model.lmax == 2
    assert model.num_groups == 2
    assert model.channels_per_group == 4
    assert model.vis_layers[0].__class__.__name__ == "FreeCGInteraction"
    assert model.vis_layers[0].group_cg.n_paths == 4
    assert "atom_l2_feature" in model._output_fields
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "two_layer"


def test_unknown_width_lmax_and_groups_raise():
    from enerzyme.models.freecg import FreeCGCore

    try:
        FreeCGCore(dim_embedding=8, num_rbf=4, num_heads=2, num_layers=1, num_groups=3)
    except ValueError as exc:
        assert "num_groups" in str(exc)
    else:
        raise AssertionError("expected a bad group count to fail")
    try:
        FreeCGCore(dim_embedding=8, num_rbf=4, num_heads=2, num_layers=1, lmax=1)
    except ValueError as exc:
        assert "lmax" in str(exc)
    else:
        raise AssertionError("expected lmax 1 to fail")
    try:
        FreeCGCore(
            dim_embedding=8, num_rbf=4, num_heads=2, num_layers=1, attention_enhancer="softmax"
        )
    except ValueError as exc:
        assert "attention_enhancer" in str(exc)
    else:
        raise AssertionError("expected an unknown enhancer to fail")


def test_features_rotation_parity_translation_and_batch():
    from enerzyme.models.freecg import FreeCGCore

    torch.manual_seed(3)
    n, features, num_rbf = 4, 8, 4
    core = _attach(
        FreeCGCore(
            dim_embedding=features,
            num_rbf=num_rbf,
            num_layers=2,
            num_heads=2,
            num_groups=2,
            vertex_type="edge",
            vecnorm_type="rms",
            trainable_vecnorm=True,
            cutoff_sr=5.0,
            max_Za=10,
            attention_enhancer="max",
        ),
        num_rbf,
    )
    graph = _graph(n, features, num_rbf, seed=3)
    out = core.get_output(**graph)
    assert out["atom_feature"].shape == (n, features)
    assert out["atom_vector_feature"].shape == (n, 3, features)
    assert out["atom_l2_feature"].shape == (n, 5, features)

    rotation = _rotation(1)
    rotated = dict(graph)
    rotated["vij_sr"] = graph["vij_sr"] @ rotation
    rotated_out = core.get_output(**rotated)
    assert torch.allclose(out["atom_feature"], rotated_out["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _rotate_degree(out["atom_vector_feature"], rotation, 1),
        rotated_out["atom_vector_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        _rotate_degree(out["atom_l2_feature"], rotation, 2),
        rotated_out["atom_l2_feature"],
        atol=1e-5,
        rtol=1e-5,
    )

    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    mirrored = dict(graph)
    mirrored["vij_sr"] = graph["vij_sr"] @ mirror
    mirrored_out = core.get_output(**mirrored)
    assert torch.allclose(out["atom_feature"], mirrored_out["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _rotate_degree(out["atom_vector_feature"], mirror, 1),
        mirrored_out["atom_vector_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        _rotate_degree(out["atom_l2_feature"], mirror, 2),
        mirrored_out["atom_l2_feature"],
        atol=1e-5,
        rtol=1e-5,
    )

    order = torch.randperm(graph["idx_i_sr"].numel())
    permuted = dict(graph)
    for key in ("rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "cutoff_values_sr"):
        permuted[key] = graph[key][order]
    permuted_out = core.get_output(**permuted)
    assert torch.allclose(out["atom_feature"], permuted_out["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        out["atom_vector_feature"], permuted_out["atom_vector_feature"], atol=1e-5, rtol=1e-5
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


def test_stack_energy_force_dipole_and_image():
    from enerzyme.models.ff import build_model

    torch.manual_seed(7)
    model = build_model(
        "freecg",
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
    assert out["atom_l2_feature"].shape == (4, 5, 8)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shifted = (pos.detach() + torch.tensor([2.0, -1.0, 0.4])).requires_grad_(True)
    moved = _forward(model, shifted, za, batch)
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["M2"], moved["M2"], atol=1e-6, rtol=1e-6)

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
    assert torch.allclose(out["atom_l2_feature"], inverted["atom_l2_feature"], atol=1e-6, rtol=1e-6)

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
