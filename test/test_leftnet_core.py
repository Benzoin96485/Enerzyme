"""LEFTNet Core: registration, equivariance, and the external energy head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "leftnet_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotation(seed: int = 0, improper: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    if improper:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _core(**overrides):
    from enerzyme.models.leftnet import LEFTNetCore

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        cutoff_sr=5.0,
        num_layers=2,
        fte_scalarization="l2",
    )
    params.update(overrides)
    return LEFTNetCore(**params).double().eval()


def _call(core, pos, embedding, rbf, idx_i, idx_j, batch=None):
    vij = pos[idx_j] - pos[idx_i]
    return core.get_output(
        atom_embedding=embedding,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=torch.linalg.vector_norm(vij, dim=-1),
        Ra=pos,
        batch_seg=batch,
    )


def _molecule(n, f, n_rbf, seed):
    torch.manual_seed(seed)
    pos = torch.randn(n, 3, dtype=torch.float64)
    pos = pos - pos.mean(dim=0)
    return {
        "pos": pos,
        "embedding": torch.randn(n, f, dtype=torch.float64),
        "idx": _edges(n),
        "rbf": torch.randn(n * (n - 1), n_rbf, dtype=torch.float64),
    }


def test_leftnet_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("leftnet", "LEFTNet"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "LEFTNetCore"
        assert build_params["num_rbf"] == 32
        assert build_params["cutoff_sr"] == 5.0
        assert layers[1]["params"]["alpha"] == 1.0
        assert layers[3]["params"]["fte_scalarization"] == "l2"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "LEFTNetCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "dense"
    assert readout.dim_feature_in == 8


def test_leftnet_rejects_bad_settings():
    from enerzyme.models.leftnet import LEFTNetCore

    with pytest.raises(ValueError):
        LEFTNetCore(dim_embedding=3)
    with pytest.raises(ValueError):
        LEFTNetCore(num_layers=0)
    with pytest.raises(ValueError):
        LEFTNetCore(num_rbf=0)
    with pytest.raises(ValueError):
        LEFTNetCore(cutoff_sr=0.0)
    with pytest.raises(ValueError):
        LEFTNetCore(fte_scalarization="l1")


@pytest.mark.parametrize("scalarization", ["l2", "frame"])
@pytest.mark.parametrize("improper", [False, True])
def test_leftnet_rotation_and_translation(scalarization, improper):
    core = _core(fte_scalarization=scalarization)
    graph = _molecule(4, 8, 4, seed=1)
    idx_i, idx_j = graph["idx"]
    base = _call(core, graph["pos"], graph["embedding"], graph["rbf"], idx_i, idx_j)
    assert base["atom_feature"].shape == (4, 8)
    assert base["atom_vector_feature"].shape == (4, 3, 8)

    shift = torch.tensor([3.0, -1.5, 0.4], dtype=torch.float64)
    moved = _call(core, graph["pos"] + shift, graph["embedding"], graph["rbf"], idx_i, idx_j)
    assert torch.allclose(base["atom_feature"], moved["atom_feature"], atol=1e-10)
    assert torch.allclose(base["atom_vector_feature"], moved["atom_vector_feature"], atol=1e-10)

    rotation = _rotation(2, improper=improper)
    rotated = _call(
        core, graph["pos"] @ rotation, graph["embedding"], graph["rbf"], idx_i, idx_j
    )
    assert torch.allclose(base["atom_feature"], rotated["atom_feature"], atol=1e-8, rtol=1e-6)
    expected = torch.einsum("ncf,cd->ndf", base["atom_vector_feature"], rotation)
    assert torch.allclose(expected, rotated["atom_vector_feature"], atol=1e-8, rtol=1e-6)


def test_leftnet_molecules_do_not_share_a_center():
    core = _core(num_layers=1)
    graph = _molecule(3, 8, 4, seed=3)
    idx_i, idx_j = graph["idx"]
    single = _call(
        core,
        graph["pos"],
        graph["embedding"],
        graph["rbf"],
        idx_i,
        idx_j,
        batch=torch.zeros(3, dtype=torch.long),
    )
    offset = torch.tensor([9.0, -2.0, 1.0], dtype=torch.float64)
    pos = torch.cat((graph["pos"], graph["pos"] + offset), dim=0)
    embedding = torch.cat((graph["embedding"], graph["embedding"]), dim=0)
    both = _call(
        core,
        pos,
        embedding,
        torch.cat((graph["rbf"], graph["rbf"]), dim=0),
        torch.cat((idx_i, idx_i + 3)),
        torch.cat((idx_j, idx_j + 3)),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    assert torch.allclose(both["atom_feature"][:3], single["atom_feature"], atol=1e-10)
    assert torch.allclose(both["atom_feature"][3:], single["atom_feature"], atol=1e-10)
    assert torch.allclose(both["atom_vector_feature"][:3], single["atom_vector_feature"], atol=1e-10)


def test_leftnet_isolated_atom_has_zero_vectors():
    core = _core(num_layers=2, fte_scalarization="frame")
    pos = torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.float64)
    embedding = torch.randn(1, 8, dtype=torch.float64)
    empty = torch.zeros(0, dtype=torch.long)
    out = core.get_output(
        atom_embedding=embedding,
        rbf=torch.zeros(0, 4, dtype=torch.float64),
        idx_i_sr=empty,
        idx_j_sr=empty,
        vij_sr=torch.zeros(0, 3, dtype=torch.float64),
        Dij_sr=torch.zeros(0, dtype=torch.float64),
        Ra=pos,
    )
    assert torch.isfinite(out["atom_feature"]).all()
    assert torch.count_nonzero(out["atom_vector_feature"]) == 0
    out["atom_feature"].sum().backward()


def test_leftnet_backward_reaches_positions():
    core = _core(num_layers=1)
    graph = _molecule(3, 8, 4, seed=4)
    pos = graph["pos"].detach().requires_grad_(True)
    out = _call(core, pos, graph["embedding"], graph["rbf"], *graph["idx"])
    out["atom_feature"].sum().backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    assert pos.grad.abs().sum() > 0


def test_leftnet_stack_energy_and_force():
    from enerzyme.models.ff import build_model

    torch.manual_seed(5)
    model = build_model(
        "leftnet",
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "ExpNormalSmearing", "params": {"alpha": 1.0, "trainable": False}},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"num_layers": 1, "fte_scalarization": "l2"}},
            {"name": "SimpleReadout", "params": {"output_fields": ["Ea"], "head_type": "dense"}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
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
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.3, 0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    idx_i, idx_j = _edges(3)
    out = model(
        {
            "Ra": pos,
            "Za": torch.tensor([6, 1, 1]),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(3, dtype=torch.long),
        }
    )
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (3, 3)
    assert torch.isfinite(out["Fa"]).all()
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
            kwargs = {
                "Za": torch.tensor([6, 1, 1]),
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": torch.zeros(3, dtype=torch.long),
            }
            e_plus = model({**kwargs, "Ra": plus})["E"][0]
            e_minus = model({**kwargs, "Ra": minus})["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)
