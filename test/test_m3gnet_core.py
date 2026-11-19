"""M3GNet Core: registration, invariance, three-body cutoff, conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "m3gnet_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotate(pos: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return pos @ q


def _basis(dist: torch.Tensor, max_l: int = 2, num_rbf: int = 2, cutoff: float = 5.0):
    from enerzyme.models.m3gnet.basis import SphericalBesselRBFLayer

    layer = SphericalBesselRBFLayer(num_rbf=num_rbf, cutoff_sr=cutoff, max_l=max_l)
    return layer._get_rbf(dist)


def _tiny(**kwargs):
    from enerzyme.models.m3gnet import M3GNetCore

    params = dict(
        dim_embedding=8,
        num_rbf=2,
        cutoff_sr=5.0,
        max_l=2,
        num_blocks=2,
        threebody_cutoff=4.0,
        units=8,
        n_conv_hidden=1,
    )
    params.update(kwargs)
    return M3GNetCore(**params)


def _inputs(n=4, seed=0, max_l=2, num_rbf=2):
    torch.manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1).clamp(min=1e-6)
    return {
        "atom_embedding": torch.randn(n, 8),
        "rbf": _basis(dist, max_l=max_l, num_rbf=num_rbf),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "pos": pos,
    }


def _forward(core, inp):
    return core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )["atom_feature"]


def test_registration_and_example_yaml():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, defaults, layers = get_ff_core("m3gnet")
    assert core_cls.__name__ == "M3GNetCore"
    assert defaults["cutoff_sr"] == 5.0
    assert defaults["threebody_cutoff"] == 4.0
    assert defaults["num_rbf"] == 3
    assert any(layer["name"] == "SphericalBesselRBF" for layer in layers)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "M3GNetCore"
    assert model.degree == 9
    assert model.threebody_cutoff == 4.0
    assert model.dim_feature_out == 64
    assert model.feature_irreps == "64x0e"
    assert model.blocks[0].bond_update.value_net[0].bias is None


def test_feature_shape_and_isolated_atom():
    core = _tiny()
    feat = _forward(core, _inputs())
    assert feat.shape == (4, 8)
    assert torch.isfinite(feat).all()

    isolated = core.get_output(
        atom_embedding=torch.randn(1, 8),
        rbf=torch.zeros(0, 4),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )["atom_feature"]
    assert isolated.shape == (1, 8)
    # No edges: the state is the species embedding, before the external head.
    emb = torch.randn(1, 8)
    same = core.get_output(
        atom_embedding=emb,
        rbf=torch.zeros(0, 4),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )["atom_feature"]
    assert_allclose(same.detach().numpy(), emb.detach().numpy(), atol=1e-6)


def test_threebody_cutoff_changes_features_and_rejects_a_longer_cutoff():
    torch.manual_seed(0)
    wide = _tiny(threebody_cutoff=4.0)
    narrow = _tiny(threebody_cutoff=0.2)
    narrow.load_state_dict(wide.state_dict())
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.3, 1.1, 0.0], [0.4, 0.2, 1.0]]
    )
    idx_i, idx_j = _edges(4)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    inp = {
        "atom_embedding": torch.randn(4, 8),
        "rbf": _basis(dist),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }
    assert dist.min() < 4.0
    assert dist.min() > 0.2
    left = _forward(wide, inp)
    right = _forward(narrow, inp)
    assert not torch.allclose(left, right, atol=1e-5, rtol=1e-5)
    try:
        _tiny(threebody_cutoff=6.0)
    except ValueError as exc:
        assert "threebody_cutoff" in str(exc)
    else:
        raise AssertionError("threebody_cutoff above cutoff_sr must be rejected")


def test_rotation_translation_inversion_and_edge_order():
    torch.manual_seed(1)
    core = _tiny()
    core.eval()
    inp = _inputs(seed=1)
    base = _forward(core, inp)
    idx_i, idx_j = inp["idx_i_sr"], inp["idx_j_sr"]
    embedding = inp["atom_embedding"]

    def _from_pos(pos, sign=1.0):
        vij = sign * (pos[idx_j] - pos[idx_i])
        dist = vij.norm(dim=-1).clamp(min=1e-6)
        return core.get_output(
            atom_embedding=embedding,
            rbf=_basis(dist),
            Dij_sr=dist,
            vij_sr=vij,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
        )["atom_feature"]

    rotated = _from_pos(_rotate(inp["pos"]))
    assert_allclose(base.detach().numpy(), rotated.detach().numpy(), atol=1e-5, rtol=1e-5)
    translated = _from_pos(inp["pos"] + torch.tensor([3.0, -2.0, 0.4]))
    assert_allclose(base.detach().numpy(), translated.detach().numpy(), atol=1e-5, rtol=1e-5)
    inverted = _from_pos(inp["pos"], sign=-1.0)
    assert_allclose(base.detach().numpy(), inverted.detach().numpy(), atol=1e-5, rtol=1e-5)

    perm = torch.randperm(idx_i.numel())
    vij = inp["vij_sr"][perm]
    shuffled = core.get_output(
        atom_embedding=embedding,
        rbf=inp["rbf"][perm],
        Dij_sr=inp["Dij_sr"][perm],
        vij_sr=vij,
        idx_i_sr=idx_i[perm],
        idx_j_sr=idx_j[perm],
    )["atom_feature"]
    assert_allclose(base.detach().numpy(), shuffled.detach().numpy(), atol=1e-5, rtol=1e-5)


def test_disjoint_molecules_do_not_mix():
    torch.manual_seed(2)
    core = _tiny()
    left = _inputs(n=3, seed=2)
    right = _inputs(n=3, seed=3)
    feat_l = _forward(core, left)
    feat_r = _forward(core, right)
    merged = {
        "atom_embedding": torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
    }
    feat = _forward(core, merged)
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)


def test_layer_stack_energy_forces_and_wrong_basis():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "SphericalBesselRBF"},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {"num_blocks": 2, "n_conv_hidden": 1},
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "gated",
                "num_hidden_layers": 1,
                "activate_last": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "m3gnet",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 2,
            "max_l": 2,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "threebody_cutoff": 4.0,
        },
        verbose=0,
    )
    n = 4
    idx_i, idx_j = _edges(n)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.1, 0.9, 0.2], [0.7, 0.6, 0.4]],
        requires_grad=True,
    )
    out = model(
        {
            "Ra": pos,
            "Za": torch.tensor([1, 6, 8, 1]),
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    force = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-5, rtol=1e-5)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)

    dimer = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], requires_grad=True)
    out_d = model(
        {
            "Ra": dimer,
            "Za": torch.tensor([1, 8]),
            "batch_seg": torch.zeros(2, dtype=torch.long),
            "idx_i": torch.tensor([0, 1]),
            "idx_j": torch.tensor([1, 0]),
        }
    )
    bond = dimer[1] - dimer[0]
    cross = torch.linalg.cross(out_d["Fa"][0], bond)
    assert_allclose(cross.detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)

    alone = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
    out_a = model(
        {
            "Ra": alone,
            "Za": torch.tensor([6]),
            "batch_seg": torch.zeros(1, dtype=torch.long),
            "idx_i": torch.zeros(0, dtype=torch.long),
            "idx_j": torch.zeros(0, dtype=torch.long),
        }
    )
    assert torch.isfinite(out_a["E"]).all()
    assert_allclose(out_a["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-6)

    wrong = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"flavor": "dimenet", "trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_blocks": 1, "max_l": 2}},
        {"name": "EnergyReduce"},
    ]
    try:
        build_model(
            "m3gnet",
            layer_params=wrong,
            build_params={
                "dim_embedding": 4,
                "num_rbf": 3,
                "max_l": 2,
                "max_Za": 10,
                "cutoff_sr": 5.0,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "m3gnet" in str(exc)
    else:
        raise AssertionError("dimenet Bessel flavor must be rejected")
