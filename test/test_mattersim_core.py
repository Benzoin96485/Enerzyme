"""MatterSim Core: Graphormer extension of M3GNet, invariance, conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "mattersim_layers_example.yaml"


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


def _gaussian(dist: torch.Tensor, num_rbf: int = 4, cutoff: float = 5.0):
    from enerzyme.models.layers.rbf import GaussianRBFLayer

    layer = GaussianRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        apply_cutoff_fn=False,
        flavor="SchNet",
    )
    return layer._get_rbf(dist)


def _tiny(**kwargs):
    from enerzyme.models.presets.triplet import mattersim

    params = dict(
        dim_embedding=8,
        num_rbf=2,
        cutoff_sr=5.0,
        max_l=2,
        num_blocks=2,
        threebody_cutoff=4.0,
        units=8,
        n_conv_hidden=1,
        num_attn_layers=2,
        num_heads=2,
        graphormer_num_basis=4,
        attention_cutoff=5.0,
        use_centrality=True,
    )
    params.update(kwargs)
    return mattersim(**params)


def _m3gnet_kwargs():
    return dict(
        dim_embedding=8,
        num_rbf=2,
        cutoff_sr=5.0,
        max_l=2,
        num_blocks=2,
        threebody_cutoff=4.0,
        units=8,
        n_conv_hidden=1,
    )


def _inputs(n=4, seed=0, max_l=2, num_rbf=2, graphormer_num_basis=4):
    torch.manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1).clamp(min=1e-6)
    return {
        "atom_embedding": torch.randn(n, 8),
        "rbf": _basis(dist, max_l=max_l, num_rbf=num_rbf),
        "graphormer_rbf": _gaussian(dist, num_rbf=graphormer_num_basis),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "pos": pos,
    }


def _forward(core, inp):
    kwargs = dict(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )
    if core.uses_graphormer_basis:
        kwargs["graphormer_rbf"] = inp["graphormer_rbf"]
    return core.get_output(**kwargs)["atom_feature"]


def test_registration_aliases_and_example_yaml():
    from enerzyme.models.ff import build_model, get_ff_core

    for name in ("mattersim", "m3gnet-graphormer", "m3gnet_graphormer"):
        core_cls, defaults, layers = get_ff_core(name)
        assert core_cls.__name__ == "mattersim"
        assert defaults["cutoff_sr"] == 5.0
        assert defaults["threebody_cutoff"] == 4.0
        assert any(layer["name"] == "GaussianRBF" for layer in layers)
        assert any(layer["name"] == "SphericalBesselRBF" for layer in layers)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "ComposableCore"
    assert model.degree == 9
    assert model.num_attn_layers == 2
    assert model.num_heads == 4
    assert model.graphormer_num_basis == 16
    assert model.attention_cutoff == 5.0
    assert model.feature_irreps == "64x0e"
    assert model.graphormer_basis._name_mapping["rbf"] == "graphormer_rbf"


def test_disabled_graphormer_matches_m3gnet():
    from enerzyme.models.presets.triplet import m3gnet

    torch.manual_seed(0)
    base = m3gnet(**_m3gnet_kwargs())
    torch.manual_seed(0)
    extended = _tiny(num_attn_layers=0, use_centrality=False)
    assert not extended.uses_graphormer_basis
    inp = _inputs(seed=4)
    left = base.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )["atom_feature"]
    right = _forward(extended, inp)
    assert_allclose(left.detach().numpy(), right.detach().numpy(), atol=1e-6)


def test_centrality_and_attention_cutoff_change_features():
    torch.manual_seed(0)
    plain = _tiny(num_attn_layers=0, use_centrality=False)
    torch.manual_seed(1)
    with_centrality = _tiny(num_attn_layers=0, use_centrality=True)
    with_centrality.load_state_dict(plain.state_dict(), strict=False)
    inp = _inputs(seed=5)
    left = _forward(plain, inp)
    right = _forward(with_centrality, inp)
    assert not torch.allclose(left, right, atol=1e-5, rtol=1e-5)

    wide = _tiny(attention_cutoff=5.0, num_attn_layers=1, use_centrality=False)
    narrow = _tiny(attention_cutoff=0.2, num_attn_layers=1, use_centrality=False)
    narrow.load_state_dict(wide.state_dict())
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.3, 1.1, 0.0], [0.4, 0.2, 1.0]]
    )
    idx_i, idx_j = _edges(4)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    cutoff_inp = {
        "atom_embedding": torch.randn(4, 8),
        "rbf": _basis(dist),
        "graphormer_rbf": _gaussian(dist),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }
    assert dist.min() < 5.0
    assert dist.min() > 0.2
    assert not torch.allclose(
        _forward(wide, cutoff_inp),
        _forward(narrow, cutoff_inp),
        atol=1e-5,
        rtol=1e-5,
    )
    try:
        _tiny(attention_cutoff=6.0)
    except ValueError as exc:
        assert "attention_cutoff" in str(exc)
    else:
        raise AssertionError("attention_cutoff above cutoff_sr must be rejected")
    try:
        _tiny(graphormer_num_basis=1)
    except ValueError as exc:
        assert "graphormer_num_basis" in str(exc)
    else:
        raise AssertionError("a single Gaussian center must be rejected")


def test_feature_shape_isolated_atom_and_same_species():
    core = _tiny()
    feat = _forward(core, _inputs())
    assert feat.shape == (4, 8)
    assert torch.isfinite(feat).all()

    empty = {
        "atom_embedding": torch.randn(2, 8),
        "rbf": torch.zeros(0, 4),
        "graphormer_rbf": torch.zeros(0, 4),
        "Dij_sr": torch.zeros(0),
        "vij_sr": torch.zeros(0, 3),
        "idx_i_sr": torch.zeros(0, dtype=torch.long),
        "idx_j_sr": torch.zeros(0, dtype=torch.long),
    }
    empty["atom_embedding"][1] = empty["atom_embedding"][0]
    isolated = _forward(core, empty)
    assert isolated.shape == (2, 8)
    assert torch.isfinite(isolated).all()
    assert_allclose(
        isolated[0].detach().numpy(), isolated[1].detach().numpy(), atol=1e-6
    )
    bare = _tiny(num_attn_layers=0, use_centrality=False)
    embedding = torch.randn(1, 8)
    same = bare.get_output(
        atom_embedding=embedding,
        rbf=torch.zeros(0, 4),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )["atom_feature"]
    assert_allclose(same.detach().numpy(), embedding.detach().numpy(), atol=1e-6)


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
        return _forward(
            core,
            {
                "atom_embedding": embedding,
                "rbf": _basis(dist),
                "graphormer_rbf": _gaussian(dist),
                "Dij_sr": dist,
                "vij_sr": vij,
                "idx_i_sr": idx_i,
                "idx_j_sr": idx_j,
            },
        )

    rotated = _from_pos(_rotate(inp["pos"]))
    assert_allclose(base.detach().numpy(), rotated.detach().numpy(), atol=1e-5, rtol=1e-5)
    translated = _from_pos(inp["pos"] + torch.tensor([3.0, -2.0, 0.4]))
    assert_allclose(base.detach().numpy(), translated.detach().numpy(), atol=1e-5, rtol=1e-5)
    inverted = _from_pos(inp["pos"], sign=-1.0)
    assert_allclose(base.detach().numpy(), inverted.detach().numpy(), atol=1e-5, rtol=1e-5)

    perm = torch.randperm(idx_i.numel())
    shuffled = core.get_output(
        atom_embedding=embedding,
        rbf=inp["rbf"][perm],
        graphormer_rbf=inp["graphormer_rbf"][perm],
        Dij_sr=inp["Dij_sr"][perm],
        vij_sr=inp["vij_sr"][perm],
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
        "graphormer_rbf": torch.cat([left["graphormer_rbf"], right["graphormer_rbf"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
    }
    feat = _forward(core, merged)
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)


def _stack(core_params, include_gaussian=True):
    layers = [
        {"name": "RangeSeparation"},
        {"name": "SphericalBesselRBF"},
    ]
    if include_gaussian:
        layers.append(
            {
                "name": "GaussianRBF",
                "params": {
                    "num_rbf": 4,
                    "flavor": "SchNet",
                    "apply_cutoff_fn": False,
                },
            }
        )
    layers.extend(
        [
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": core_params},
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
    )
    return layers


def test_layer_stack_energy_forces_and_basis_contract():
    from enerzyme.models.ff import build_model

    model = build_model(
        "mattersim",
        layer_params=_stack(
            {
                "num_blocks": 2,
                "num_attn_layers": 1,
                "num_heads": 2,
                "graphormer_num_basis": 4,
                "n_conv_hidden": 1,
                "use_centrality": True,
            }
        ),
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

    try:
        build_model(
            "mattersim",
            layer_params=_stack(
                {"num_blocks": 1, "num_attn_layers": 1, "num_heads": 2, "graphormer_num_basis": 4},
                include_gaussian=False,
            ),
            build_params={
                "dim_embedding": 8,
                "num_rbf": 2,
                "max_l": 2,
                "max_Za": 10,
                "cutoff_sr": 5.0,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "GaussianRBF" in str(exc)
    else:
        raise AssertionError("attention without GaussianRBF must be rejected")

    enveloped = _stack(
        {"num_blocks": 1, "num_attn_layers": 1, "num_heads": 2, "graphormer_num_basis": 4}
    )
    for layer in enveloped:
        if layer["name"] == "GaussianRBF":
            layer["params"]["apply_cutoff_fn"] = True
            layer["params"]["cutoff_fn"] = "bump"
    try:
        build_model(
            "mattersim",
            layer_params=enveloped,
            build_params={
                "dim_embedding": 8,
                "num_rbf": 2,
                "max_l": 2,
                "max_Za": 10,
                "cutoff_sr": 5.0,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "apply_cutoff_fn" in str(exc)
    else:
        raise AssertionError("an envelope on the Gaussian basis must be rejected")

    try:
        build_model(
            "m3gnet",
            layer_params=_stack({"num_blocks": 1}),
            build_params={
                "dim_embedding": 8,
                "num_rbf": 2,
                "max_l": 2,
                "max_Za": 10,
                "cutoff_sr": 5.0,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("M3GNet must reject a second radial basis")
