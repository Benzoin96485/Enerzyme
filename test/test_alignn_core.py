"""ALIGNN Core: registration, angles, invariance, conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "alignn_layers_example.yaml"


def _rotate(pos: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return pos @ q


def _basis(dist: torch.Tensor, num_rbf: int, cutoff: float) -> torch.Tensor:
    from enerzyme.models.layers.rbf import GaussianRBFLayer

    layer = GaussianRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        flavor="alignn",
        apply_cutoff_fn=False,
    )
    return layer._get_rbf(dist)


def _tiny(**kwargs):
    from enerzyme.models.alignn import ALIGNNCore

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        cutoff_sr=5.0,
        num_angle_rbf=4,
        embedding_features=6,
        hidden_features=8,
        alignn_layers=1,
        gcn_layers=1,
        norm="layer",
        update_order="code",
    )
    params.update(kwargs)
    return ALIGNNCore(**params)


def _star(theta: float, n_feat: int = 8, num_rbf: int = 4):
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [float(torch.cos(torch.tensor(theta))), float(torch.sin(torch.tensor(theta))), 0.0],
        ]
    )
    idx_i = torch.tensor([1, 0, 2, 0])
    idx_j = torch.tensor([0, 1, 0, 2])
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    return {
        "atom_embedding": torch.randn(3, n_feat),
        "rbf": _basis(dist, num_rbf, 5.0),
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
    )


def test_registration_and_example_yaml():
    from enerzyme.models.alignn import PUBLISHED_PROPERTY_CORE_PARAMS
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, defaults, layers = get_ff_core("alignn")
    assert core_cls.__name__ == "ALIGNNCore"
    assert defaults["cutoff_sr"] == 8.0
    assert defaults["num_rbf"] == 80
    assert defaults["num_angle_rbf"] == 40
    assert defaults["hidden_features"] == 64
    assert defaults["Hartree_in_E"] > 20
    assert PUBLISHED_PROPERTY_CORE_PARAMS["alignn_layers"] == 4
    assert PUBLISHED_PROPERTY_CORE_PARAMS["hidden_features"] == 256
    assert PUBLISHED_PROPERTY_CORE_PARAMS["norm"] == "batch"
    assert any(layer["name"] == "GaussianRBF" for layer in layers)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "ALIGNNCore"
    assert model.alignn_layers == 2
    assert model.gcn_layers == 2
    assert model.hidden_features == 64
    assert model.dim_feature_out == 64
    assert model.feature_irreps == "64x0e"
    assert model.update_order == "code"
    assert model.norm == "layer"
    assert model.radial_basis_function.flavor == "alignn"
    assert model.radial_basis_function.apply_cutoff_fn is False
    assert abs(float(model.angle_coeff) - (-1.0 / (2.0 / 39.0))) < 1e-4


def _angle_carrier(theta: float, n_feat: int = 8, num_rbf: int = 4):
    """Star plus one extra atom, with every bond length independent of theta.

    Atom 3 is bonded only to atom 1. Atom 1 then has two incoming bonds, so
    a later convolution can read the angle that the line graph wrote into
    the bond from the center. On the bare star that bond is the leaf's only
    incoming edge and the normalized gate cancels it.
    """
    angle = torch.tensor(theta)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [float(torch.cos(angle)), float(torch.sin(angle)), 0.0],
            [1.0, 0.8, 0.3],
        ]
    )
    pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)]
    idx_i = torch.tensor([dst for _, dst in pairs])
    idx_j = torch.tensor([src for src, _ in pairs])
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    return {
        "atom_embedding": torch.randn(4, n_feat),
        "rbf": _basis(dist, num_rbf, 5.0),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }


def test_angles_enter_only_after_the_bond_update_is_read():
    def pair(seed: int):
        torch.manual_seed(seed)
        embedding = torch.randn(4, 8)
        wide = _angle_carrier(1.2)
        bent = _angle_carrier(2.2)
        wide["atom_embedding"] = embedding
        bent["atom_embedding"] = embedding
        return wide, bent

    wide, bent = pair(6)
    # Every stored bond has the same length at both angles, so the radial
    # basis matches. The cosine at the center does not.
    assert_allclose(wide["Dij_sr"].detach().numpy(), bent["Dij_sr"].detach().numpy(), atol=1e-6)
    assert_allclose(wide["rbf"].detach().numpy(), bent["rbf"].detach().numpy(), atol=1e-6)

    def features(seed: int, **kwargs):
        left, right = pair(seed)
        torch.manual_seed(seed)
        core = _tiny(**kwargs).eval()
        return _forward(core, left)["atom_feature"], _forward(core, right)["atom_feature"]

    # Code order updates atoms before the line graph, so one ALIGNN layer
    # misses the angle. Paper order runs the line graph first, but the two
    # spokes still share one radial embedding, so that gate also cancels.
    code_a, code_b = features(6, alignn_layers=1, gcn_layers=0, update_order="code")
    assert_allclose(code_a.detach().numpy(), code_b.detach().numpy(), atol=1e-6)
    paper_a, paper_b = features(6, alignn_layers=1, gcn_layers=1, update_order="paper")
    assert_allclose(paper_a.detach().numpy(), paper_b.detach().numpy(), atol=1e-6)

    # The code-order line graph writes the angle into the center's outgoing
    # bond. The following convolution has a second incoming bond at atom 1,
    # so the gate no longer cancels. A second paper-order layer does the
    # same after the first atom update has split the two spokes.
    followed_a, followed_b = features(6, alignn_layers=1, gcn_layers=1, update_order="code")
    assert float((followed_a - followed_b).detach().abs().max()) > 1e-4
    paper2_a, paper2_b = features(6, alignn_layers=2, gcn_layers=0, update_order="paper")
    assert float((paper2_a - paper2_b).detach().abs().max()) > 1e-4
    gcn_a, gcn_b = features(6, alignn_layers=0, gcn_layers=2)
    assert_allclose(gcn_a.detach().numpy(), gcn_b.detach().numpy(), atol=1e-6)


def test_feature_shapes_isolated_atom_and_rejections():
    torch.manual_seed(0)
    core = _tiny()
    out = _forward(core, _star(0.7))
    assert out["atom_feature"].shape == (3, 8)
    assert torch.isfinite(out["atom_feature"]).all()

    emb = torch.randn(1, 8)
    alone = core.get_output(
        atom_embedding=emb,
        rbf=torch.zeros(0, 4),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )
    assert alone["atom_feature"].shape == (1, 8)
    assert torch.isfinite(alone["atom_feature"]).all()
    other = core.get_output(
        atom_embedding=emb + 0.5,
        rbf=torch.zeros(0, 4),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )
    assert not torch.allclose(alone["atom_feature"], other["atom_feature"], atol=1e-6)

    try:
        _tiny(update_order="softmax")
    except ValueError as exc:
        assert "update_order" in str(exc)
    else:
        raise AssertionError("an unknown update order must be rejected")
    try:
        _tiny(num_rbf=1)
    except ValueError as exc:
        assert "num_rbf" in str(exc)
    else:
        raise AssertionError("a one-bin bond basis must be rejected")


def test_rotation_translation_inversion_and_edge_order():
    torch.manual_seed(1)
    core = _tiny()
    core.eval()
    inp = _star(0.9)
    base = _forward(core, inp)["atom_feature"]
    embedding = inp["atom_embedding"]
    idx_i, idx_j = inp["idx_i_sr"], inp["idx_j_sr"]

    def _from_pos(pos, sign=1.0):
        vij = sign * (pos[idx_j] - pos[idx_i])
        dist = vij.norm(dim=-1).clamp(min=1e-6)
        return core.get_output(
            atom_embedding=embedding,
            rbf=_basis(dist, 4, 5.0),
            Dij_sr=dist,
            vij_sr=vij,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
        )["atom_feature"]

    assert_allclose(
        base.detach().numpy(),
        _from_pos(_rotate(inp["pos"])).detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_allclose(
        base.detach().numpy(),
        _from_pos(inp["pos"] + torch.tensor([3.0, -2.0, 0.4])).detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    assert_allclose(
        base.detach().numpy(),
        _from_pos(inp["pos"], sign=-1.0).detach().numpy(),
        atol=1e-5,
        rtol=1e-5,
    )
    perm = torch.randperm(idx_i.numel())
    shuffled = core.get_output(
        atom_embedding=embedding,
        rbf=inp["rbf"][perm],
        Dij_sr=inp["Dij_sr"][perm],
        vij_sr=inp["vij_sr"][perm],
        idx_i_sr=idx_i[perm],
        idx_j_sr=idx_j[perm],
    )["atom_feature"]
    assert_allclose(base.detach().numpy(), shuffled.detach().numpy(), atol=1e-5, rtol=1e-5)


def test_disjoint_molecules_and_envelope_and_backtracking():
    torch.manual_seed(2)
    core = _tiny()
    left = _star(0.6)
    right = _star(1.4)
    feat_l = _forward(core, left)["atom_feature"]
    feat_r = _forward(core, right)["atom_feature"]
    merged = {
        "atom_embedding": torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
    }
    feat = _forward(core, merged)["atom_feature"]
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)

    enveloped = _tiny(multiply_envelope=True, envelope_exponent=5)
    enveloped.load_state_dict(core.state_dict())
    enveloped.eval()
    core.eval()
    assert not torch.allclose(
        _forward(core, left)["atom_feature"],
        _forward(enveloped, left)["atom_feature"],
        atol=1e-5,
    )
    direct = _tiny(backtracking=False)
    direct.load_state_dict(core.state_dict())
    direct.eval()
    assert not torch.allclose(
        _forward(core, left)["atom_feature"],
        _forward(direct, left)["atom_feature"],
        atol=1e-5,
    )


def test_layer_stack_energy_forces_and_wrong_basis():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "GaussianRBF",
            "params": {"flavor": "alignn", "apply_cutoff_fn": False},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "alignn_layers": 1,
                "gcn_layers": 1,
                "norm": "layer",
                "num_angle_rbf": 4,
                "embedding_features": 8,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {"output_fields": ["Ea"], "head_type": "dense", "use_bias": True},
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    build = {
        "dim_embedding": 8,
        "num_rbf": 4,
        "hidden_features": 8,
        "max_Za": 20,
        "cutoff_sr": 5.0,
    }
    model = build_model("alignn", layer_params=layers, build_params=build, verbose=0)
    model.double()
    n = 4
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 1.1, 0.2], [0.4, 0.3, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = {
        "Ra": pos,
        "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "idx_i": torch.tensor(idx_i),
        "idx_j": torch.tensor(idx_j),
    }
    out = model(batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    force = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-8)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-8)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(rotation) < 0:
        rotation = rotation.clone()
        rotation[:, 0] *= -1
    rotated = (pos.detach() @ rotation).requires_grad_(True)
    out_r = model({**batch, "Ra": rotated})
    assert_allclose(out["E"].detach().numpy(), out_r["E"].detach().numpy(), atol=1e-8)
    assert_allclose(
        out_r["Fa"].detach().numpy(),
        (out["Fa"].detach() @ rotation).numpy(),
        atol=1e-7,
        rtol=1e-6,
    )
    shifted = (pos.detach() + torch.tensor([1.5, -0.4, 0.2], dtype=torch.float64)).requires_grad_(True)
    out_t = model({**batch, "Ra": shifted})
    assert_allclose(out["E"].detach().numpy(), out_t["E"].detach().numpy(), atol=1e-8)
    assert_allclose(out["Fa"].detach().numpy(), out_t["Fa"].detach().numpy(), atol=1e-8)

    eps = 1e-5
    plus = pos.detach().clone()
    minus = pos.detach().clone()
    plus[1, 0] = plus[1, 0] + eps
    minus[1, 0] = minus[1, 0] - eps
    plus.requires_grad_(True)
    minus.requires_grad_(True)
    e_plus = model({**batch, "Ra": plus})["E"]
    e_minus = model({**batch, "Ra": minus})["E"]
    finite = -(e_plus - e_minus) / (2 * eps)
    assert_allclose(finite.detach().numpy(), out["Fa"][1, 0].detach().numpy(), atol=1e-5, rtol=1e-4)

    alone = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
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
    assert_allclose(out_a["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-8)

    wrong = [
        {"name": "RangeSeparation"},
        {"name": "GaussianRBF", "params": {"flavor": "SchNet", "apply_cutoff_fn": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"alignn_layers": 1, "gcn_layers": 0}},
        {"name": "EnergyReduce"},
    ]
    try:
        build_model("alignn", layer_params=wrong, build_params=build, verbose=0)
    except ValueError as exc:
        assert "alignn" in str(exc)
    else:
        raise AssertionError("a SchNet Gaussian basis must be rejected")

    mismatched = [
        {"name": "RangeSeparation"},
        {
            "name": "GaussianRBF",
            "params": {"flavor": "alignn", "apply_cutoff_fn": False, "num_rbf": 6},
        },
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"alignn_layers": 1, "gcn_layers": 0, "num_angle_rbf": 4}},
    ]
    try:
        build_model("alignn", layer_params=mismatched, build_params=build, verbose=0)
    except ValueError as exc:
        assert "num_rbf" in str(exc)
    else:
        raise AssertionError("a radial width mismatch must be rejected")
