"""GemNet-OC Core: registration, symmetry, caps, and conservative forces."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "gemnet_oc_layers_example.yaml"
TRAIN = ROOT / "enerzyme" / "config" / "train.yaml"


def _complete_edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i), torch.tensor(idx_j)


def _tiny(**kwargs):
    from enerzyme.models.gemnet import GemNetOCCore
    from enerzyme.models.layers.rbf import GaussianRBFLayer

    params = dict(
        dim_embedding=4,
        num_rbf=4,
        cutoff_sr=5.0,
        num_blocks=1,
        num_spherical=2,
        emb_size_edge=4,
        emb_size_trip_in=4,
        emb_size_trip_out=4,
        emb_size_quad_in=4,
        emb_size_quad_out=4,
        emb_size_aint_in=4,
        emb_size_aint_out=4,
        emb_size_rbf=4,
        emb_size_cbf=4,
        emb_size_sbf=4,
        num_before_skip=0,
        num_after_skip=0,
        num_concat=0,
        num_atom=1,
        num_atom_emb_layers=0,
        num_output_afteratom=1,
        cutoff=5.0,
        cutoff_qint=5.0,
        cutoff_aeaint=5.0,
        cutoff_aint=5.0,
        max_neighbors=8,
        max_neighbors_qint=8,
        max_neighbors_aeaint=8,
        max_neighbors_aint=8,
        envelope_exponent=5,
    )
    params.update(kwargs)
    core = GemNetOCCore(**params)
    core.radial_basis = GaussianRBFLayer(
        params["num_rbf"],
        params["cutoff_sr"],
        apply_cutoff_fn=False,
        flavor="gemnet-oc",
        envelope_exponent=params["envelope_exponent"],
    )
    return core


def _inputs(n=4, dim=4, num_rbf=4, seed=0):
    torch.manual_seed(seed)
    idx_i, idx_j = _complete_edges(n)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.15, 0.0],
            [0.2, 1.05, 0.1],
            [1.1, 0.85, 0.45],
            [0.4, 0.35, 0.95],
            [1.4, 0.2, 0.7],
        ][:n]
    )
    vij = pos[idx_j] - pos[idx_i]
    return {
        "atom_embedding": torch.randn(n, dim),
        "rbf": torch.randn(idx_i.numel(), num_rbf),
        "Dij_sr": vij.norm(dim=-1),
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "pos": pos,
    }


def _rotate(pos: torch.Tensor) -> torch.Tensor:
    axis = torch.tensor([0.2, -0.5, 0.8])
    axis = axis / axis.norm()
    angle = 0.7
    k = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rot = torch.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
    return pos @ rot.T


def _forward(core, inp):
    return core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )["atom_feature"]


def _stack(**core_extra):
    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "GaussianRBF",
            "params": {
                "flavor": "gemnet-oc",
                "apply_cutoff_fn": False,
                "envelope_exponent": 5,
            },
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_blocks": 1,
                "num_spherical": 2,
                "emb_size_edge": 4,
                "emb_size_trip_in": 4,
                "emb_size_trip_out": 4,
                "emb_size_quad_in": 4,
                "emb_size_quad_out": 4,
                "emb_size_aint_in": 4,
                "emb_size_aint_out": 4,
                "emb_size_rbf": 4,
                "emb_size_cbf": 4,
                "emb_size_sbf": 4,
                "num_before_skip": 0,
                "num_after_skip": 0,
                "num_concat": 0,
                "num_atom": 1,
                "num_atom_emb_layers": 0,
                "num_output_afteratom": 1,
                "max_neighbors": 8,
                "max_neighbors_qint": 8,
                "max_neighbors_aeaint": 8,
                "max_neighbors_aint": 8,
                "envelope_exponent": 5,
                **core_extra,
            },
        },
        {
            "name": "HierachicalReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "use_bias": False,
                "keep_feature": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    from enerzyme.models.ff import build_model

    return build_model(
        "gemnet-oc",
        layer_params=layers,
        build_params={
            "dim_embedding": 4,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 5.0,
        },
        verbose=0,
    )


def _molecule(model, pos, za=None):
    n = pos.shape[0]
    idx_i, idx_j = _complete_edges(n)
    if za is None:
        za = torch.tensor([1, 6, 8, 1, 6, 7][:n])
    return model(
        {
            "Ra": pos,
            "Za": za,
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )


def test_registration_and_example_yaml():
    from enerzyme.models.ff import get_ff_core
    from enerzyme.models.gemnet import GemNetOCCore

    for name in ("gemnet-oc", "gemnetoc", "gemnet_oc"):
        core_cls, build, layers = get_ff_core(name)
        assert core_cls is GemNetOCCore
        assert build["cutoff_sr"] == 12.0
        assert build["dim_embedding"] == 256
        assert build["num_rbf"] == 128
    core = next(layer for layer in layers if layer["name"] == "Core")["params"]
    assert core["num_blocks"] == 4
    assert core["num_spherical"] == 7
    assert core["emb_size_edge"] == 512
    assert core["cutoff"] == 12.0
    assert core["max_neighbors"] == 30
    assert core["max_neighbors_qint"] == 8
    assert core["quad_interaction"] is True
    assert core["output_mode"] == "feature"
    gaussian = next(layer for layer in layers if layer["name"] == "GaussianRBF")
    assert gaussian["params"]["flavor"] == "gemnet-oc"
    assert gaussian["params"]["apply_cutoff_fn"] is False
    assert "Force" in {layer["name"] for layer in layers}
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    assert ff["architecture"] == "gemnet-oc"
    assert ff["build_params"]["cutoff_sr"] == 12.0
    example_core = next(layer for layer in ff["layers"] if layer["name"] == "Core")
    assert example_core["params"]["emb_size_sbf"] == 32
    assert example_core["params"]["atom_interaction"] is True
    with TRAIN.open() as stream:
        hub = yaml.safe_load(stream)["Modelhub"]
    # Modelhub merges both maps. Entries after ``external_FFs`` in train.yaml
    # share that key because the file does not indent back.
    train = {**hub["internal_FFs"], **hub["external_FFs"]}["FF34"]
    assert train["architecture"] == "gemnet-oc"
    assert train["active"] is False


def test_feature_shape_isolated_atom_and_last_mode():
    torch.manual_seed(0)
    inp = _inputs()
    core = _tiny()
    feat = _forward(core, inp)
    assert feat.shape == (4, 4, 2)
    assert torch.isfinite(feat).all()
    last = _tiny(output_mode="last")
    last.load_state_dict(core.state_dict())
    last_feat = _forward(last, inp)
    assert last_feat.shape == (4, 4)
    assert_allclose(last_feat.detach().numpy(), feat[:, :, -1].detach().numpy(), atol=1e-6)

    isolated = _tiny()
    emb = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    empty = {
        "rbf": torch.zeros(0, 4),
        "Dij_sr": torch.zeros(0),
        "vij_sr": torch.zeros(0, 3),
        "idx_i_sr": torch.zeros(0, dtype=torch.long),
        "idx_j_sr": torch.zeros(0, dtype=torch.long),
    }
    one = isolated.get_output(atom_embedding=emb[:1], **empty)["atom_feature"]
    two = isolated.get_output(atom_embedding=emb[1:], **empty)["atom_feature"]
    assert one.shape == (1, 4, 2)
    assert torch.isfinite(one).all() and torch.count_nonzero(one) > 0
    assert not torch.allclose(one, two)


def test_symmetries_edge_order_and_batch_isolation():
    torch.manual_seed(1)
    core = _tiny()
    inp = _inputs(seed=1)
    base = _forward(core, inp)
    rot = _rotate(inp["pos"])
    idx_i, idx_j = inp["idx_i_sr"], inp["idx_j_sr"]
    vij = rot[idx_j] - rot[idx_i]
    rotated = core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=vij.norm(dim=-1),
        vij_sr=vij,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
    )["atom_feature"]
    assert_allclose(base.detach().numpy(), rotated.detach().numpy(), atol=1e-5, rtol=1e-5)

    shifted = rot + torch.tensor([3.0, -2.0, 0.5])
    vij_s = shifted[idx_j] - shifted[idx_i]
    translated = core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=vij_s.norm(dim=-1),
        vij_sr=vij_s,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
    )["atom_feature"]
    assert_allclose(rotated.detach().numpy(), translated.detach().numpy(), atol=1e-5)

    inverted = core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=-inp["vij_sr"],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
    )["atom_feature"]
    assert_allclose(base.detach().numpy(), inverted.detach().numpy(), atol=1e-5, rtol=1e-5)

    order = torch.randperm(idx_i.numel())
    permuted = core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"][order],
        Dij_sr=inp["Dij_sr"][order],
        vij_sr=inp["vij_sr"][order],
        idx_i_sr=idx_i[order],
        idx_j_sr=idx_j[order],
    )["atom_feature"]
    assert_allclose(base.detach().numpy(), permuted.detach().numpy(), atol=1e-5)

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


def test_neighbor_cap_and_optional_interactions():
    torch.manual_seed(4)
    inp = _inputs()
    wide = _tiny(max_neighbors=8)
    capped = _tiny(max_neighbors=1)
    capped.load_state_dict(wide.state_dict())
    assert not torch.allclose(_forward(wide, inp), _forward(capped, inp))

    bare = _tiny(
        quad_interaction=False,
        atom_edge_interaction=False,
        edge_atom_interaction=False,
        atom_interaction=False,
    )
    assert bare.interactions[0].quad is None
    feat = _forward(bare, inp)
    assert feat.shape == (4, 4, 2) and torch.isfinite(feat).all()


def test_build_rejects_a_wrong_basis_and_last_mode_readout():
    from enerzyme.models.ff import build_model

    bad_flavor = [
        {"name": "RangeSeparation"},
        {"name": "GaussianRBF", "params": {"flavor": "SchNet"}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_blocks": 1, "num_spherical": 2}},
        {"name": "EnergyReduce"},
    ]
    try:
        build_model(
            "gemnet-oc",
            layer_params=bad_flavor,
            build_params={"dim_embedding": 4, "num_rbf": 4, "max_Za": 10, "cutoff_sr": 5.0},
            verbose=0,
        )
    except ValueError as exc:
        assert "gemnet-oc" in str(exc)
    else:
        raise AssertionError("SchNet Gaussian flavor must be rejected")

    try:
        _stack(output_mode="last")
    except ValueError as exc:
        assert "output_mode" in str(exc)
    else:
        raise AssertionError("last mode with HierachicalReadout should fail")

    core = _tiny()
    inp = _inputs()
    try:
        core.get_output(
            atom_embedding=inp["atom_embedding"],
            rbf=inp["rbf"][:-1],
            Dij_sr=inp["Dij_sr"],
            vij_sr=inp["vij_sr"],
            idx_i_sr=inp["idx_i_sr"],
            idx_j_sr=inp["idx_j_sr"],
        )
    except ValueError as exc:
        assert "same edges" in str(exc)
    else:
        raise AssertionError("mismatched edge tensors should fail")


def test_energy_is_conservative_and_the_gaussian_is_not_duplicated():
    torch.manual_seed(0)
    model = _stack()
    centers = [key for key in model.state_dict() if key.endswith("center")]
    assert len(centers) == 1
    assert all(not key.startswith("radial_basis") for key in model.state_dict())

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.15, 0.0], [0.2, 1.05, 0.1], [1.1, 0.85, 0.45]],
        requires_grad=True,
    )
    out = _molecule(model, pos)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    grad = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-grad).detach().numpy(), atol=1e-5, rtol=1e-5)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-4)
    step = 1e-3
    plus = pos.detach().clone()
    minus = pos.detach().clone()
    plus[1, 0] = plus[1, 0] + step
    minus[1, 0] = minus[1, 0] - step
    plus.requires_grad_(True)
    minus.requires_grad_(True)
    finite = (_molecule(model, plus)["E"] - _molecule(model, minus)["E"]) / (2 * step)
    assert_allclose(finite.detach().numpy(), grad[1, 0].detach().numpy(), atol=1e-4, rtol=1e-3)

    dimer = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], requires_grad=True)
    out_d = _molecule(model, dimer, za=torch.tensor([1, 1]))
    bond = dimer[1] - dimer[0]
    cross = torch.linalg.cross(out_d["Fa"][0], bond)
    assert_allclose(cross.detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)
    assert_allclose(
        (out_d["Fa"][0] + out_d["Fa"][1]).detach().numpy(),
        torch.zeros(3).numpy(),
        atol=1e-5,
    )

    lone = torch.tensor([[0.2, -0.4, 0.1]], requires_grad=True)
    out_l = model(
        {
            "Ra": lone,
            "Za": torch.tensor([8]),
            "batch_seg": torch.zeros(1, dtype=torch.long),
            "idx_i": torch.zeros(0, dtype=torch.long),
            "idx_j": torch.zeros(0, dtype=torch.long),
        }
    )
    assert torch.isfinite(out_l["E"]).all()
    assert_allclose(out_l["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-6)

    short = _stack(cutoff=1.8, cutoff_qint=1.8, cutoff_aeaint=1.8, cutoff_aint=4.0)
    spread = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]],
        requires_grad=True,
    )
    out_s = _molecule(short, spread)
    assert torch.isfinite(out_s["E"]).all() and torch.isfinite(out_s["Fa"]).all()
    grad_s = torch.autograd.grad(out_s["E"].sum(), spread)[0]
    assert_allclose(
        out_s["Fa"].detach().numpy(), (-grad_s).detach().numpy(), atol=1e-4, rtol=1e-4
    )
