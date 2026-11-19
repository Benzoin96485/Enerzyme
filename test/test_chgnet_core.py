"""CHGNet Core: registration, invariance, magmom latent, conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "chgnet_layers_example.yaml"


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


def _basis(dist: torch.Tensor, num_rbf: int, cutoff: float):
    from enerzyme.models.layers.rbf import BesselRBFLayer

    layer = BesselRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        flavor="gemnet",
        trainable=False,
        envelope_exponent=5,
    )
    return layer._get_rbf(dist)


def _tiny(**kwargs):
    from enerzyme.models.chgnet import CHGNetCore

    params = dict(
        dim_embedding=8,
        num_rbf=3,
        cutoff_sr=5.0,
        bond_cutoff=3.0,
        num_angular=5,
        num_blocks=2,
        atom_hidden=8,
        bond_hidden=8,
        angle_hidden=0,
        gmlp_norm="layer",
        readout_norm="layer",
    )
    params.update(kwargs)
    return CHGNetCore(**params)


def _inputs(n=4, seed=0, num_rbf=3):
    torch.manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.rand(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1).clamp(min=1e-6)
    return {
        "atom_embedding": torch.randn(n, 8),
        "rbf": _basis(dist, num_rbf, 5.0),
        "rbf_bg": _basis(dist, num_rbf, 3.0),
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
        rbf_bg=inp["rbf_bg"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )


def test_registration_and_example_yaml():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, defaults, layers = get_ff_core("chgnet")
    assert core_cls.__name__ == "CHGNetCore"
    assert defaults["cutoff_sr"] == 5.0
    assert defaults["bond_cutoff"] == 3.0
    assert defaults["num_rbf"] == 9
    assert defaults["num_angular"] == 9
    assert defaults["Hartree_in_E"] > 20
    assert any(layer["name"] == "AbsoluteLinearReadout" for layer in layers)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "CHGNetCore"
    assert model.num_blocks == 4
    assert model.bond_cutoff == 3.0
    assert model.dim_feature_out == 64
    assert model.feature_irreps == "64x0e"
    assert model.bond_basis._name_mapping["rbf"] == "rbf_bg"
    assert model.atom_basis.basis_cutoff == 5.0
    assert model.bond_basis.basis_cutoff == 3.0
    assert model.fourier.learnable is True


def test_feature_shapes_and_isolated_atom():
    core = _tiny(num_blocks=1)
    out = _forward(core, _inputs())
    assert out["atom_feature"].shape == (4, 8)
    assert out["atom_feature_charge"].shape == (4, 8)
    assert torch.isfinite(out["atom_feature"]).all()

    emb = torch.randn(1, 8)
    alone = core.get_output(
        atom_embedding=emb,
        rbf=torch.zeros(0, 3),
        rbf_bg=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )
    assert_allclose(
        alone["atom_feature_charge"].detach().numpy(),
        emb.detach().numpy(),
        atol=1e-6,
    )
    assert_allclose(
        alone["atom_feature"].detach().numpy(),
        core.readout_norm(emb).detach().numpy(),
        atol=1e-6,
    )


def test_bond_cutoff_changes_features_and_rejects_a_longer_cutoff():
    torch.manual_seed(0)
    wide = _tiny(bond_cutoff=3.0)
    narrow = _tiny(bond_cutoff=0.2)
    narrow.load_state_dict(wide.state_dict())
    idx_i, idx_j = _edges(4)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.05], [0.3, 0.2, 1.2]]
    )
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    inp = {
        "atom_embedding": torch.randn(4, 8),
        "rbf": _basis(dist, 3, 5.0),
        "rbf_bg": _basis(dist, 3, 3.0),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }
    assert float(dist.min()) > 0.2
    assert float(dist.max()) < 3.0
    left = _forward(wide, inp)["atom_feature"]
    right = _forward(narrow, inp)["atom_feature"]
    assert not torch.allclose(left, right, atol=1e-5, rtol=1e-5)
    try:
        _tiny(bond_cutoff=6.0)
    except ValueError as exc:
        assert "bond_cutoff" in str(exc)
    else:
        raise AssertionError("bond_cutoff above cutoff_sr must be rejected")
    try:
        _tiny(num_angular=4)
    except ValueError as exc:
        assert "num_angular" in str(exc)
    else:
        raise AssertionError("an even angular width must be rejected")


def test_charge_latent_precedes_the_last_convolution():
    torch.manual_seed(1)
    core = _tiny(num_blocks=2, readout_norm=None)
    core.eval()
    out = _forward(core, _inputs(seed=1))
    assert not torch.allclose(
        out["atom_feature"], out["atom_feature_charge"], atol=1e-6, rtol=1e-6
    )


def test_rotation_translation_inversion_and_edge_order():
    torch.manual_seed(1)
    core = _tiny()
    core.eval()
    inp = _inputs(seed=1)
    base = _forward(core, inp)["atom_feature"]
    idx_i, idx_j = inp["idx_i_sr"], inp["idx_j_sr"]
    embedding = inp["atom_embedding"]

    def _from_pos(pos, sign=1.0):
        vij = sign * (pos[idx_j] - pos[idx_i])
        dist = vij.norm(dim=-1).clamp(min=1e-6)
        return core.get_output(
            atom_embedding=embedding,
            rbf=_basis(dist, 3, 5.0),
            rbf_bg=_basis(dist, 3, 3.0),
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
        rbf_bg=inp["rbf_bg"][perm],
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
    feat_l = _forward(core, left)["atom_feature"]
    feat_r = _forward(core, right)["atom_feature"]
    merged = {
        "atom_embedding": torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "rbf_bg": torch.cat([left["rbf_bg"], right["rbf_bg"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
    }
    feat = _forward(core, merged)["atom_feature"]
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)


def test_layer_stack_energy_forces_magmom_and_wrong_basis():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"flavor": "gemnet", "trainable": False}},
        {
            "name": "BesselRBF",
            "params": {"flavor": "gemnet", "trainable": False, "cutoff_sr": 3.0},
        },
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_blocks": 2, "num_angular": 5}},
        {"name": "AbsoluteLinearReadout", "params": {"output_fields": ["Ma"]}},
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "mlp",
                "num_hidden_layers": 1,
                "activation_fn": "swish",
                "use_bias_out": True,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    build = {
        "dim_embedding": 8,
        "num_rbf": 3,
        "max_Za": 20,
        "cutoff_sr": 5.0,
        "bond_cutoff": 3.0,
        "envelope_exponent": 5,
    }
    model = build_model("chgnet", layer_params=layers, build_params=build, verbose=0)
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
    assert out["Ma"].shape == (n,)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()
    assert torch.isfinite(out["Ma"]).all()
    assert torch.all(out["Ma"] >= 0)
    force = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-5, rtol=1e-5)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)
    magmom_force = torch.autograd.grad(out["Ma"].sum(), pos, retain_graph=True)[0]
    assert magmom_force is not None and torch.isfinite(magmom_force).all()

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
    assert torch.all(out_d["Ma"] >= 0)
    assert_allclose(out_d["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)

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
    assert torch.isfinite(out_a["Ma"]).all()
    assert_allclose(out_a["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-6)

    wrong = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"flavor": "gemnet", "trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_blocks": 1}},
        {"name": "EnergyReduce"},
    ]
    try:
        build_model("chgnet", layer_params=wrong, build_params=build, verbose=0)
    except ValueError as exc:
        assert "gemnet" in str(exc)
    else:
        raise AssertionError("a single radial basis must be rejected")
