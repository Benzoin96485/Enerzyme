"""GemNet-T / GemNet-Q Core: registration, symmetry, and conservative forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_T = ROOT / "enerzyme" / "config" / "gemnet_layers_example.yaml"
EXAMPLE_Q = ROOT / "enerzyme" / "config" / "gemnetq_layers_example.yaml"


def _complete_edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i), torch.tensor(idx_j)


def _tiny(**kwargs):
    from enerzyme.models.gemnet import GemNetCore

    params = dict(
        dim_embedding=4,
        num_rbf=3,
        cutoff_sr=5.0,
        num_blocks=2,
        num_spherical=2,
        emb_size_edge=4,
        emb_size_trip=4,
        emb_size_quad=4,
        emb_size_rbf=4,
        emb_size_cbf=4,
        emb_size_sbf=4,
        emb_size_bil_trip=4,
        emb_size_bil_quad=4,
        num_before_skip=1,
        num_after_skip=1,
        num_concat=1,
        num_atom=1,
        envelope_exponent=5,
    )
    params.update(kwargs)
    return GemNetCore(**params)


def _inputs(n=4, dim=4, num_rbf=3, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    idx_i, idx_j = _complete_edges(n)
    pos = torch.randn(n, 3)
    pos = pos - pos.mean(0)
    # Keep every pair inside the embedding cutoff.
    pos = pos / pos.norm(dim=-1).max().clamp(min=1e-3) * 0.8
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    return {
        "atom_embedding": torch.randn(n, dim),
        "rbf": torch.randn(idx_i.numel(), num_rbf),
        "Dij_sr": dist,
        "vij_sr": vij,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "pos": pos.detach().requires_grad_(requires_grad),
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
    rot = torch.eye(3) + math_sin(angle) * k + (1 - math_cos(angle)) * (k @ k)
    return pos @ rot.T


def math_sin(angle: float) -> float:
    import math
    return math.sin(angle)


def math_cos(angle: float) -> float:
    import math
    return math.cos(angle)


def _forward(core, inp):
    return core.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=inp["vij_sr"],
        idx_i_sr=inp["idx_i_sr"],
        idx_j_sr=inp["idx_j_sr"],
    )["atom_feature"]


def test_registration_names():
    from enerzyme.models.ff import get_ff_core
    from enerzyme.models.gemnet import GemNetCore

    for name in ("gemnet", "gemnet-t", "GemNetT"):
        core_cls, build, layers = get_ff_core(name)
        assert core_cls is GemNetCore
        assert build["cutoff_sr"] == 5.0
        core_params = next(layer["params"] for layer in layers if layer["name"] == "Core")
        assert core_params["triplets_only"] is True
    core_cls, build, layers = get_ff_core("gemnet-q")
    assert core_cls is GemNetCore
    assert build["cutoff_sr"] == 10.0
    core_params = next(layer["params"] for layer in layers if layer["name"] == "Core")
    assert core_params["triplets_only"] is False
    assert core_params["emb_cutoff"] == 5.0
    assert core_params["int_cutoff"] == 10.0
    bessel = next(layer["params"] for layer in layers if layer["name"] == "BesselRBF")
    assert bessel["flavor"] == "gemnet"
    assert bessel["basis_cutoff"] == 5.0
    get_ff_core("gemnetq")


def test_feature_shape_hierarchical_last_and_isolated():
    torch.manual_seed(0)
    inp = _inputs()
    core = _tiny()
    feat = _forward(core, inp)
    assert feat.shape == (4, 4, 3)
    last_core = _tiny(output_mode="last")
    last_core.load_state_dict(core.state_dict())
    last = _forward(last_core, inp)
    assert last.shape == (4, 4)
    assert_allclose(last.detach().numpy(), feat[:, :, -1].detach().numpy(), atol=1e-6)

    isolated = _tiny()
    out = isolated.get_output(
        atom_embedding=torch.randn(1, 4),
        rbf=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
        vij_sr=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
    )["atom_feature"]
    assert out.shape == (1, 4, 3)
    assert torch.count_nonzero(out) == 0

    quad = _tiny(triplets_only=False, emb_cutoff=5.0, int_cutoff=5.0)
    qfeat = _forward(quad, inp)
    assert qfeat.shape == (4, 4, 3)
    assert torch.isfinite(qfeat).all()


def test_rotation_translation_and_inversion():
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

    qcore = _tiny(triplets_only=False, emb_cutoff=5.0, int_cutoff=5.0)
    qbase = _forward(qcore, inp)
    qinv = qcore.get_output(
        atom_embedding=inp["atom_embedding"],
        rbf=inp["rbf"],
        Dij_sr=inp["Dij_sr"],
        vij_sr=-inp["vij_sr"],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
    )["atom_feature"]
    assert_allclose(qbase.detach().numpy(), qinv.detach().numpy(), atol=1e-5, rtol=1e-5)


def test_disjoint_molecules_do_not_mix():
    torch.manual_seed(2)
    core = _tiny()
    left = _inputs(n=3, seed=2)
    right = _inputs(n=3, seed=3)
    feat_l = _forward(core, left)
    feat_r = _forward(core, right)
    n_l = left["idx_i_sr"].numel()
    merged = {
        "atom_embedding": torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + 3]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + 3]),
    }
    # rbf rows follow edge order, which is left edges then right edges.
    assert merged["rbf"].shape[0] == n_l + right["idx_i_sr"].numel()
    feat = _forward(core, merged)
    assert_allclose(feat[:3].detach().numpy(), feat_l.detach().numpy(), atol=1e-5)
    assert_allclose(feat[3:].detach().numpy(), feat_r.detach().numpy(), atol=1e-5)


def _stack(
    architecture: str,
    triplets_only: bool,
    cutoff_sr: float,
    basis_cutoff: float | None = None,
    **core_extra,
):
    bessel = {
        "flavor": "gemnet",
        "trainable": False,
        "envelope_exponent": 5,
    }
    if basis_cutoff is not None:
        bessel["basis_cutoff"] = basis_cutoff
    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": bessel},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_blocks": 1,
                "num_spherical": 2,
                "emb_size_edge": 4,
                "emb_size_trip": 4,
                "emb_size_quad": 4,
                "emb_size_rbf": 4,
                "emb_size_cbf": 4,
                "emb_size_sbf": 4,
                "emb_size_bil_trip": 4,
                "emb_size_bil_quad": 4,
                "num_before_skip": 0,
                "num_after_skip": 0,
                "num_concat": 0,
                "num_atom": 1,
                "triplets_only": triplets_only,
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
        architecture,
        layer_params=layers,
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 20,
            "cutoff_sr": cutoff_sr,
        },
        verbose=0,
    )


def test_layer_stack_conserves_energy_and_dimer_forces():
    model = _stack("gemnet-t", True, 5.0)
    n = 4
    idx_i, idx_j = _complete_edges(n)
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
    edges = torch.tensor([0, 1]), torch.tensor([1, 0])
    out_d = model(
        {
            "Ra": dimer,
            "Za": torch.tensor([1, 1]),
            "batch_seg": torch.zeros(2, dtype=torch.long),
            "idx_i": edges[0],
            "idx_j": edges[1],
        }
    )
    bond = dimer[1] - dimer[0]
    cross = torch.linalg.cross(out_d["Fa"][0], bond)
    assert_allclose(cross.detach().numpy(), torch.zeros(3).numpy(), atol=1e-5)
    assert_allclose(
        (out_d["Fa"][0] + out_d["Fa"][1]).detach().numpy(),
        torch.zeros(3).numpy(),
        atol=1e-5,
    )


def test_gemnet_q_uses_the_longer_bond_and_matches_yaml():
    model = _stack(
        "gemnet-q",
        False,
        cutoff_sr=3.0,
        basis_cutoff=1.5,
        emb_cutoff=1.5,
        int_cutoff=3.0,
    )
    # A chain. Embedding edges are the contacts; the 2 Å gap is an interaction bond.
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.3], [3.0, 0.1, 0.0]],
        requires_grad=True,
    )
    idx_i, idx_j = _complete_edges(4)
    out = model(
        {
            "Ra": pos,
            "Za": torch.tensor([6, 6, 6, 1]),
            "batch_seg": torch.zeros(4, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    force = torch.autograd.grad(out["E"].sum(), pos)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-4, rtol=1e-4)
    assert model.triplets_only is False
    assert model.interactions[0].quad is not None

    wrong = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"flavor": "dimenet", "trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_blocks": 1, "num_spherical": 2, "triplets_only": True}},
        {"name": "EnergyReduce"},
    ]
    from enerzyme.models.ff import build_model

    try:
        build_model(
            "gemnet",
            layer_params=wrong,
            build_params={"dim_embedding": 4, "num_rbf": 3, "max_Za": 10, "cutoff_sr": 5.0},
            verbose=0,
        )
    except ValueError as exc:
        assert "gemnet" in str(exc)
    else:
        raise AssertionError("dimenet Bessel flavor must be rejected")

    if EXAMPLE_T.exists():
        with EXAMPLE_T.open() as stream:
            ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
        assert ff["architecture"] == "gemnet-t"
        bessel = next(layer for layer in ff["layers"] if layer["name"] == "BesselRBF")
        core = next(layer for layer in ff["layers"] if layer["name"] == "Core")
        assert bessel["params"]["flavor"] == "gemnet"
        assert core["params"]["triplets_only"] is True
        assert "basis_cutoff" not in bessel["params"]
    if EXAMPLE_Q.exists():
        with EXAMPLE_Q.open() as stream:
            ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
        assert ff["architecture"] == "gemnet-q"
        bessel = next(layer for layer in ff["layers"] if layer["name"] == "BesselRBF")
        core = next(layer for layer in ff["layers"] if layer["name"] == "Core")
        assert ff["build_params"]["cutoff_sr"] == 10.0
        assert bessel["params"]["flavor"] == "gemnet"
        assert bessel["params"]["basis_cutoff"] == 5.0
        assert core["params"]["triplets_only"] is False
    assert core["params"]["emb_cutoff"] == 5.0
    assert core["params"]["int_cutoff"] == 10.0


def test_two_images_conserve_force_without_grad_through_the_partner():
    """Two periodic images of one bond. The reverse partner is a fixed index."""
    model = _stack("gemnet-t", True, 5.0)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 1.0, 0.0]],
        requires_grad=True,
    )
    idx_i = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])
    idx_j = torch.tensor([1, 0, 2, 0, 2, 1, 1, 0])
    offsets = torch.zeros(idx_i.shape[0], 3)
    offsets[-2] = torch.tensor([0.0, 3.0, 0.0])
    offsets[-1] = torch.tensor([0.0, -3.0, 0.0])
    fields = {
        "Za": torch.tensor([1, 6, 1]),
        "batch_seg": torch.zeros(3, dtype=torch.long),
        "idx_i": idx_i,
        "idx_j": idx_j,
        "offsets": offsets,
    }
    out = model({"Ra": pos, **fields})
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    grad = torch.autograd.grad(out["E"].sum(), pos)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-grad).detach().numpy(), atol=1e-5, rtol=1e-5)
    step = 1e-4
    plus = pos.detach().clone()
    minus = pos.detach().clone()
    plus[1, 0] = plus[1, 0] + step
    minus[1, 0] = minus[1, 0] - step
    plus.requires_grad_(True)
    minus.requires_grad_(True)
    finite = (model({"Ra": plus, **fields})["E"] - model({"Ra": minus, **fields})["E"]) / (2 * step)
    assert_allclose(finite.detach().numpy(), grad[1, 0].detach().numpy(), atol=1e-4, rtol=1e-3)


def test_last_mode_rejects_hierarchical_readout_and_mismatched_edges():
    try:
        _stack("gemnet-t", True, 5.0, output_mode="last")
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
