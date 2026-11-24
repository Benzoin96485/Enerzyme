"""AlphaNet Core: registration, the temporal state, and the external heads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "alphanet_layers_example.yaml"


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
    from enerzyme.models.presets.cartesian_vector import alphanet

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        cutoff_sr=5.0,
        num_layers=2,
        head=2,
        main_chi1=2,
        mp_chi1=2,
        chi2=2,
        hidden_channels_chi=4,
        has_dropout_flag=True,
        has_norm_before_flag=True,
        has_norm_after_flag=False,
        reduce_mode="sum",
    )
    params.update(overrides)
    return alphanet(**params).double()


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
    return {
        "pos": pos,
        "embedding": torch.randn(n, f, dtype=torch.float64),
        "idx": _edges(n),
        "rbf": torch.randn(n * (n - 1), n_rbf, dtype=torch.float64),
    }


def test_alphanet_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    for name in ("alphanet", "AlphaNet"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "alphanet"
        assert build_params["num_rbf"] == 96
        assert layers[1]["params"]["flavor"] == "equflash"
        assert layers[1]["params"]["apply_cutoff_fn"] is False
        assert layers[3]["params"]["fte_scalarization"] == "l2"
        assert layers[3]["params"]["head"] == 16

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    assert ff["architecture"] == "alphanet"
    assert ff["layers"][1]["params"]["flavor"] == "equflash"
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.architecture == "alphanet"
    assert model.dim_feature_out == 128
    assert model.vector_feature_layout == "cartesian_n3f"
    assert model.fte_scalarization == "l2"
    assert not hasattr(model, "output_mode")


def test_alphanet_rejects_bad_settings():
    from enerzyme.models.presets.cartesian_vector import alphanet

    with pytest.raises(ValueError):
        alphanet(dim_embedding=6)
    with pytest.raises(ValueError):
        alphanet(dim_embedding=8, head=3, hidden_channels_chi=8)
    with pytest.raises(ValueError):
        alphanet(dim_embedding=8, head=2, hidden_channels_chi=4, main_chi1=0)
    with pytest.raises(ValueError):
        alphanet(reduce_mode="max")
    with pytest.raises(ValueError):
        alphanet(fte_scalarization="frame")
    with pytest.raises(ValueError):
        alphanet(num_layers=0)


def test_alphanet_quantum_state_has_unit_modulus():
    core = _core(num_layers=1).eval()
    graph = _molecule(4, 8, 4, seed=1)
    out = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
    assert out["atom_feature"].shape == (4, 8)
    assert out["atom_vector_feature"].shape == (4, 3, 8)
    assert out["quantum_feature"].shape == (4, 4)
    real, imag = out["quantum_feature"].split(core.main_chi1, dim=-1)
    modulus = torch.sqrt(real**2 + imag**2) * core.main_chi1
    assert torch.allclose(modulus, torch.ones_like(modulus), atol=1e-6)


def test_alphanet_both_norms_and_mean_reduction():
    graph = _molecule(3, 8, 4, seed=2)
    for overrides in (
        {"has_norm_before_flag": True, "has_norm_after_flag": True},
        {"reduce_mode": "mean", "has_dropout_flag": False},
    ):
        core = _core(num_layers=1, **overrides).eval()
        out = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
        assert torch.isfinite(out["atom_feature"]).all()
        assert torch.isfinite(out["quantum_feature"]).all()
        out["atom_feature"].sum().backward()


def test_alphanet_dropout_is_inactive_in_eval():
    core = _core(num_layers=2, has_norm_before_flag=True)
    graph = _molecule(4, 8, 4, seed=3)
    core.train()
    first = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
    second = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
    assert not torch.allclose(first["atom_feature"], second["atom_feature"])
    core.eval()
    third = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
    fourth = _call(core, graph["pos"], graph["embedding"], graph["rbf"], *graph["idx"])
    assert torch.allclose(third["atom_feature"], fourth["atom_feature"])


@pytest.mark.parametrize("improper", [False, True])
def test_alphanet_l2_rotation_and_translation(improper):
    core = _core(has_dropout_flag=True).eval()
    assert core.fte_scalarization == "l2"
    graph = _molecule(4, 8, 4, seed=4)
    idx_i, idx_j = graph["idx"]
    base = _call(core, graph["pos"], graph["embedding"], graph["rbf"], idx_i, idx_j)
    shift = torch.tensor([2.0, -4.0, 0.5], dtype=torch.float64)
    moved = _call(core, graph["pos"] + shift, graph["embedding"], graph["rbf"], idx_i, idx_j)
    assert torch.allclose(base["atom_feature"], moved["atom_feature"], atol=1e-10)
    assert torch.allclose(base["atom_vector_feature"], moved["atom_vector_feature"], atol=1e-10)
    assert torch.allclose(base["quantum_feature"], moved["quantum_feature"], atol=1e-10)

    rotation = _rotation(5, improper=improper)
    rotated = _call(
        core, graph["pos"] @ rotation, graph["embedding"], graph["rbf"], idx_i, idx_j
    )
    assert torch.allclose(base["atom_feature"], rotated["atom_feature"], atol=1e-8, rtol=1e-6)
    assert torch.allclose(base["quantum_feature"], rotated["quantum_feature"], atol=1e-8, rtol=1e-6)
    expected = torch.einsum("ncf,cd->ndf", base["atom_vector_feature"], rotation)
    assert torch.allclose(expected, rotated["atom_vector_feature"], atol=1e-8, rtol=1e-6)


def test_alphanet_molecules_do_not_mix():
    core = _core(num_layers=1, has_dropout_flag=False).eval()
    graph = _molecule(3, 8, 4, seed=6)
    idx_i, idx_j = graph["idx"]
    single = _call(core, graph["pos"], graph["embedding"], graph["rbf"], idx_i, idx_j)
    offset = torch.tensor([8.0, 1.0, -3.0], dtype=torch.float64)
    both = _call(
        core,
        torch.cat((graph["pos"], graph["pos"] + offset), dim=0),
        torch.cat((graph["embedding"], graph["embedding"]), dim=0),
        torch.cat((graph["rbf"], graph["rbf"]), dim=0),
        torch.cat((idx_i, idx_i + 3)),
        torch.cat((idx_j, idx_j + 3)),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )
    assert torch.allclose(both["atom_feature"][:3], single["atom_feature"], atol=1e-10)
    assert torch.allclose(both["atom_feature"][3:], single["atom_feature"], atol=1e-10)
    assert torch.allclose(both["quantum_feature"][:3], single["quantum_feature"], atol=1e-10)


def test_alphanet_rejects_a_cross_molecule_edge_and_a_stale_distance():
    core = _core(num_layers=1, has_dropout_flag=False).eval()
    graph = _molecule(3, 8, 4, seed=8)
    idx_i, idx_j = graph["idx"]
    with pytest.raises(ValueError, match="molecule"):
        _call(
            core,
            torch.cat((graph["pos"], graph["pos"] + 5), dim=0),
            torch.cat((graph["embedding"], graph["embedding"]), dim=0),
            graph["rbf"],
            idx_i,
            idx_j + 3,
            batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
    vij = graph["pos"][idx_j] - graph["pos"][idx_i]
    with pytest.raises(ValueError, match="Dij_sr"):
        core.get_output(
            atom_embedding=graph["embedding"],
            rbf=graph["rbf"],
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=vij,
            Dij_sr=torch.linalg.vector_norm(vij, dim=-1) * 1.5,
            Ra=graph["pos"],
        )


def test_alphanet_default_transition_is_equivariant():
    from enerzyme.models.presets.cartesian_vector import alphanet

    core = alphanet(
        dim_embedding=8,
        num_rbf=4,
        num_layers=1,
        head=2,
        main_chi1=2,
        mp_chi1=2,
        chi2=2,
        hidden_channels_chi=4,
        has_dropout_flag=False,
    )
    assert core.fte_scalarization == "l2"
    opted = alphanet(
        dim_embedding=8,
        num_rbf=4,
        num_layers=1,
        head=2,
        main_chi1=2,
        mp_chi1=2,
        chi2=2,
        hidden_channels_chi=4,
        has_dropout_flag=False,
        fte_scalarization="l1",
    )
    assert opted.fte_scalarization == "l1"


def test_alphanet_isolated_atom():
    core = _core(num_layers=1, has_dropout_flag=False).eval()
    pos = torch.tensor([[0.2, 0.0, -0.4]], dtype=torch.float64)
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
    real, imag = out["quantum_feature"].split(core.main_chi1, dim=-1)
    modulus = torch.sqrt(real**2 + imag**2) * core.main_chi1
    assert torch.allclose(modulus, torch.ones_like(modulus), atol=1e-6)
    out["atom_feature"].sum().backward()


def test_alphanet_charge_stack_and_default_backward():
    from enerzyme.models.alphanet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
    from enerzyme.models.ff import build_model

    torch.manual_seed(7)
    tiny = build_model(
        "alphanet",
        layer_params=[
            {"name": "RangeSeparation"},
            {
                "name": "BesselRBF",
                "params": {"flavor": "equflash", "apply_cutoff_fn": False},
            },
            {"name": "RandomAtomEmbedding"},
            {
                "name": "Core",
                "params": {
                    "num_layers": 1,
                    "head": 2,
                    "main_chi1": 2,
                    "mp_chi1": 2,
                    "chi2": 2,
                    "hidden_channels_chi": 4,
                    "has_dropout_flag": False,
                    "has_norm_before_flag": True,
                    "has_norm_after_flag": True,
                    "reduce_mode": "sum",
                    "fte_scalarization": "l2",
                },
            },
            {
                "name": "SimpleReadout",
                "params": {"output_fields": ["Ea", "Qa"], "head_type": "dense"},
            },
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
    tiny.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    idx_i, idx_j = _edges(3)
    out = tiny(
        {
            "Ra": pos,
            "Za": torch.tensor([8, 1, 1]),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(3, dtype=torch.long),
        }
    )
    assert out["E"].shape == (1,)
    assert out["Qa"].shape == (3,)
    assert out["Fa"].shape == (3, 3)
    assert out["quantum_feature"].shape == (3, 4)
    assert torch.isfinite(out["Fa"]).all()
    (out["E"].sum() + out["Qa"].sum()).backward()

    model = build_model(
        "AlphaNet",
        layer_params=DEFAULT_LAYER_PARAMS,
        build_params=DEFAULT_BUILD_PARAMS,
        verbose=0,
    ).double()
    model.train()
    wide = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 0.9, 0.3], [0.4, 0.2, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    idx_i, idx_j = _edges(4)
    trained = model(
        {
            "Ra": wide,
            "Za": torch.tensor([6, 1, 1, 8]),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(4, dtype=torch.long),
        }
    )
    assert trained["E"].shape == (1,)
    trained["E"].sum().backward()
    assert torch.isfinite(wide.grad).all()
