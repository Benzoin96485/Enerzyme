"""PaiNN Core: registration, equivariance, energy/force stack, tensor heads."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "painn_layers_example.yaml"
SPECTRA = ROOT / "enerzyme" / "config" / "painn_spectra_layers_example.yaml"


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
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _rotate_vectors(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    # Row vectors: v' = v @ R when positions transform as R_row @ R.
    return torch.einsum("ncf,cd->ndf", vectors, rotation)


def test_painn_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("painn", "PaiNN"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "PaiNNCore"
        assert build_params["num_rbf"] == 20
        assert build_params["cutoff_fn"] == "cosine"
        assert layers[1]["params"]["flavor"] == "painn"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "PaiNNCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "two_layer"
    assert readout.head[0].dim_feature_out == 4  # pyramidal: F // 2


def test_two_layer_default_width_is_unchanged():
    from enerzyme.models.layers.readout import SimpleReadout

    class _Core:
        dim_feature_out = 8

    plain = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="two_layer",
        activation_fn="swish",
        activation_params={
            "dim_feature": 1,
            "initial_alpha": 1.0,
            "initial_beta": 1.0,
            "learnable": False,
        },
    )
    assert plain.head[0].dim_feature_out == 8
    pyramidal = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="two_layer",
        pyramidal=True,
        activation_fn="swish",
    )
    assert pyramidal.head[0].dim_feature_out == 4


def test_bessel_painn_matches_sine_over_r():
    import math

    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 5.0
    layer = BesselRBFLayer(
        num_rbf=4, cutoff_sr=cutoff, flavor="painn", trainable=False
    )
    assert layer.apply_cutoff_fn is False
    distances = torch.tensor([0.0, 0.4, 1.5, 4.9])
    got = layer.get_rbf(distances)
    n = torch.arange(1, 5, dtype=distances.dtype)
    expected = []
    for dist in distances:
        if dist == 0:
            expected.append(torch.zeros(4))
        else:
            expected.append(torch.sin(n * math.pi * dist / cutoff) / dist)
    expected = torch.stack(expected)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    default = BesselRBFLayer(
        num_rbf=4,
        cutoff_sr=cutoff,
        flavor="default",
        apply_cutoff_fn=False,
        trainable=False,
    )
    scale = math.sqrt(2.0 / cutoff)
    assert torch.allclose(
        default.get_rbf(distances[1:]) ,
        got[1:] * scale,
        atol=1e-6,
        rtol=1e-6,
    )


def test_painn_feature_shapes_rotation_and_isolated_atom():
    from enerzyme.models.painn import PaiNNCore

    torch.manual_seed(0)
    n, f, n_rbf = 5, 8, 4
    core = PaiNNCore(dim_embedding=f, num_rbf=n_rbf, num_interactions=3, cutoff_sr=5.0)
    core.eval()
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    embedding = torch.randn(n, f)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    rbf = torch.randn(idx_i.numel(), n_rbf)

    def _run(points):
        edges = points[idx_j] - points[idx_i]
        return core.get_output(
            atom_embedding=embedding,
            rbf=rbf,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=edges,
            Dij_sr=torch.linalg.vector_norm(edges, dim=-1),
        )

    out = _run(pos)
    assert out["atom_feature"].shape == (n, f)
    assert out["atom_vector_feature"].shape == (n, 3, f)
    rotation = _rotation(1)
    rotated = _run(pos @ rotation)
    assert torch.allclose(out["atom_feature"], rotated["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _rotate_vectors(out["atom_vector_feature"], rotation),
        rotated["atom_vector_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
    shifted = _run(pos + torch.tensor([3.0, -2.0, 0.5]))
    assert torch.allclose(out["atom_feature"], shifted["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        out["atom_vector_feature"], shifted["atom_vector_feature"], atol=1e-5, rtol=1e-5
    )

    # Same weights, vector-propagation ablation changes the representation.
    torch.manual_seed(2)
    full = PaiNNCore(dim_embedding=f, num_rbf=n_rbf, num_interactions=2)
    torch.manual_seed(2)
    ablated = PaiNNCore(
        dim_embedding=f, num_rbf=n_rbf, num_interactions=2, propagate_vectors=False
    )
    kwargs = dict(
        atom_embedding=embedding,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=dist,
    )
    assert not torch.allclose(
        full.get_output(**kwargs)["atom_feature"],
        ablated.get_output(**kwargs)["atom_feature"],
        atol=1e-5,
        rtol=1e-5,
    )

    isolated = PaiNNCore(dim_embedding=f, num_rbf=n_rbf, num_interactions=2)
    alone = isolated.get_output(
        atom_embedding=torch.randn(1, f),
        rbf=torch.zeros(0, n_rbf),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
    )
    assert alone["atom_feature"].shape == (1, f)
    assert torch.count_nonzero(alone["atom_vector_feature"]) == 0
    assert torch.isfinite(alone["atom_feature"]).all()


def test_painn_messages_do_not_cross_molecules():
    from enerzyme.models.painn import PaiNNCore

    torch.manual_seed(3)
    f, n_rbf = 6, 3
    core = PaiNNCore(dim_embedding=f, num_rbf=n_rbf, num_interactions=2)
    core.eval()

    def _pack(n):
        idx_i, idx_j = _edges(n)
        pos = torch.randn(n, 3)
        emb = torch.randn(n, f)
        vij = pos[idx_j] - pos[idx_i]
        return {
            "atom_embedding": emb,
            "rbf": torch.randn(idx_i.numel(), n_rbf),
            "idx_i_sr": idx_i,
            "idx_j_sr": idx_j,
            "vij_sr": vij,
            "Dij_sr": torch.linalg.vector_norm(vij, dim=-1),
        }

    left = _pack(3)
    right = _pack(4)
    separate_l = core.get_output(**left)["atom_feature"]
    separate_r = core.get_output(**right)["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    joined = {
        "atom_embedding": torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        "rbf": torch.cat([left["rbf"], right["rbf"]]),
        "idx_i_sr": torch.cat([left["idx_i_sr"], right["idx_i_sr"] + n_left]),
        "idx_j_sr": torch.cat([left["idx_j_sr"], right["idx_j_sr"] + n_left]),
        "vij_sr": torch.cat([left["vij_sr"], right["vij_sr"]]),
        "Dij_sr": torch.cat([left["Dij_sr"], right["Dij_sr"]]),
    }
    both = core.get_output(**joined)["atom_feature"]
    assert torch.allclose(both[:n_left], separate_l, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], separate_r, atol=1e-6, rtol=1e-6)


def _tiny_layers(spectra: bool = False):
    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "painn", "trainable": False, "apply_cutoff_fn": False},
        },
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_interactions": 2}},
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
    if spectra:
        layers.extend(
            [
                {
                    "name": "EquivariantDipole",
                    "params": {"n_layers": 2, "return_charges": True, "predict_magnitude": True},
                },
                {"name": "EquivariantPolarizability", "params": {"n_layers": 2}},
            ]
        )
    layers.extend([{"name": "EnergyReduce"}, {"name": "Force"}])
    return layers


def _forward_stack(model, pos, za, batch_seg, total_charge=None):
    n = pos.shape[0]
    idx_i, idx_j = _edges(n)
    net = {
        "Ra": pos,
        "Za": za,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "batch_seg": batch_seg,
    }
    if total_charge is not None:
        net["Q"] = total_charge
    return model(net)


def test_painn_stack_energy_force_and_spectra_heads():
    from enerzyme.models.ff import build_model

    torch.manual_seed(4)
    model = build_model(
        "painn",
        layer_params=_tiny_layers(spectra=True),
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

    # Two molecules, far enough apart that the cutoff drops cross edges.
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.2, 0.8, 0.1],
            [20.0, 0.0, 0.0],
            [20.7, 0.2, 0.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([8, 1, 1, 6, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    charge = torch.tensor([0.0, 1.0], dtype=torch.float64)
    out = _forward_stack(model, pos, za, batch, charge)
    assert out["E"].shape == (2,)
    assert out["Fa"].shape == (5, 3)
    assert out["M2"].shape == (2, 3)
    assert out["M2mag"].shape == (2,)
    assert out["Qa"].shape == (5,)
    assert out["polarizability"].shape == (2, 3, 3)
    assert torch.allclose(out["Qa"][:3].sum(), charge[0], atol=1e-8)
    assert torch.allclose(out["Qa"][3:].sum(), charge[1], atol=1e-8)
    assert torch.allclose(out["M2mag"], torch.linalg.vector_norm(out["M2"], dim=-1))
    # alpha is symmetric by construction (diag + outer + outer^T).
    alpha = out["polarizability"]
    assert torch.allclose(alpha, alpha.transpose(-1, -2), atol=1e-8)
    assert torch.isfinite(out["Fa"]).all()

    # Neutral dipole is origin-independent. Recompute on a shifted copy.
    shifted = pos.detach() + torch.tensor([4.0, -1.0, 2.0], dtype=torch.float64)
    shifted.requires_grad_(True)
    neutral = torch.zeros(2, dtype=torch.float64)
    base = _forward_stack(model, pos.detach().requires_grad_(True), za, batch, neutral)
    moved = _forward_stack(model, shifted, za, batch, neutral)
    assert torch.allclose(base["M2"], moved["M2"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(base["E"], moved["E"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(5).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward_stack(model, rotated_pos, za, batch, neutral)
    assert torch.allclose(base["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(base["M2"] @ rotation, rotated["M2"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)
    alpha_rot = torch.einsum("ab,nbc,cd->nad", rotation.T, base["polarizability"], rotation)
    assert torch.allclose(alpha_rot, rotated["polarizability"], atol=1e-6, rtol=1e-6)


def test_painn_force_matches_energy_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = build_model(
        "painn",
        layer_params=_tiny_layers(spectra=False),
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 10,
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
    za = torch.tensor([6, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    out = _forward_stack(model, pos, za, batch)
    analytical = out["Fa"].detach()
    energy = float(out["E"][0].detach())
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
            e_plus = _forward_stack(model, plus, za, batch)["E"][0]
            e_minus = _forward_stack(model, minus, za, batch)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-6, rtol=1e-5)
    assert abs(energy) < 1e3


def test_rank2_metric_flattens_polarizability():
    import numpy as np

    from enerzyme.tasks.metrics import build_single_metric

    metric = build_single_metric("mae")
    label = np.zeros((2, 3, 3))
    label[0, 0, 0] = 1.0
    pred = np.zeros_like(label)
    score = metric(
        {"polarizability": [label]},
        {"polarizability": [pred]},
        "polarizability",
    )
    # One entry of 1 among 2 molecules x 9 components. sklearn's default
    # multioutput average is that mean absolute error, 1/18.
    assert abs(score - (1.0 / 18.0)) < 1e-8


def test_shared_filters_and_interactions_reuse_one_block():
    from enerzyme.models.painn.core import PaiNNCore

    torch.manual_seed(4)
    core = PaiNNCore(
        dim_embedding=4,
        num_rbf=3,
        num_interactions=3,
        cutoff_sr=5.0,
        shared_filters=True,
        shared_interactions=True,
    )
    assert len(core.interactions) == 1
    assert len(core.mixings) == 1
    assert core.filter_net.dim_feature_out == 12
    n = 4
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    dij = torch.linalg.vector_norm(vij, dim=-1)
    out = core.get_output(
        atom_embedding=torch.randn(n, 4),
        rbf=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=dij,
    )
    assert torch.isfinite(out["atom_feature"]).all()
    assert torch.isfinite(out["atom_vector_feature"]).all()
    assert out["atom_vector_feature"].shape == (n, 3, 4)
