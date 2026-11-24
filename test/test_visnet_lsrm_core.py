"""ViSNet-LSRM Core: registration, fragments, equivariance, and forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "visnet_lsrm_layers_example.yaml"


def _edges(n: int, groups=None):
    idx_i, idx_j = [], []
    if groups is None:
        groups = [list(range(n))]
    for group in groups:
        for i in group:
            for j in group:
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


def _tiny_layers(**core_overrides):
    params = {
        "num_layers": 2,
        "long_layers": 1,
        "num_heads": 2,
        "lmax": 1,
        "vertex_type": "none",
        "vecnorm_type": "max_min",
        "neighbor_embedding": True,
        "include_self": False,
        "fragment_mode": "precomputed",
        "center": "atomic_number",
        "min_nodes_per_group": 2,
        "dropout": 0.0,
        "trainable_rbf_lr": False,
    }
    params.update(core_overrides)
    return [
        {"name": "RangeSeparation"},
        {"name": "ExpNormalSmearing", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": params},
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
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


def _build(dipole: bool = False, **core_overrides):
    from enerzyme.models.ff import build_model

    layers = _tiny_layers(**core_overrides)
    if dipole:
        layers.insert(
            -2,
            {"name": "EquivariantDipole", "params": {"n_layers": 1, "correct_charges": True}},
        )
    return build_model(
        "visnet_lsrm",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_lr": 9.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    )


def _forward(model, pos, za, batch, cluster_ids, edges=None):
    n = pos.shape[0]
    if edges is None:
        idx_i, idx_j = _edges(n)
    else:
        idx_i, idx_j = edges
    return model(
        {
            "Ra": pos,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch,
            "cluster_ids": cluster_ids,
        }
    )


def test_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout
    from enerzyme.models.visnet import PUBLISHED_MD22_CORE_PARAMS

    for name in ("visnet_lsrm", "visnet-lsrm", "visnet_lsr", "ViSNet-LSR"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "visnet_lsrm"
        assert build_params["dim_embedding"] == 128
        assert build_params["num_rbf"] == 50
        assert build_params["cutoff_sr"] == 4.0
        assert build_params["cutoff_lr"] == 9.0
        assert layers[3]["params"]["num_layers"] == 6
        assert layers[3]["params"]["long_layers"] == 2
        assert layers[3]["params"]["lmax"] == 1
    assert PUBLISHED_MD22_CORE_PARAMS["include_self"] is False
    assert PUBLISHED_MD22_CORE_PARAMS["vecnorm_type"] == "max_min"
    assert PUBLISHED_MD22_CORE_PARAMS["center"] == "atomic_number"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.architecture == "visnet_lsrm"
    assert model.dim_feature_out == 16
    assert model.feature_irreps == "16x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.dim_feature_in == 16
    assert readout.head[0].dim_feature_out == 8


def test_precomputed_shapes_and_lmax2():
    from enerzyme.models.presets.cartesian_vector import visnet_lsrm

    torch.manual_seed(0)
    n = 4
    core = visnet_lsrm(
        dim_embedding=8,
        num_rbf=4,
        num_layers=2,
        long_layers=1,
        num_heads=2,
        lmax=2,
        max_Za=10,
        fragment_mode="precomputed",
        neighbor_embedding=True,
        include_self=False,
        trainable_rbf_lr=False,
    )
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    out = core.get_output(
        atom_embedding=torch.randn(n, 8),
        rbf=torch.rand(idx_i.numel(), 4),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=torch.linalg.vector_norm(vij, dim=-1),
        cutoff_values_sr=torch.ones(idx_i.numel()),
        Za=torch.tensor([6, 1, 8, 1]),
        Ra=pos,
        cluster_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert out["atom_feature"].shape == (n, 16)
    assert out["atom_vector_feature"].shape == (n, 3, 16)
    assert out["atom_l2_feature"].shape == (n, 5, 8)
    assert torch.isfinite(out["atom_feature"]).all()


def test_long_layers_zero_still_concatenates():
    from enerzyme.models.presets.cartesian_vector import visnet_lsrm

    torch.manual_seed(1)
    core = visnet_lsrm(
        dim_embedding=4,
        num_rbf=2,
        num_layers=1,
        long_layers=0,
        num_heads=2,
        lmax=1,
        max_Za=8,
        neighbor_embedding=False,
    )
    out = core.get_output(
        atom_embedding=torch.randn(2, 4),
        rbf=torch.zeros(0, 2),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
        Za=torch.tensor([1, 6]),
    )
    assert out["atom_feature"].shape == (2, 8)
    assert out["atom_vector_feature"].shape == (2, 3, 8)
    assert torch.isfinite(out["atom_feature"]).all()


def test_kmeans_and_spectral_run():
    torch.manual_seed(2)
    model = _build(fragment_mode="kmeans", min_nodes_per_group=2).eval()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.1, 0.9, 0.1],
            [0.1, 1.0, 0.2],
            [2.2, 0.2, 0.1],
            [2.0, 1.1, 0.3],
        ],
        requires_grad=True,
    )
    za = torch.tensor([6, 1, 8, 1, 7, 1])
    batch = torch.zeros(6, dtype=torch.long)
    out = _forward(model, pos, za, batch, cluster_ids=None)
    assert out["E"].shape == (1,)
    assert torch.isfinite(out["Fa"]).all()

    spectral = _build(fragment_mode="spectral", min_nodes_per_group=3).eval()
    spectral_out = _forward(spectral, pos, za, batch, cluster_ids=None)
    assert torch.isfinite(spectral_out["E"]).all()
    assert not torch.allclose(out["E"], spectral_out["E"], atol=1e-6, rtol=1e-6)


def test_batch_isolation_with_overlapping_coordinates():
    torch.manual_seed(3)
    model = _build().eval()
    left = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.2, 0.1], [0.2, 1.0, 0.0]],
        requires_grad=True,
    )
    right = (left.detach() + torch.tensor([0.05, -0.02, 0.03])).requires_grad_(True)
    za_left = torch.tensor([8, 1, 6])
    za_right = torch.tensor([7, 1, 6])
    ids_left = torch.tensor([0, 0, 1])
    ids_right = torch.tensor([0, 1, 1])
    batch_one = torch.zeros(3, dtype=torch.long)
    edges_one = _edges(3)
    alone_left = _forward(model, left, za_left, batch_one, ids_left, edges=edges_one)
    alone_right = _forward(model, right, za_right, batch_one, ids_right, edges=edges_one)

    pos = torch.cat((left, right), dim=0)
    za = torch.cat((za_left, za_right))
    ids = torch.cat((ids_left, ids_right))
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    edges = _edges(6, groups=[[0, 1, 2], [3, 4, 5]])
    together = _forward(model, pos, za, batch, ids, edges=edges)
    assert torch.allclose(together["E"][0], alone_left["E"][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(together["E"][1], alone_right["E"][0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(together["Fa"][:3], alone_left["Fa"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(together["Fa"][3:], alone_right["Fa"], atol=1e-5, rtol=1e-5)


def test_energy_force_rotation_translation_and_dipole():
    torch.manual_seed(4)
    model = _build(dipole=True, lmax=1).double().eval()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.15, 0.05],
            [0.1, 0.85, 0.2],
            [0.4, 0.2, 0.95],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([8, 1, 6, 1])
    batch = torch.zeros(4, dtype=torch.long)
    cluster_ids = torch.tensor([0, 0, 1, 1])
    out = _forward(model, pos, za, batch, cluster_ids)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert out["M2"].shape == (1, 3)
    assert out["atom_feature"].shape[-1] == 16
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shift = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    moved = _forward(model, (pos.detach() + shift).requires_grad_(True), za, batch, cluster_ids)
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(5).double()
    rotated = _forward(
        model, (pos.detach() @ rotation).requires_grad_(True), za, batch, cluster_ids
    )
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["M2"] @ rotation, rotated["M2"], atol=1e-6, rtol=1e-6)

    inverted = _forward(model, (-pos.detach()).requires_grad_(True), za, batch, cluster_ids)
    assert torch.allclose(out["E"], inverted["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(-out["Fa"], inverted["Fa"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), za, batch, cluster_ids)["E"][0]
            e_minus = _forward(model, minus.requires_grad_(True), za, batch, cluster_ids)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-4, rtol=1e-4)
