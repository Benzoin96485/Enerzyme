"""NewtonNet Core: registration, equivariance, forces, and the direction loss."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "newtonnet_layers_example.yaml"


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
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    if improper:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _rotate_vectors(vectors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ncf,cd->ndf", vectors, rotation)


def _symmetric_rbf(idx_i, idx_j, n_rbf):
    """Same radial basis on (i, j) and (j, i), as a distance basis would."""
    rbf = torch.empty(idx_i.numel(), n_rbf)
    seen = {}
    for edge, (i, j) in enumerate(zip(idx_i.tolist(), idx_j.tolist())):
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen[key] = torch.randn(n_rbf)
        rbf[edge] = seen[key]
    return rbf


def _graph(n, f, n_rbf, seed=0):
    torch.manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    vij = pos[idx_j] - pos[idx_i]
    return {
        "atom_embedding": torch.randn(n, f),
        "rbf": _symmetric_rbf(idx_i, idx_j, n_rbf),
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "vij_sr": vij,
        "Dij_sr": torch.linalg.vector_norm(vij, dim=-1),
        "pos": pos,
    }


def _run(core, pos, embedding, rbf, idx_i, idx_j):
    vij = pos[idx_j] - pos[idx_i]
    return core.get_output(
        atom_embedding=embedding,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
        Dij_sr=torch.linalg.vector_norm(vij, dim=-1),
    )


def test_newtonnet_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout
    from enerzyme.models.loss import LOSS_REGISTER

    for name in ("newtonnet", "NewtonNet"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "NewtonNetCore"
        assert build_params["num_rbf"] == 20
        assert build_params["cutoff_sr"] == 5.0
        assert layers[1]["params"]["flavor"] == "default"
        assert layers[3]["params"]["variant"] == "paper"
        assert layers[3]["params"]["envelope_p"] == 7

    assert "latent_force_cosine" in LOSS_REGISTER

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "NewtonNetCore"
    assert model.variant == "paper"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "mlp"
    assert len(readout.head) == 3


def test_newtonnet_rejects_a_second_cutoff_on_the_basis():
    from enerzyme.models.ff import build_model

    layers = _tiny_layers()
    layers[1]["params"]["apply_cutoff_fn"] = True
    layers[1]["params"]["cutoff_fn"] = "polynomial"
    try:
        build_model(
            "newtonnet",
            layer_params=layers,
            build_params={
                "dim_embedding": 4,
                "num_rbf": 3,
                "max_Za": 10,
                "cutoff_sr": 5.0,
            },
            verbose=0,
        )
    except ValueError as exc:
        assert "apply_cutoff_fn" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_newtonnet_rejects_bad_settings():
    from enerzyme.models.newtonnet import NewtonNetCore

    try:
        NewtonNetCore(dim_embedding=4, num_rbf=2, variant="schnet")
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        NewtonNetCore(dim_embedding=4, num_rbf=2, envelope_p=1)
    except ValueError as exc:
        assert "envelope_p" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_paper_geometry_and_third_law():
    from enerzyme.models.newtonnet import NewtonNetCore

    torch.manual_seed(0)
    n, f, n_rbf = 5, 8, 4
    core = NewtonNetCore(
        dim_embedding=f, num_rbf=n_rbf, num_interactions=3, cutoff_sr=5.0, variant="paper"
    )
    core.eval()
    graph = _graph(n, f, n_rbf, seed=1)
    idx_i, idx_j = graph["idx_i_sr"], graph["idx_j_sr"]
    embedding, rbf, pos = graph["atom_embedding"], graph["rbf"], graph["pos"]

    out = _run(core, pos, embedding, rbf, idx_i, idx_j)
    assert out["atom_feature"].shape == (n, f)
    assert out["atom_vector_feature"].shape == (n, 3, f)
    assert out["latent_force"].shape == (n, 3)
    assert torch.isfinite(out["atom_feature"]).all()
    # The physical latent force is not the channel sum of the force features.
    gap = (out["latent_force"] - out["atom_vector_feature"].sum(dim=-1)).abs().max()
    assert gap > 1e-3
    # Bidirectional edges: pair forces cancel.
    assert torch.allclose(out["latent_force"].sum(dim=0), torch.zeros(3), atol=1e-5)
    assert torch.allclose(
        out["atom_vector_feature"].sum(dim=0), torch.zeros(3, f), atol=1e-5
    )

    rotation = _rotation(2)
    rotated = _run(core, pos @ rotation, embedding, rbf, idx_i, idx_j)
    assert torch.allclose(out["atom_feature"], rotated["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _rotate_vectors(out["atom_vector_feature"], rotation),
        rotated["atom_vector_feature"],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(out["latent_force"] @ rotation, rotated["latent_force"], atol=1e-5, rtol=1e-5)

    reflection = _rotation(3, improper=True)
    assert torch.det(reflection) < 0
    reflected = _run(core, pos @ reflection, embedding, rbf, idx_i, idx_j)
    assert torch.allclose(out["atom_feature"], reflected["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        out["latent_force"] @ reflection, reflected["latent_force"], atol=1e-5, rtol=1e-5
    )

    shifted = _run(core, pos + torch.tensor([3.0, -2.0, 0.5]), embedding, rbf, idx_i, idx_j)
    assert torch.allclose(out["atom_feature"], shifted["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out["latent_force"], shifted["latent_force"], atol=1e-5, rtol=1e-5)

    alone = core.get_output(
        atom_embedding=embedding[:1],
        rbf=torch.zeros(0, n_rbf),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
        Dij_sr=torch.zeros(0),
    )
    assert torch.allclose(alone["atom_feature"], embedding[:1], atol=1e-6)
    assert torch.count_nonzero(alone["atom_vector_feature"]) == 0
    assert torch.count_nonzero(alone["latent_force"]) == 0


def test_envelope_drops_distant_edges_and_messages_stay_local():
    from enerzyme.models.newtonnet import NewtonNetCore

    torch.manual_seed(4)
    f, n_rbf = 6, 3
    core = NewtonNetCore(
        dim_embedding=f, num_rbf=n_rbf, num_interactions=2, cutoff_sr=5.0, variant="paper"
    )
    core.eval()
    graph = _graph(3, f, n_rbf, seed=5)
    base = core.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
        Dij_sr=graph["Dij_sr"],
    )
    far = torch.tensor([[0, 3], [3, 0]], dtype=torch.long)
    extra_i = torch.cat([graph["idx_i_sr"], far[:, 0]])
    extra_j = torch.cat([graph["idx_j_sr"], far[:, 1]])
    extra_vij = torch.cat([graph["vij_sr"], torch.tensor([[8.0, 0.0, 0.0], [-8.0, 0.0, 0.0]])])
    extra_rbf = torch.cat([graph["rbf"], torch.randn(2, n_rbf)])
    extra_emb = torch.cat([graph["atom_embedding"], torch.randn(1, f)])
    with_far = core.get_output(
        atom_embedding=extra_emb,
        rbf=extra_rbf,
        idx_i_sr=extra_i,
        idx_j_sr=extra_j,
        vij_sr=extra_vij,
        Dij_sr=torch.linalg.vector_norm(extra_vij, dim=-1),
    )
    assert torch.allclose(with_far["atom_feature"][:3], base["atom_feature"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(with_far["latent_force"][:3], base["latent_force"], atol=1e-5, rtol=1e-5)

    # A real neighbor inside the cutoff changes the receiver.
    dropped = core.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"][:1],
        idx_i_sr=graph["idx_i_sr"][:1],
        idx_j_sr=graph["idx_j_sr"][:1],
        vij_sr=graph["vij_sr"][:1],
        Dij_sr=graph["Dij_sr"][:1],
    )
    assert not torch.allclose(dropped["atom_feature"], base["atom_feature"], atol=1e-5)

    left = _graph(3, f, n_rbf, seed=6)
    right = _graph(4, f, n_rbf, seed=7)
    separate_l = core.get_output(
        atom_embedding=left["atom_embedding"],
        rbf=left["rbf"],
        idx_i_sr=left["idx_i_sr"],
        idx_j_sr=left["idx_j_sr"],
        vij_sr=left["vij_sr"],
        Dij_sr=left["Dij_sr"],
    )["atom_feature"]
    separate_r = core.get_output(
        atom_embedding=right["atom_embedding"],
        rbf=right["rbf"],
        idx_i_sr=right["idx_i_sr"],
        idx_j_sr=right["idx_j_sr"],
        vij_sr=right["vij_sr"],
        Dij_sr=right["Dij_sr"],
    )["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    joined = core.get_output(
        atom_embedding=torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        rbf=torch.cat([left["rbf"], right["rbf"]]),
        idx_i_sr=torch.cat([left["idx_i_sr"], right["idx_i_sr"] + n_left]),
        idx_j_sr=torch.cat([left["idx_j_sr"], right["idx_j_sr"] + n_left]),
        vij_sr=torch.cat([left["vij_sr"], right["vij_sr"]]),
        Dij_sr=torch.cat([left["Dij_sr"], right["Dij_sr"]]),
    )["atom_feature"]
    assert torch.allclose(joined[:n_left], separate_l, atol=1e-5, rtol=1e-5)
    assert torch.allclose(joined[n_left:], separate_r, atol=1e-5, rtol=1e-5)


def test_repo_variant_contracts_force_features():
    from enerzyme.models.newtonnet import NewtonNetCore

    torch.manual_seed(8)
    f, n_rbf = 5, 3
    core = NewtonNetCore(
        dim_embedding=f,
        num_rbf=n_rbf,
        num_interactions=3,
        cutoff_sr=5.0,
        variant="repo",
        shared_interactions=True,
        layer_norm=True,
    )
    core.eval()
    assert len(core.blocks) == 1
    assert len(core.norms) == 3
    graph = _graph(4, f, n_rbf, seed=9)
    out = core.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
        Dij_sr=graph["Dij_sr"],
    )
    assert torch.allclose(
        out["latent_force"], out["atom_vector_feature"].sum(dim=-1), atol=1e-6, rtol=1e-6
    )
    assert torch.isfinite(out["atom_feature"]).all()

    torch.manual_seed(8)
    plain = NewtonNetCore(
        dim_embedding=f, num_rbf=n_rbf, num_interactions=3, cutoff_sr=5.0, variant="repo"
    )
    plain.eval()
    plain_out = plain.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
        Dij_sr=graph["Dij_sr"],
    )
    # Same construction seed, but layer norm changes the scalar state.
    assert not torch.allclose(out["atom_feature"], plain_out["atom_feature"], atol=1e-5)

    one = NewtonNetCore(
        dim_embedding=f, num_rbf=n_rbf, num_interactions=1, cutoff_sr=5.0, variant="repo"
    )
    one.eval()
    first = one.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
        Dij_sr=graph["Dij_sr"],
    )
    assert torch.allclose(first["latent_force"].sum(dim=0), torch.zeros(3), atol=1e-5)


def test_latent_force_cosine_loss():
    from enerzyme.models.newtonnet.loss import LatentForceCosineLoss, latent_force_cosine

    direction = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert latent_force_cosine(direction, direction * 3).item() == 0.0
    opposite = latent_force_cosine(direction, -direction)
    assert abs(opposite.item() - 2.0) < 1e-6
    orthogonal = latent_force_cosine(
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert abs(orthogonal.item() - 1.0) < 1e-6
    zeros = latent_force_cosine(torch.zeros(2, 3), torch.zeros(2, 3))
    assert zeros.item() == 0.0

    loss = LatentForceCosineLoss(weight=4.0)
    value = loss({"latent_force": direction}, {"Fa": -direction})
    assert abs(value.item() - 8.0) < 1e-5
    try:
        loss({}, {"Fa": direction})
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def _tiny_layers(spectra: bool = False, variant: str = "paper"):
    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "default", "trainable": False, "apply_cutoff_fn": False},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {"num_interactions": 2, "variant": variant, "envelope_p": 7},
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "mlp",
                "num_hidden_layers": 2,
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
                {"name": "EquivariantDipole", "params": {"n_layers": 1}},
                {"name": "EquivariantPolarizability", "params": {"n_layers": 1}},
            ]
        )
    layers.extend([{"name": "EnergyReduce"}, {"name": "Force"}])
    return layers


def _forward_stack(model, pos, za, batch_seg):
    n = pos.shape[0]
    idx_i, idx_j = _edges(n)
    return model(
        {
            "Ra": pos,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
        }
    )


def test_newtonnet_stack_energy_force_and_cartesian_heads():
    from enerzyme.models.ff import build_model

    torch.manual_seed(4)
    model = build_model(
        "newtonnet",
        layer_params=_tiny_layers(spectra=True, variant="paper"),
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 5.0,
        },
        verbose=0,
    ).double()
    model.eval()
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
    out = _forward_stack(model, pos, za, batch)
    assert out["E"].shape == (2,)
    assert out["Fa"].shape == (5, 3)
    assert out["M2"].shape == (2, 3)
    assert out["polarizability"].shape == (2, 3, 3)
    assert out["latent_force"].shape == (5, 3)
    assert torch.isfinite(out["Fa"]).all()
    alpha = out["polarizability"]
    assert torch.allclose(alpha, alpha.transpose(-1, -2), atol=1e-8)

    neutral_pos = pos.detach().requires_grad_(True)
    base = _forward_stack(model, neutral_pos, za, batch)
    shifted = (pos.detach() + torch.tensor([4.0, -1.0, 2.0], dtype=torch.float64)).requires_grad_(True)
    moved = _forward_stack(model, shifted, za, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-6, rtol=1e-6)
    # Corrected charges sum to zero, so the dipole does not follow the origin.
    assert torch.allclose(base["M2"], moved["M2"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(5).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward_stack(model, rotated_pos, za, batch)
    assert torch.allclose(base["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(base["M2"] @ rotation, rotated["M2"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        base["latent_force"] @ rotation, rotated["latent_force"], atol=1e-6, rtol=1e-6
    )
    alpha_rot = torch.einsum("ab,nbc,cd->nad", rotation.T, base["polarizability"], rotation)
    assert torch.allclose(alpha_rot, rotated["polarizability"], atol=1e-6, rtol=1e-6)


def test_newtonnet_force_matches_energy_finite_difference():
    from enerzyme.models.ff import build_model

    for variant in ("paper", "repo"):
        torch.manual_seed(6)
        model = build_model(
            "newtonnet",
            layer_params=_tiny_layers(variant=variant),
            build_params={
                "dim_embedding": 4,
                "num_rbf": 3,
                "max_Za": 10,
                "cutoff_sr": 5.0,
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
        assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4), variant
