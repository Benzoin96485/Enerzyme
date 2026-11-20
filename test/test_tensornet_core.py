"""TensorNet Core: registration, equivariance, charge, energy, and forces."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "tensornet_layers_example.yaml"


def _edges(n: int, self_pairs: bool = False):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j or self_pairs:
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


def _rotate_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    # Row positions transform as p @ R, so the 3x3 matrix transforms as R.T X R.
    return torch.einsum("ia,nacf,cj->nijf", rotation.T, tensor, rotation)


def _rotate_channels(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    # ``[N, 3, F]`` stores the spatial axis in the middle, same as PaiNN.
    return torch.einsum("ncf,cd->ndf", vector, rotation)


def _graph(n: int, features: int, num_rbf: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    idx_i, idx_j = _edges(n, self_pairs=False)
    pos = torch.randn(n, 3, generator=generator)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    return {
        "atom_embedding": torch.randn(n, features, generator=generator),
        "rbf": torch.rand(idx_i.numel(), num_rbf, generator=generator),
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "vij_sr": vij,
        "Dij_sr": dist,
        "cutoff_values_sr": torch.rand(idx_i.numel(), generator=generator).clamp(0.2, 1),
    }


def _tiny_layers(dipole: bool = False):
    layers = [
        {"name": "RangeSeparation"},
        {"name": "ExpNormalSmearing", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {"num_interactions": 1, "include_self": True}},
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
    if dipole:
        layers.append(
            {
                "name": "EquivariantDipole",
                "params": {"n_layers": 1, "correct_charges": True},
            }
        )
    layers.extend([{"name": "EnergyReduce"}, {"name": "Force"}])
    return layers


def _forward(model, pos, za, batch_seg, charge=None, offsets=None, edges=None):
    n = pos.shape[0]
    if edges is None:
        idx_i, idx_j = _edges(n)
    else:
        idx_i, idx_j = edges
    net = {
        "Ra": pos,
        "Za": za,
        "idx_i": idx_i,
        "idx_j": idx_j,
        "batch_seg": batch_seg,
    }
    if charge is not None:
        net["Q"] = charge
    if offsets is not None:
        net["offsets"] = offsets
    return model(net)


def test_tensornet_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("tensornet", "TensorNet"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "TensorNetCore"
        assert build_params["num_rbf"] == 32
        assert build_params["cutoff_sr"] == 4.5
        assert build_params["cutoff_fn"] == "cosine"
        assert layers[1]["name"] == "ExpNormalSmearing"
        assert layers[3]["params"]["equivariance_group"] == "O(3)"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "TensorNetCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.vector_feature_layout == "cartesian_n3f"
    assert model.equivariance_group == "O(3)"
    assert model._radial_holder[0].__class__.__name__ == "ExpNormalSmearing"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "two_layer"
    assert readout.head[0].dim_feature_out == 4


def test_unknown_group_and_cutoff_mismatch_raise():
    from enerzyme.models.ff import build_model
    from enerzyme.models.tensornet import TensorNetCore

    try:
        TensorNetCore(dim_embedding=4, num_rbf=3, equivariance_group="SU(2)")
    except ValueError as exc:
        assert "O(3)" in str(exc)
    else:
        raise AssertionError("expected an unknown group to fail")

    try:
        build_model(
            "tensornet",
            layer_params=[
                {"name": "ExpNormalSmearing", "params": {"cutoff_sr": 5.0}},
                {"name": "Core"},
            ],
            build_params={"dim_embedding": 4, "num_rbf": 4, "cutoff_sr": 4.5, "max_Za": 8},
            verbose=0,
        )
    except ValueError as exc:
        assert "cutoff" in str(exc)
    else:
        raise AssertionError("expected a radial cutoff mismatch to fail")


def test_embedding_and_interaction_match_reference_wiring():
    from enerzyme.models.functional import reduce_edge
    from enerzyme.models.tensornet.interaction import TensorNetEmbedding, TensorNetInteraction
    from enerzyme.models.tensornet.ops import (
        compose_tensor,
        decompose_tensor,
        frobenius_squared,
        skew_tensor_to_vector,
        symmetric_traceless,
        tensor_anticommutator,
        tensor_matmul,
        vector_to_skew_tensor,
    )

    torch.manual_seed(4)
    n, features, num_rbf = 5, 4, 3
    embedding = TensorNetEmbedding(features, num_rbf)
    graph = _graph(n, features, num_rbf, seed=4)
    hat = graph["vij_sr"] / graph["Dij_sr"].unsqueeze(-1)
    window = graph["cutoff_values_sr"]
    got = embedding(
        graph["atom_embedding"],
        graph["idx_i_sr"],
        graph["idx_j_sr"],
        hat,
        graph["rbf"],
        window,
    )

    pair = embedding.pair_mix(
        torch.cat(
            (
                graph["atom_embedding"][graph["idx_i_sr"]],
                graph["atom_embedding"][graph["idx_j_sr"]],
            ),
            dim=-1,
        )
    )
    radial = torch.stack(
        (
            embedding.distance_proj1(graph["rbf"]),
            embedding.distance_proj2(graph["rbf"]),
            embedding.distance_proj3(graph["rbf"]),
        ),
        dim=1,
    )
    processed = window[:, None, None] * pair[:, None, :] * radial
    iso = reduce_edge(processed[:, 0], graph["idx_i_sr"], (n, features))
    vector = reduce_edge(
        processed[:, 1, None, :] * hat[:, :, None],
        graph["idx_i_sr"],
        (n, 3, features),
    )
    outer = hat[:, :, None] * hat[:, None, :]
    symmetric = reduce_edge(
        processed[:, 2, None, None, :] * outer[:, :, :, None],
        graph["idx_i_sr"],
        (n, 3, 3, features),
    )
    skew = vector_to_skew_tensor(vector)
    traceless = symmetric_traceless(symmetric)
    features_t = compose_tensor(iso, skew, traceless)
    gates = embedding.init_norm(frobenius_squared(features_t))
    for linear in embedding.linears_scalar:
        gates = F.silu(linear(gates))
    gates = gates.reshape(n, 3, features)
    iso = embedding.linears_tensor[0](iso) * gates[:, 0]
    skew = embedding.linears_tensor[1](skew) * gates[:, 1, None, None]
    traceless = embedding.linears_tensor[2](traceless) * gates[:, 2, None, None]
    expected = compose_tensor(iso, skew, traceless)
    assert torch.allclose(got, expected, atol=1e-6)

    interaction = TensorNetInteraction(features, num_rbf, "O(3)")
    charge = torch.tensor([1.0, 0.0, -1.0, 0.5, 2.0])
    updated = interaction(
        expected,
        graph["idx_i_sr"],
        graph["idx_j_sr"],
        graph["rbf"],
        window,
        1 + 0.1 * charge,
    )
    edge = graph["rbf"]
    for linear in interaction.linears_scalar:
        edge = F.silu(linear(edge))
    edge = (edge * window.unsqueeze(-1)).reshape(-1, 3, features)
    normalized = expected / (frobenius_squared(expected) + 1)[:, None, None, :]
    iso_n, skew_n, traceless_n = decompose_tensor(normalized)
    iso_n = interaction.linears_tensor[0](iso_n)
    skew_n = interaction.linears_tensor[1](skew_n)
    traceless_n = interaction.linears_tensor[2](traceless_n)
    node = compose_tensor(iso_n, skew_n, traceless_n)
    vec = skew_tensor_to_vector(skew_n)
    iso_m = reduce_edge(
        edge[:, 0] * iso_n[graph["idx_j_sr"]], graph["idx_i_sr"], (n, features)
    )
    vec_m = reduce_edge(
        edge[:, 1, None, :] * vec[graph["idx_j_sr"]],
        graph["idx_i_sr"],
        (n, 3, features),
    )
    traceless_m = reduce_edge(
        edge[:, 2, None, None, :] * traceless_n[graph["idx_j_sr"]],
        graph["idx_i_sr"],
        (n, 3, 3, features),
    )
    message = compose_tensor(iso_m, vector_to_skew_tensor(vec_m), traceless_m)
    scale = (1 + 0.1 * charge)[:, None, None, None]
    mixed = scale * tensor_anticommutator(node, message)
    iso_c, skew_c, traceless_c = decompose_tensor(mixed)
    norm = frobenius_squared(mixed) + 1
    delta = compose_tensor(
        interaction.linears_tensor[3](iso_c / norm),
        interaction.linears_tensor[4](skew_c / norm[:, None, None, :]),
        interaction.linears_tensor[5](traceless_c / norm[:, None, None, :]),
    )
    reference = normalized + delta + scale * tensor_matmul(delta, delta)
    assert torch.allclose(updated, reference, atol=1e-5)


def test_geometry_self_pairs_charge_and_batch():
    from enerzyme.models.layers.rbf import ExpNormalSmearing
    from enerzyme.models.tensornet import TensorNetCore

    torch.manual_seed(5)
    n, features, num_rbf = 4, 6, 4
    core = TensorNetCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_interactions=2,
        cutoff_sr=4.5,
        include_self=True,
    ).double()
    radial = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=4.5, trainable=False).double()
    core._radial_holder = [radial]
    core.eval()
    graph = {
        key: value.double() if value.is_floating_point() else value
        for key, value in _graph(n, features, num_rbf, 5).items()
    }
    out = core.get_output(**graph)
    assert out["atom_feature"].shape == (n, features)
    assert out["atom_vector_feature"].shape == (n, 3, features)
    assert out["atom_tensor_feature"].shape == (n, 3, 3, features)

    rotation = _rotation(2).double()
    rotated = dict(graph)
    rotated["vij_sr"] = graph["vij_sr"] @ rotation
    rotated_out = core.get_output(**rotated)
    assert torch.allclose(out["atom_feature"], rotated_out["atom_feature"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        _rotate_channels(out["atom_vector_feature"], rotation),
        rotated_out["atom_vector_feature"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        _rotate_tensor(out["atom_tensor_feature"], rotation),
        rotated_out["atom_tensor_feature"],
        atol=1e-6,
        rtol=1e-6,
    )
    reflected = dict(graph)
    reflected["vij_sr"] = -graph["vij_sr"]
    reflected_out = core.get_output(**reflected)
    assert torch.allclose(out["atom_feature"], reflected_out["atom_feature"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        -out["atom_vector_feature"],
        reflected_out["atom_vector_feature"],
        atol=1e-6,
        rtol=1e-6,
    )

    so3 = TensorNetCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_interactions=2,
        cutoff_sr=4.5,
        include_self=False,
        equivariance_group="so3",
    ).double()
    # Default init keeps the skew part small, so YM+MY and 2YM are close.
    # Larger tensor maps make the parity-breaking product visible.
    with torch.no_grad():
        for layer in core.interactions:
            for linear in layer.linears_tensor:
                linear.weight.mul_(5)
    so3.load_state_dict(core.state_dict())
    plain = core.get_output(**graph)
    core.include_self = False
    o3 = core.get_output(**graph)
    so3_out = so3.get_output(**graph)
    assert (o3["atom_tensor_feature"] - so3_out["atom_tensor_feature"]).abs().max() > 1e-3
    assert (o3["atom_feature"] - so3_out["atom_feature"]).abs().max() > 1e-4
    core.include_self = True
    assert torch.allclose(plain["atom_feature"], core.get_output(**graph)["atom_feature"])

    order = torch.randperm(graph["idx_i_sr"].numel())
    permuted = dict(graph)
    for key in ("rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "cutoff_values_sr"):
        permuted[key] = graph[key][order]
    assert torch.allclose(
        core.get_output(**graph)["atom_tensor_feature"],
        core.get_output(**permuted)["atom_tensor_feature"],
        atol=1e-6,
        rtol=1e-6,
    )

    manual = dict(graph)
    atoms = torch.arange(n)
    zeros = torch.zeros(n, dtype=torch.float64)
    manual["idx_i_sr"] = torch.cat((graph["idx_i_sr"], atoms))
    manual["idx_j_sr"] = torch.cat((graph["idx_j_sr"], atoms))
    manual["vij_sr"] = torch.cat((graph["vij_sr"], torch.zeros(n, 3, dtype=torch.float64)))
    manual["Dij_sr"] = torch.cat((graph["Dij_sr"], zeros))
    manual["rbf"] = torch.cat((graph["rbf"], radial.get_rbf(zeros)))
    manual["cutoff_values_sr"] = torch.cat(
        (graph["cutoff_values_sr"], torch.ones(n, dtype=torch.float64))
    )
    core.include_self = False
    assert torch.allclose(
        plain["atom_feature"],
        core.get_output(**manual)["atom_feature"],
        atol=1e-6,
        rtol=1e-6,
    )
    core.include_self = True
    without_self = TensorNetCore(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_interactions=2,
        cutoff_sr=4.5,
        include_self=False,
    ).double()
    without_self.load_state_dict(core.state_dict())
    assert not torch.allclose(
        core.get_output(**graph)["atom_feature"],
        without_self.get_output(**graph)["atom_feature"],
        atol=1e-5,
        rtol=1e-5,
    )

    neutral = core.get_output(
        **graph, Q=torch.zeros(1), batch_seg=torch.zeros(n, dtype=torch.long)
    )
    omitted = core.get_output(**graph)
    assert torch.allclose(neutral["atom_feature"], omitted["atom_feature"], atol=1e-6, rtol=1e-6)
    charged = core.get_output(
        **graph, Q=torch.tensor([1.0]), batch_seg=torch.zeros(n, dtype=torch.long)
    )
    assert not torch.allclose(
        neutral["atom_feature"], charged["atom_feature"], atol=1e-5, rtol=1e-5
    )

    keep = torch.ones(graph["idx_i_sr"].numel(), dtype=torch.bool)
    keep[0] = False
    dropped = {
        key: value[keep] if key != "atom_embedding" else value for key, value in graph.items()
    }
    zero_window = dict(graph)
    zero_window["cutoff_values_sr"] = graph["cutoff_values_sr"].clone()
    zero_window["cutoff_values_sr"][0] = 0
    assert torch.allclose(
        core.get_output(**dropped)["atom_feature"],
        core.get_output(**zero_window)["atom_feature"],
        atol=1e-6,
        rtol=1e-6,
    )

    def _pack(count, seed):
        packed = _graph(count, features, num_rbf, seed)
        return {
            key: value.double() if torch.is_floating_point(value) else value
            for key, value in packed.items()
        }

    left = _pack(3, 7)
    right = _pack(4, 8)
    separate_left = core.get_output(**left)["atom_feature"]
    separate_right = core.get_output(**right)["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    joined = {
        "atom_embedding": torch.cat((left["atom_embedding"], right["atom_embedding"])),
        "rbf": torch.cat((left["rbf"], right["rbf"])),
        "idx_i_sr": torch.cat((left["idx_i_sr"], right["idx_i_sr"] + n_left)),
        "idx_j_sr": torch.cat((left["idx_j_sr"], right["idx_j_sr"] + n_left)),
        "vij_sr": torch.cat((left["vij_sr"], right["vij_sr"])),
        "Dij_sr": torch.cat((left["Dij_sr"], right["Dij_sr"])),
        "cutoff_values_sr": torch.cat((left["cutoff_values_sr"], right["cutoff_values_sr"])),
    }
    both = core.get_output(**joined)["atom_feature"]
    assert torch.allclose(both[:n_left], separate_left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], separate_right, atol=1e-6, rtol=1e-6)


def test_stack_energy_force_dipole_and_image():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = build_model(
        "tensornet",
        layer_params=_tiny_layers(dipole=True),
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 4.5,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    ).double()
    model.eval()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.15, 0.05],
            [0.1, 0.85, 0.2],
            [0.2, 0.1, 0.95],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([8, 1, 6, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    out = _forward(model, pos, za, batch, charge=torch.tensor([0.0]))
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert out["M2"].shape == (1, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shifted = (pos.detach() + torch.tensor([2.0, -1.0, 0.4])).requires_grad_(True)
    moved = _forward(model, shifted, za, batch, charge=torch.tensor([0.0]))
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(9).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, za, batch, charge=torch.tensor([0.0]))
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["M2"] @ rotation, rotated["M2"], atol=1e-6, rtol=1e-6)

    inverted = _forward(
        model, (-pos.detach()).requires_grad_(True), za, batch, charge=torch.tensor([0.0])
    )
    assert torch.allclose(out["E"], inverted["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(-out["Fa"], inverted["Fa"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(-out["M2"], inverted["M2"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), za, batch)["E"][0]
            e_minus = _forward(model, minus.requires_grad_(True), za, batch)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)

    far = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    species = torch.tensor([6, 8], dtype=torch.long)
    pair = (
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
    )
    isolated = _forward(model, far, species, torch.zeros(2, dtype=torch.long), edges=pair)
    image = _forward(
        model,
        far,
        species,
        torch.zeros(2, dtype=torch.long),
        edges=pair,
        offsets=torch.tensor([[-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=torch.float64),
    )
    assert torch.isfinite(isolated["E"]).all()
    assert not torch.allclose(isolated["E"], image["E"], atol=1e-6, rtol=1e-6)
