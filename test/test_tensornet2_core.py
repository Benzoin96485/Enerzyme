"""TensorNet2 Core: registration, charge conservation, equivariance, forces."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "tensornet2_layers_example.yaml"


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
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _rotate_tensor(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ia,nacf,cj->nijf", rotation.T, tensor, rotation)


def _rotate_channels(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ncf,cd->ndf", vector, rotation)


def _graph(n: int, features: int, num_rbf: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    idx_i, idx_j = _edges(n)
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


def _block_sums(channels: torch.Tensor, q_dim: int, batch: torch.Tensor, n_mol: int):
    sums = []
    for start in range(0, channels.shape[-1], q_dim):
        block = channels[:, start : start + q_dim]
        total = torch.zeros(n_mol, q_dim, dtype=channels.dtype)
        total.index_add_(0, batch, block)
        sums.append(total)
    return sums


def test_tensornet2_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.electrostatics import WeightedCoulombEnergyLayer
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("tensornet2", "tensornet-2"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "TensorNet2Core"
        assert build_params["q_dim"] == 16
        assert build_params["cutoff_sr"] == 4.5
        assert layers[-3]["name"] == "WeightedCoulombEnergy"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], dim_embedding=8, num_rbf=4, max_Za=20, q_dim=2),
        verbose=0,
    )
    assert model.__class__.__name__ == "TensorNet2Core"
    assert model.q_dim == 2
    assert model.charge_channels_dim == 6
    assert model.charge_scale == 0.0
    assert len(model.charge_heads) == 3
    assert model.interactions[0].__class__.__name__ == "TensorNet2Interaction"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    coulomb = next(
        layer for layer in model.post_sequence if isinstance(layer, WeightedCoulombEnergyLayer)
    )
    assert readout.head_type == "two_layer"
    assert coulomb.layer_weights == [1.0, 2.0, 3.0]
    assert coulomb.pairs == "all_to_all"

    try:
        from enerzyme.models.tensornet import TensorNet2Core

        TensorNet2Core(dim_embedding=4, num_rbf=3, q_dim=0)
    except ValueError as exc:
        assert "q_dim" in str(exc)
    else:
        raise AssertionError("expected q_dim 0 to fail")

    try:
        build_model(
            "tensornet2",
            build_params={
                "dim_embedding": 4,
                "num_rbf": 3,
                "max_Za": 8,
                "cutoff_sr": 4.5,
                "q_dim": 2,
            },
            layer_params=[
                {"name": "Core", "params": {"num_interactions": 1, "include_self": False}},
                {
                    "name": "WeightedCoulombEnergy",
                    "params": {"layer_weights": [1.0, 2.0, 3.0]},
                },
            ],
            verbose=0,
        )
    except ValueError as exc:
        assert "layer_weights" in str(exc)
        assert "4 charge channels" in str(exc)
    else:
        raise AssertionError("expected mismatched layer_weights to fail at build")


def test_per_atom_charge_must_be_constant_in_a_molecule():
    from enerzyme.models.tensornet import TensorNet2Core

    core = TensorNet2Core(dim_embedding=4, num_rbf=2, q_dim=1, num_interactions=1)
    like = torch.zeros(4)
    batch = torch.tensor([0, 0, 1, 1])
    same = core.resolve_total_charge(
        4, torch.tensor([1.5, 1.5, -2.0, -2.0]), batch, like
    )
    assert torch.equal(same, torch.tensor([1.5, 1.5, -2.0, -2.0]))
    try:
        core.resolve_total_charge(4, torch.tensor([1.5, 0.5, -2.0, -2.0]), batch, like)
    except ValueError as exc:
        assert "molecular total" in str(exc)
    else:
        raise AssertionError("expected inconsistent per-atom Q to fail")


def test_charge_conservation_equivariance_and_batch():
    from enerzyme.models.layers.rbf import ExpNormalSmearing
    from enerzyme.models.tensornet import TensorNet2Core

    torch.manual_seed(3)
    n, features, num_rbf, q_dim = 4, 5, 3, 2
    core = TensorNet2Core(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_interactions=2,
        cutoff_sr=4.5,
        q_dim=q_dim,
        include_self=True,
    ).double()
    radial = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=4.5, trainable=False).double()
    core._radial_holder = [radial]
    core.eval()
    graph = {
        key: value.double() if value.is_floating_point() else value
        for key, value in _graph(n, features, num_rbf, 3).items()
    }
    batch = torch.zeros(n, dtype=torch.long)
    out = core.get_output(**graph, Q=torch.tensor([1.5]), batch_seg=batch)
    assert out["charge_channels"].shape == (n, 6)
    for block in _block_sums(out["charge_channels"], q_dim, batch, 1):
        assert torch.allclose(block, block.new_full((1, q_dim), 1.5), atol=1e-3)

    neutral = core.get_output(**graph)
    for block in _block_sums(neutral["charge_channels"], q_dim, batch, 1):
        assert torch.allclose(block, block.new_zeros(1, q_dim), atol=1e-3)
    assert not torch.allclose(out["atom_feature"], neutral["atom_feature"], atol=1e-5)

    rotation = _rotation(4).double()
    rotated = dict(graph)
    rotated["vij_sr"] = graph["vij_sr"] @ rotation
    rotated_out = core.get_output(**rotated, Q=torch.tensor([1.5]), batch_seg=batch)
    assert torch.allclose(out["atom_feature"], rotated_out["atom_feature"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["charge_channels"], rotated_out["charge_channels"], atol=1e-6, rtol=1e-6)
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
    reflected_out = core.get_output(**reflected, Q=torch.tensor([1.5]), batch_seg=batch)
    assert torch.allclose(out["atom_feature"], reflected_out["atom_feature"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["charge_channels"], reflected_out["charge_channels"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        -out["atom_vector_feature"],
        reflected_out["atom_vector_feature"],
        atol=1e-6,
        rtol=1e-6,
    )

    so3 = TensorNet2Core(
        dim_embedding=features,
        num_rbf=num_rbf,
        num_interactions=2,
        cutoff_sr=4.5,
        q_dim=q_dim,
        include_self=False,
        equivariance_group="SO(3)",
    ).double()
    with torch.no_grad():
        for layer in core.interactions:
            for linear in layer.linears_tensor:
                linear.weight.mul_(5)
    so3.load_state_dict(core.state_dict())
    so3._radial_holder = [radial]
    core.include_self = False
    o3 = core.get_output(**graph, Q=torch.tensor([1.5]), batch_seg=batch)
    so3_out = so3.get_output(**graph, Q=torch.tensor([1.5]), batch_seg=batch)
    assert (o3["atom_tensor_feature"] - so3_out["atom_tensor_feature"]).abs().max() > 1e-3
    assert (o3["atom_feature"] - so3_out["atom_feature"]).abs().max() > 1e-4
    core.include_self = True

    order = torch.randperm(graph["idx_i_sr"].numel())
    permuted = dict(graph)
    for key in ("rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "Dij_sr", "cutoff_values_sr"):
        permuted[key] = graph[key][order]
    assert torch.allclose(
        core.get_output(**graph)["charge_channels"],
        core.get_output(**permuted)["charge_channels"],
        atol=1e-6,
        rtol=1e-6,
    )

    left = _graph(3, features, num_rbf, 7)
    right = _graph(4, features, num_rbf, 8)
    left = {key: value.double() if value.is_floating_point() else value for key, value in left.items()}
    right = {key: value.double() if value.is_floating_point() else value for key, value in right.items()}
    n_left = left["atom_embedding"].shape[0]
    separate_left = core.get_output(
        **left, Q=torch.tensor([1.0]), batch_seg=torch.zeros(n_left, dtype=torch.long)
    )
    separate_right = core.get_output(
        **right, Q=torch.tensor([-2.0]), batch_seg=torch.zeros(4, dtype=torch.long)
    )
    joined = {
        "atom_embedding": torch.cat((left["atom_embedding"], right["atom_embedding"])),
        "rbf": torch.cat((left["rbf"], right["rbf"])),
        "idx_i_sr": torch.cat((left["idx_i_sr"], right["idx_i_sr"] + n_left)),
        "idx_j_sr": torch.cat((left["idx_j_sr"], right["idx_j_sr"] + n_left)),
        "vij_sr": torch.cat((left["vij_sr"], right["vij_sr"])),
        "Dij_sr": torch.cat((left["Dij_sr"], right["Dij_sr"])),
        "cutoff_values_sr": torch.cat((left["cutoff_values_sr"], right["cutoff_values_sr"])),
    }
    batch_joined = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    both = core.get_output(**joined, Q=torch.tensor([1.0, -2.0]), batch_seg=batch_joined)
    assert torch.allclose(both["atom_feature"][:n_left], separate_left["atom_feature"], atol=1e-6)
    assert torch.allclose(both["atom_feature"][n_left:], separate_right["atom_feature"], atol=1e-6)
    assert torch.allclose(
        both["charge_channels"][:n_left], separate_left["charge_channels"], atol=1e-6
    )
    sums = _block_sums(both["charge_channels"], q_dim, batch_joined, 2)
    assert torch.allclose(sums[0][0], sums[0].new_full((q_dim,), 1.0), atol=1e-3)
    assert torch.allclose(sums[0][1], sums[0].new_full((q_dim,), -2.0), atol=1e-3)


def test_stack_energy_force_translation_and_image():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = build_model(
        "tensornet2",
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 4.5,
            "cutoff_fn": "cosine",
            "q_dim": 2,
        },
        layer_params=[
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
            {"name": "WeightedCoulombEnergy", "params": {"layer_weights": [1.0, 2.0]}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
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
    charge = torch.tensor([1.0])
    out = _forward(model, pos, za, batch, charge=charge)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert out["charge_channels"].shape == (4, 4)
    assert torch.allclose(
        out["charge_channels"].sum(0),
        out["charge_channels"].new_full((4,), 1.0),
        atol=1e-3,
    )
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shifted = (pos.detach() + torch.tensor([2.0, -1.0, 0.4])).requires_grad_(True)
    moved = _forward(model, shifted, za, batch, charge=charge)
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(9).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, za, batch, charge=charge)
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), za, batch, charge=charge)["E"][0]
            e_minus = _forward(model, minus.requires_grad_(True), za, batch, charge=charge)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)

    far = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    species = torch.tensor([6, 8], dtype=torch.long)
    pair = (torch.tensor([0, 1], dtype=torch.long), torch.tensor([1, 0], dtype=torch.long))
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
