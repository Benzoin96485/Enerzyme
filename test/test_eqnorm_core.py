"""EQNorm registration, dual branch, equivariance, and conservative forces."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "eqnorm_layers_example.yaml"
UPSTREAM = Path("/tmp/eqnorm-src")


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


def _small_core(**overrides):
    from enerzyme.models.presets.irreps import eqnorm

    params = dict(
        dim_embedding=3,
        num_rbf=4,
        cutoff_sr=6.0,
        num_layers=2,
        lmax=1,
        hidden_irreps="4x0e+2x1o",
        num_features=4,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=2.5,
        normalization="constant",
        message_envelope=False,
    )
    params.update(overrides)
    return eqnorm(**params)


def test_registration_and_yaml_build():
    from enerzyme.models.mp.nn.mlp import VariancePreservingMLP
    from enerzyme.models.eqnorm import PUBLISHED_MPTRJ_CORE_PARAMS
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.atom_embedding import OneHotAtomEmbedding

    core_cls, build_params, layers = get_ff_core("eqnorm")
    assert core_cls.__name__ == "eqnorm"
    assert build_params["cutoff_fn"] == "nequip"
    assert build_params["cutoff_sr"] == 6.0
    assert layers[1]["params"]["flavor"] == "equflash"
    assert layers[1]["params"]["trainable"] is True
    assert layers[2]["name"] == "OneHotAtomEmbedding"
    assert layers[3]["params"]["avg_num_neighbors"] == pytest.approx(
        PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"]
    )
    assert layers[4]["params"]["head_type"] == "variance_preserving"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "ComposableCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.readout_hidden_width == 4
    assert model.num_layers == 2
    embedding = next(layer for layer in model.pre_sequence if isinstance(layer, OneHotAtomEmbedding))
    assert embedding.dim_embedding == 4
    readout = model.post_sequence[0]
    assert isinstance(readout.head, VariancePreservingMLP)
    assert readout.head.dims == [8, 4, 1]


def test_published_presets_select_the_reference_normalization():
    from enerzyme.models.eqnorm import (
        PUBLISHED_MAX_CORE_PARAMS,
        PUBLISHED_MPTRJ_CORE_PARAMS,
        PUBLISHED_OMAT_CORE_PARAMS,
    )

    assert PUBLISHED_MPTRJ_CORE_PARAMS["normalization"] == "constant"
    assert PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"] == pytest.approx(61.84)
    assert PUBLISHED_MPTRJ_CORE_PARAMS["message_envelope"] is False
    assert PUBLISHED_MAX_CORE_PARAMS["message_envelope"] is True
    assert PUBLISHED_OMAT_CORE_PARAMS["normalization"] == "degree"
    assert PUBLISHED_OMAT_CORE_PARAMS["avg_num_neighbors"] is None
    assert PUBLISHED_MPTRJ_CORE_PARAMS["hidden_irreps"] == "128x0e+64x1o+32x2e+32x3o"


def test_unreachable_and_gapped_irreps_are_rejected():
    with pytest.raises(ValueError, match="one irrep"):
        _small_core(hidden_irreps="4x0e+2x2e", lmax=2)
    with pytest.raises(ValueError, match="not reachable"):
        _small_core(hidden_irreps="4x0e+2x1e", lmax=1)
    with pytest.raises(ValueError, match="avg_num_neighbors"):
        _small_core(normalization="constant", avg_num_neighbors=None)
    with pytest.raises(ValueError, match="constant"):
        _small_core(normalization="post")


def test_scalar_branch_does_not_update_the_tensor_state():
    torch.manual_seed(0)
    core = _small_core(num_layers=1)
    core.eval()
    embedding = torch.randn(3, 3)
    idx_i, idx_j = _edges(3)
    vij = torch.randn(idx_i.numel(), 3)
    radial = torch.rand(idx_i.numel(), 4)
    edge_sh = core.embed.sph(vij)
    base = core.embed.projection(embedding)
    tensor, delta = core.blocks[0].module(base, idx_i, idx_j, radial, edge_sh, None)
    expected = base + F.silu(delta)
    got = core.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)

    tensor_again, delta_again = core.blocks[0].module(base, idx_i, idx_j, radial, edge_sh, None)
    with torch.no_grad():
        core.blocks[0].module.scalar_conv.linear_2.weight.add_(1.0)
    tensor_changed, delta_changed = core.blocks[0].module(base, idx_i, idx_j, radial, edge_sh, None)
    assert torch.allclose(tensor_again, tensor_changed, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(delta_again, delta_changed, atol=1e-4)


def test_degree_norm_matches_constant_norm_on_a_regular_graph():
    torch.manual_seed(1)
    shared = dict(
        dim_embedding=3,
        num_rbf=4,
        num_layers=1,
        hidden_irreps="4x0e+2x1o",
        num_features=4,
        invariant_layers=1,
        invariant_neurons=8,
        message_envelope=False,
    )
    degree = _small_core(**shared, normalization="degree", avg_num_neighbors=None)
    constant = _small_core(**shared, normalization="constant", avg_num_neighbors=2.0)
    constant.load_state_dict(degree.state_dict())
    embedding = torch.randn(3, 3)
    idx_i, idx_j = _edges(3)
    vij = torch.randn(idx_i.numel(), 3)
    radial = torch.rand(idx_i.numel(), 4)
    left = degree.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    right = constant.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)

    other = _small_core(**shared, normalization="constant", avg_num_neighbors=8.0)
    other.load_state_dict(degree.state_dict())
    shifted = other.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    assert not torch.allclose(left, shifted, atol=1e-4)


def test_message_envelope_kills_edges_outside_the_cutoff():
    torch.manual_seed(2)
    core = _small_core(num_layers=1, message_envelope=True, cutoff_sr=1.0, polynomial_p=6)
    core.eval()
    embedding = torch.eye(3)
    idx_i, idx_j = _edges(3)
    vij = torch.tensor(
        [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, -4.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, -5.0]]
    )
    radial = torch.rand(6, 4)
    outside = core.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    alone = []
    empty_i = torch.zeros(0, dtype=torch.long)
    empty_v = torch.zeros(0, 3)
    empty_r = torch.zeros(0, 4)
    for row in range(3):
        alone.append(
            core.get_output(embedding[row : row + 1], empty_r, empty_i, empty_i, empty_v)[
                "atom_feature"
            ]
        )
    assert torch.allclose(outside, torch.cat(alone, dim=0), atol=1e-5, rtol=1e-5)


def test_core_is_invariant_to_rotation_translation_and_edge_order():
    torch.manual_seed(3)
    core = _small_core()
    core.eval()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3]])
    embedding = torch.randn(3, 3)
    idx_i, idx_j = _edges(3)
    radial = torch.rand(idx_i.numel(), 4)

    def run(positions, order=None):
        vij = positions[idx_j] - positions[idx_i]
        if order is None:
            return core.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
        return core.get_output(
            embedding, radial[order], idx_i[order], idx_j[order], vij[order]
        )["atom_feature"]

    reference = run(pos)
    rotation = _rotation(4)
    assert torch.allclose(reference, run(pos @ rotation.T), atol=1e-4, rtol=1e-4)
    assert torch.allclose(reference, run(-pos), atol=1e-4, rtol=1e-4)
    assert torch.allclose(reference, run(pos + torch.tensor([0.4, -0.2, 0.7])), atol=1e-4, rtol=1e-4)
    order = torch.randperm(idx_i.numel())
    assert torch.allclose(reference, run(pos, order), atol=1e-4, rtol=1e-4)


def test_disconnected_molecules_match_separate_forwards():
    torch.manual_seed(5)
    core = _small_core()
    core.eval()
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]])
    right_pos = torch.tensor([[0.2, 0.0, 0.0], [0.1, 1.1, 0.2], [-0.4, 0.3, 0.8]])
    left_h = torch.randn(2, 3)
    right_h = torch.randn(3, 3)
    left_i, left_j = _edges(2)
    right_i, right_j = _edges(3)
    left_radial = torch.rand(left_i.numel(), 4)
    right_radial = torch.rand(right_i.numel(), 4)
    left = core.get_output(
        left_h, left_radial, left_i, left_j, left_pos[left_j] - left_pos[left_i]
    )["atom_feature"]
    right = core.get_output(
        right_h,
        right_radial,
        right_i,
        right_j,
        right_pos[right_j] - right_pos[right_i],
    )["atom_feature"]
    both = core.get_output(
        torch.cat([left_h, right_h]),
        torch.cat([left_radial, right_radial]),
        torch.cat([left_i, right_i + 2]),
        torch.cat([left_j, right_j + 2]),
        torch.cat([left_pos, right_pos + 10.0], dim=0)[
            torch.cat([left_j, right_j + 2])
        ]
        - torch.cat([left_pos, right_pos + 10.0], dim=0)[
            torch.cat([left_i, right_i + 2])
        ],
    )["atom_feature"]
    assert torch.allclose(both[:2], left, atol=1e-5, rtol=1e-5)
    assert torch.allclose(both[2:], right, atol=1e-5, rtol=1e-5)


def test_stack_energy_is_invariant_and_forces_match_a_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    model.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3], [0.4, 0.2, 1.2]],
        dtype=torch.float32,
    )
    idx_i, idx_j = _edges(pos.shape[0])
    species = torch.tensor([1, 6, 7, 8])

    def run(positions: torch.Tensor):
        positions = positions.detach().requires_grad_(True)
        output = model(
            {
                "Ra": positions,
                "Za": species,
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": torch.zeros(positions.shape[0], dtype=torch.long),
            }
        )
        return output["E"], output["Fa"]

    energy, forces = run(pos)
    assert energy.shape == (1,)
    assert forces.shape == pos.shape
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()
    rotation = _rotation(7)
    energy_rot, forces_rot = run(pos @ rotation.T)
    assert torch.allclose(energy, energy_rot, atol=1e-4, rtol=1e-4)
    assert torch.allclose(forces @ rotation.T, forces_rot, atol=1e-4, rtol=1e-4)
    energy_inv, forces_inv = run(-pos)
    assert torch.allclose(energy, energy_inv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(-forces, forces_inv, atol=1e-4, rtol=1e-4)
    shift = torch.tensor([0.4, -0.3, 0.2])
    energy_shift, forces_shift = run(pos + shift)
    assert torch.allclose(energy, energy_shift, atol=1e-4, rtol=1e-4)
    assert torch.allclose(forces, forces_shift, atol=1e-4, rtol=1e-4)

    step = 1e-3
    bumped = pos.clone()
    bumped[1, 0] = bumped[1, 0] + step
    energy_plus, _ = run(bumped)
    bumped[1, 0] = pos[1, 0] - step
    energy_minus, _ = run(bumped)
    finite = -(energy_plus - energy_minus) / (2 * step)
    assert torch.allclose(forces[1, 0], finite.reshape(()), atol=2e-3, rtol=1e-3)

    alone_pos = pos[:1].detach().requires_grad_(True)
    alone = model(
        {
            "Ra": alone_pos,
            "Za": species[:1],
            "idx_i": torch.zeros(0, dtype=torch.long),
            "idx_j": torch.zeros(0, dtype=torch.long),
            "batch_seg": torch.zeros(1, dtype=torch.long),
        }
    )
    assert torch.isfinite(alone["E"]).all()
    assert torch.allclose(alone["Fa"], torch.zeros_like(alone_pos), atol=1e-6)


def test_scalar_stack_matches_upstream_e3nn():
    if not (UPSTREAM / "eqnorm" / "eqnorm-mptrj.py").is_file():
        pytest.skip("yzchen08/eqnorm checkout is not at /tmp/eqnorm-src")
    root = str(UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)
    import eqnorm.layernorm  # noqa: F401

    spec = importlib.util.spec_from_file_location(
        "eqnorm_ref_stack", UPSTREAM / "eqnorm" / "eqnorm-mptrj.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "eqnorm"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.manual_seed(8)
    reference = module.E3NN(
        irreps_hidden="4x0e+2x1o",
        irreps_sh="1x0e+1x1o",
        num_conv_layers=2,
        num_types=3,
        num_features=4,
        max_radius=6.0,
        num_basis=4,
        invariant_layers=1,
        invariant_neurons=8,
        poly_p=6,
        avg_num_neighbors=2.5,
    )
    core = _small_core(num_layers=2)
    core.embed.projection.load_state_dict(reference.embedding.linear.state_dict())
    for index in range(2):
        core.blocks[index].module.tensor_conv.load_state_dict(reference.layers[4 * index].state_dict())
        core.blocks[index].module.scalar_conv.load_state_dict(reference.layers[4 * index + 2].state_dict())

    indices = torch.tensor([0, 2, 1])
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.3]])
    idx_i, idx_j = _edges(3)
    vij = pos[idx_j] - pos[idx_i]
    distance = torch.linalg.vector_norm(vij, dim=-1)
    radial = reference.besssel_basis(distance) * reference.poly_cutoff(distance).unsqueeze(-1)
    data = {
        "atomic_numbers": indices,
        "edge_index": torch.stack([idx_i, idx_j]),
        "edge_vec": vij,
    }
    data["node_hiddens"] = reference.embedding(data["atomic_numbers"])
    data["output"] = data["node_hiddens"]
    data["edge_sh"] = reference.sph(vij)
    data["edge_attr"] = radial
    for layer in reference.layers:
        layer(data)
    one_hot = F.one_hot(indices, num_classes=3).float()
    got = core.get_output(one_hot, radial, idx_i, idx_j, vij)["atom_feature"]
    assert torch.allclose(got, data["output"], atol=1e-4, rtol=1e-4)


def test_feature_layout_is_flat_even_scalars():
    core = _small_core()
    assert o3.Irreps(core.feature_irreps) == o3.Irreps("4x0e")
    assert core.blocks[0].module.irreps_tensor == o3.Irreps("4x0e+2x1o")
    assert core.blocks[0].module.gate.irreps_in[0].mul == 4 + 2
