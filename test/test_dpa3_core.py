"""DPA3 Core: registration, invariance, and conservative forces."""

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "dpa3_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.long),
        torch.tensor([p[1] for p in pairs], dtype=torch.long),
    )


def _rotation(dtype=torch.float64, improper=False):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    if improper:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _core(**kwargs):
    from enerzyme.models.presets.triplet_graphs import dpa3

    params = dict(
        dim_embedding=8, e_dim=8, a_dim=8, nlayers=2, e_sel=8, a_sel=6,
        axis_neuron=2, e_rcut=6.0, e_rcut_smth=5.0, a_rcut=4.0, a_rcut_smth=3.5,
    )
    params.update(kwargs)
    return dpa3(**params)


def _geometry(n=4, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.0], [0.1, 1.2, 0.2], [0.3, 0.2, 1.15]],
        dtype=dtype,
    )[:n]
    idx_i, idx_j = _edges(n)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    embedding = torch.randn(n, 8, dtype=dtype)
    return embedding, dist, vij, idx_i, idx_j, pos


def test_dpa3_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("dpa3")
    assert core_cls.__name__ == "dpa3"
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"], layer_params=ff["layers"],
        build_params=ff["build_params"], verbose=0,
    )
    assert model.architecture == "dpa3"


def test_dpa3_shapes_and_geometry_autograd():
    torch.manual_seed(0)
    embedding, dist, vij, idx_i, idx_j, pos = _geometry()
    pos = pos.detach().requires_grad_(True)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    core = _core()
    out = core.get_output(embedding, dist, vij, idx_i, idx_j)
    assert out["atom_feature"].shape == (4, 8)
    assert core.feature_irreps == "8x0e"
    grad = torch.autograd.grad(out["atom_feature"].square().sum(), pos)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_dpa3_rotation_translation_and_edge_order():
    torch.manual_seed(0)
    dtype = torch.float64
    core = _core().to(dtype).eval()
    embedding, dist, vij, idx_i, idx_j, pos = _geometry(dtype=dtype)
    with torch.no_grad():
        base = core.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
        for improper in (False, True):
            rotation = _rotation(dtype, improper=improper)
            vij_r = vij @ rotation
            dist_r = torch.linalg.vector_norm(vij_r, dim=-1)
            rotated = core.get_output(embedding, dist_r, vij_r, idx_i, idx_j)["atom_feature"]
            assert_allclose(base.numpy(), rotated.numpy(), atol=1e-8, rtol=1e-8)
        order = torch.randperm(idx_i.shape[0])
        shuffled = core.get_output(
            embedding, dist[order], vij[order], idx_i[order], idx_j[order]
        )["atom_feature"]
        assert_allclose(base.numpy(), shuffled.numpy(), atol=1e-8, rtol=1e-8)

    shifted = pos + torch.tensor([3.0, -2.0, 0.5], dtype=dtype)
    vij_s = shifted[idx_j] - shifted[idx_i]
    dist_s = torch.linalg.vector_norm(vij_s, dim=-1)
    with torch.no_grad():
        moved = core.get_output(embedding, dist_s, vij_s, idx_i, idx_j)["atom_feature"]
    assert_allclose(base.numpy(), moved.numpy(), atol=1e-8, rtol=1e-8)


def test_dpa3_atom_permutation_and_batch_isolation():
    torch.manual_seed(1)
    dtype = torch.float64
    core = _core().to(dtype).eval()
    embedding, dist, vij, idx_i, idx_j, _ = _geometry(n=3, dtype=dtype)
    perm = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(3)
    with torch.no_grad():
        base = core.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
        permuted = core.get_output(
            embedding[perm], dist, vij, inverse[idx_i], inverse[idx_j]
        )["atom_feature"]
    assert_allclose(base[perm].numpy(), permuted.numpy(), atol=1e-8, rtol=1e-8)

    far = embedding.new_tensor([0.0, 0.0, 30.0])
    emb2 = embedding + 0.3
    # Second copy is translated; its internal edges match the first geometry.
    both_emb = torch.cat([embedding, emb2], dim=0)
    both_i = torch.cat([idx_i, idx_i + 3])
    both_j = torch.cat([idx_j, idx_j + 3])
    both_v = torch.cat([vij, vij], dim=0)
    both_d = torch.cat([dist, dist])
    cross_i = torch.tensor([0, 3])
    cross_j = torch.tensor([3, 0])
    cross_v = torch.stack([far, -far])
    both_i = torch.cat([both_i, cross_i])
    both_j = torch.cat([both_j, cross_j])
    both_v = torch.cat([both_v, cross_v])
    both_d = torch.cat([both_d, torch.linalg.vector_norm(cross_v, dim=-1)])
    with torch.no_grad():
        joined = core.get_output(both_emb, both_d, both_v, both_i, both_j)["atom_feature"]
        alone = core.get_output(emb2, dist, vij, idx_i, idx_j)["atom_feature"]
    assert_allclose(joined[:3].numpy(), base.numpy(), atol=1e-8, rtol=1e-8)
    assert_allclose(joined[3:].numpy(), alone.numpy(), atol=1e-8, rtol=1e-8)


def test_dpa3_variants_and_rejected_settings():
    torch.manual_seed(0)
    embedding, dist, vij, idx_i, idx_j, _ = _geometry()
    parallel = _core(sequential_update=False).eval()
    sequential = _core(sequential_update=True).eval()
    sequential.load_state_dict(parallel.state_dict())
    with torch.no_grad():
        left = parallel.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
        right = sequential.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
    assert not torch.allclose(left, right)

    for kwargs in (
        {"update_angle": False},
        {"n_multi_edge_message": 2},
        {"dynamic_normalize": True},
        {"edge_init_use_dist": True},
        {"switch": "exponential", "switch_decay": 20.0},
        {"activation": "silut"},
        {"a_compress_rate": 2, "a_compress_use_split": True},
        {"smooth_edge_update": False},
        {"update_style": "res_avg"},
    ):
        core = _core(**kwargs).eval()
        out = core.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    try:
        _core(a_rcut=7.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a_rcut larger than e_rcut should fail")
    try:
        _core(sequential_update=True, update_angle=False)
    except ValueError:
        pass
    else:
        raise AssertionError("sequential update without angles should fail")


def test_dpa3_layer_stack_energy_force_and_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    layers = [
        {"name": "RangeSeparation"},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {
            "e_dim": 8, "a_dim": 8, "nlayers": 1, "e_sel": 8, "a_sel": 6, "axis_neuron": 2,
        }},
        {"name": "SimpleReadout", "params": {
            "output_fields": ["Ea"], "head_type": "dense", "keep_feature": False,
        }},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "dpa3", layer_params=layers,
        build_params={"dim_embedding": 8, "max_Za": 20, "cutoff_sr": 6.0},
        verbose=0,
    ).double().eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.05, 0.0], [0.1, 1.15, 0.1], [0.25, 0.2, 1.05]],
        dtype=torch.float64, requires_grad=True,
    )
    idx_i, idx_j = _edges(4)
    batch = {
        "Ra": pos, "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(4, dtype=torch.long),
        "idx_i": idx_i, "idx_j": idx_j,
    }
    out = model(batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()
    assert out["Fa"].abs().sum() > 0
    assert_allclose(out["Fa"].sum(0).detach().numpy(), [0, 0, 0], atol=1e-8)

    shifted = {**batch, "Ra": (pos.detach() + 1.7).requires_grad_(True)}
    moved = model(shifted)["E"].detach()
    assert_allclose(out["E"].detach().numpy(), moved.numpy(), atol=1e-8)

    fa = out["Fa"].detach()
    eps = 1e-5
    fd = torch.zeros_like(fa)
    base = pos.detach()
    for atom in range(4):
        for axis in range(3):
            plus = base.clone()
            minus = base.clone()
            plus[atom, axis] += eps
            minus[atom, axis] -= eps
            plus.requires_grad_(True)
            minus.requires_grad_(True)
            energy_plus = model({**batch, "Ra": plus})["E"]
            energy_minus = model({**batch, "Ra": minus})["E"]
            fd[atom, axis] = -(energy_plus - energy_minus) / (2 * eps)
    torch.testing.assert_close(fa, fd, atol=2e-4, rtol=2e-4)


def test_dpa3_smooth_edge_update_sees_the_angular_cutoff():
    torch.manual_seed(0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.2, 0.0, 0.0]])
    idx_i, idx_j = _edges(3)
    vij = pos[idx_j] - pos[idx_i]
    dist = torch.linalg.vector_norm(vij, dim=-1)
    embedding = torch.randn(3, 8)
    smooth = _core(smooth_edge_update=True).eval()
    sharp = _core(smooth_edge_update=False).eval()
    sharp.load_state_dict(smooth.state_dict())
    with torch.no_grad():
        left = smooth.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
        right = sharp.get_output(embedding, dist, vij, idx_i, idx_j)["atom_feature"]
    assert not torch.allclose(left, right)


def test_dpa3_isolated_atom_is_finite():
    core = _core().eval()
    embedding = torch.randn(2, 8)
    empty = embedding.new_zeros(0)
    empty_v = embedding.new_zeros(0, 3)
    empty_i = torch.zeros(0, dtype=torch.long)
    out = core.get_output(embedding, empty, empty_v, empty_i, empty_i)
    assert out["atom_feature"].shape == (2, 8)
    assert torch.isfinite(out["atom_feature"]).all()
