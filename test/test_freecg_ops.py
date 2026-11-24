"""FreeCG group CG paths, channel roll, and attention enhancer."""

from __future__ import annotations

import e3nn
import torch
from e3nn.o3 import Irrep

from enerzyme.models.freecg.cg import GroupSparseCG, shuffle_offset
from enerzyme.models.freecg.interaction import FreeCGInteraction, normalize_attention_enhancer
from enerzyme.models.layers.polynomial_harmonics import polynomial_spherical_harmonics
from enerzyme.models.visnet.interaction import ViSInteraction


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _rotate_angular(features: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply the polynomial-basis Wigner matrices. ``features`` is ``[N, 8, C]``."""
    degree_1 = Irrep("1o").D_from_matrix(rotation).to(dtype=features.dtype)
    degree_2 = Irrep("2e").D_from_matrix(rotation).to(dtype=features.dtype)
    vector = torch.einsum("naf,ad->ndf", features[:, :3], degree_1)
    quadrupole = torch.einsum("naf,ad->ndf", features[:, 3:], degree_2)
    return torch.cat((vector, quadrupole), dim=1)


def _irreps_features(n: int, channels: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.nn.functional.normalize(
        torch.randn(n, channels, 3, generator=generator), dim=-1
    )
    harmonics = polynomial_spherical_harmonics(directions.reshape(-1, 3), 2)
    return harmonics.reshape(n, channels, 8).permute(0, 2, 1).contiguous()


def test_shuffle_offset_and_names():
    assert shuffle_offset(256, 8, 1.5) == 48
    assert shuffle_offset(256, 32, 1.5) == 12
    assert shuffle_offset(8, 2, 0.0) == 0
    assert normalize_attention_enhancer("Linear") == "linear"
    assert normalize_attention_enhancer("MAX") == "max"
    try:
        shuffle_offset(8, 3, 1.5)
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected a bad group count to fail")
    try:
        shuffle_offset(8, 2, 0.2)
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("expected a fractional roll to fail")
    try:
        normalize_attention_enhancer("softmax")
    except ValueError as exc:
        assert "attention_enhancer" in str(exc)
    else:
        raise AssertionError("expected an unknown enhancer to fail")


def test_sparse_paths_and_independent_groups():
    block = GroupSparseCG(channels=8, num_groups=2, shuffle_scale=0.0)
    assert block.n_paths == 4
    assert block.width == 4
    assert block.offset == 0
    block.tp_weight.data.fill_(0.3)

    left = torch.randn(2, 8, 8)
    right = torch.randn(2, 8, 8)
    pure_vector_left = left.clone()
    pure_vector_right = right.clone()
    pure_vector_left[:, 3:] = 0
    pure_vector_right[:, 3:] = 0
    vector_product = block(pure_vector_left, pure_vector_right)
    assert torch.allclose(vector_product[:, :3], torch.zeros_like(vector_product[:, :3]), atol=1e-6)
    assert torch.count_nonzero(vector_product[:, 3:]) > 0

    cross_left = left.clone()
    cross_right = right.clone()
    cross_left[:, 3:] = 0
    cross_right[:, :3] = 0
    cross = block(cross_left, cross_right)
    assert torch.allclose(cross[:, 3:], torch.zeros_like(cross[:, 3:]), atol=1e-6)
    assert torch.count_nonzero(cross[:, :3]) > 0

    full = block(left, right)
    block.tp_weight.data[1].add_(1.0)
    changed = block(left, right)
    assert not torch.allclose(full[..., 4:], changed[..., 4:], atol=1e-6)
    assert torch.allclose(full[..., :4], changed[..., :4], atol=1e-6)


def test_roll_increases_the_channel_index():
    stopped = GroupSparseCG(channels=8, num_groups=2, shuffle_scale=0.0)
    rolled = GroupSparseCG(channels=8, num_groups=2, shuffle_scale=1.5)
    assert rolled.offset == 6
    rolled.tp_weight.data.copy_(stopped.tp_weight.data)
    left = torch.randn(3, 8, 8)
    right = torch.randn(3, 8, 8)
    base = stopped(left, right)
    shifted = torch.roll(base, shifts=rolled.offset, dims=-1)
    assert torch.allclose(rolled(left, right), shifted, atol=1e-6)
    assert torch.equal(
        shifted[0, 0, :2],
        torch.roll(base[0, 0], shifts=6, dims=-1)[:2],
    )


def test_group_cg_rotation_and_reflection():
    torch.manual_seed(1)
    block = GroupSparseCG(channels=8, num_groups=2, shuffle_scale=1.5)
    block.eval()
    left = _irreps_features(5, 8, seed=1)
    right = _irreps_features(5, 8, seed=2)
    out = block(left, right)

    rotation = _rotation(3)
    rotated = block(_rotate_angular(left, rotation), _rotate_angular(right, rotation))
    assert torch.allclose(_rotate_angular(out, rotation), rotated, atol=1e-5, rtol=1e-5)

    reflection = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
    assert torch.det(reflection) < 0
    reflected = block(_rotate_angular(left, reflection), _rotate_angular(right, reflection))
    assert torch.allclose(_rotate_angular(out, reflection), reflected, atol=1e-5, rtol=1e-5)


def test_zero_node_vector_matches_visnet_and_enhancer_changes_a_real_vector():
    torch.manual_seed(4)
    hidden, heads = 8, 2
    vis = ViSInteraction(hidden, heads, "none", "none", False, last_layer=False)
    plain = FreeCGInteraction(
        hidden, heads, "none", "none", False, last_layer=False,
        num_groups=2, shuffle_scale=1.5, attention_enhancer="none",
    )
    linear = FreeCGInteraction(
        hidden, heads, "none", "none", False, last_layer=False,
        num_groups=2, shuffle_scale=1.5, attention_enhancer="linear",
    )
    maximum = FreeCGInteraction(
        hidden, heads, "none", "none", False, last_layer=False,
        num_groups=2, shuffle_scale=1.5, attention_enhancer="max",
    )
    plain.load_state_dict(vis.state_dict(), strict=False)
    linear.load_state_dict(vis.state_dict(), strict=False)
    maximum.load_state_dict(vis.state_dict(), strict=False)
    linear.group_cg.tp_weight.data.copy_(plain.group_cg.tp_weight.data)
    maximum.group_cg.tp_weight.data.copy_(plain.group_cg.tp_weight.data)
    for block in (vis, plain, linear, maximum):
        block.eval()

    n_atoms, n_edges = 4, 6
    scalar = torch.randn(n_atoms, hidden)
    edge = torch.randn(n_edges, hidden)
    idx_i = torch.tensor([0, 0, 1, 1, 2, 3])
    idx_j = torch.tensor([1, 2, 0, 3, 0, 1])
    window = torch.rand(n_edges).clamp(0.2, 1)
    direction = polynomial_spherical_harmonics(
        torch.nn.functional.normalize(torch.randn(n_edges, 3), dim=-1), 2
    )
    zero = torch.zeros(n_atoms, 8, hidden)
    vis_s, vis_v, vis_e = vis(scalar, zero, edge, idx_i, idx_j, window, direction)
    plain_s, plain_v, plain_e = plain(scalar, zero, edge, idx_i, idx_j, window, direction)
    linear_s, linear_v, _ = linear(scalar, zero, edge, idx_i, idx_j, window, direction)
    assert torch.allclose(vis_s, plain_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vis_v, plain_v, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vis_e, plain_e, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vis_s, linear_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vis_v, linear_v, atol=1e-5, rtol=1e-5)

    vector = _irreps_features(n_atoms, hidden, seed=5)
    base_s, base_v, _ = plain(scalar, vector, edge, idx_i, idx_j, window, direction)
    linear_s, linear_v, _ = linear(scalar, vector, edge, idx_i, idx_j, window, direction)
    max_s, max_v, _ = maximum(scalar, vector, edge, idx_i, idx_j, window, direction)
    assert not torch.allclose(base_v, linear_v, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base_s, max_s, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(linear_v, max_v, atol=1e-5, rtol=1e-5)

def test_group_product_compiles_with_weight_gradients_and_restores_defaults():
    previous = e3nn.get_optimization_defaults()
    block = GroupSparseCG(channels=8, num_groups=2).double()
    assert e3nn.get_optimization_defaults() == previous

    left, right = (torch.randn(5, 8, 8, dtype=torch.double, requires_grad=True) for _ in range(2))
    def evaluate(model):
        mixed = model(left, right)
        gradients = torch.autograd.grad(mixed.square().sum(), (block.tp_weight, left, right))
        return mixed, gradients

    compiled = torch.compile(block, backend="eager", fullgraph=True)
    torch.testing.assert_close(evaluate(compiled), evaluate(block))
