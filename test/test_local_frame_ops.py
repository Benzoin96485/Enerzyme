"""Local-frame geometry shared by LEFTNet and AlphaNet."""

from __future__ import annotations

import math

import pytest
import torch

from enerzyme.models.layers.rbf import ExpNormalSmearing
from enerzyme.models.mp.message.frame.encoding import LocalSubstructure
from enerzyme.models.mp.message.frame.geometry import (
    alphanet_edge_frame,
    leftnet_edge_frame,
    leftnet_node_frame,
    neighbor_mean,
    require_matching_distance,
    require_same_graph,
    scalarize,
    subtract_graph_mean,
)


def _triangle():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 0.9, 0.15]],
        dtype=torch.float64,
    )
    idx_i = torch.tensor([0, 1, 0, 2, 1, 2])
    idx_j = torch.tensor([1, 0, 2, 0, 2, 1])
    vij = pos[idx_j] - pos[idx_i]
    dij = torch.linalg.vector_norm(vij, dim=-1)
    return pos, idx_i, idx_j, vij, dij


def test_expnormal_alpha_one_matches_leftnet_basis():
    cutoff = 6.0
    num_rbf = 5
    dist = torch.tensor([0.4, 2.5, 6.0, 6.5], dtype=torch.float64)
    layer = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=cutoff, alpha=1.0).double()
    start = math.exp(-cutoff)
    assert abs(float(layer.means[0]) - start) < 1e-6
    window = 0.5 * (torch.cos(dist * math.pi / cutoff) + 1.0)
    window = window * (dist < cutoff).to(dist.dtype)
    manual = window.unsqueeze(-1) * torch.exp(
        -layer.betas * (torch.exp(-dist.unsqueeze(-1)) - layer.means) ** 2
    )
    assert torch.allclose(layer.get_rbf(dist), manual, atol=1e-12)
    assert torch.count_nonzero(layer.get_rbf(dist)[-1]) == 0
    default = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=cutoff).double()
    assert not torch.allclose(default.get_rbf(dist[:2]), layer.get_rbf(dist[:2]))
    at_five = ExpNormalSmearing(num_rbf=4, cutoff_sr=5.0)
    assert at_five.alpha == 1.0
    wider = ExpNormalSmearing(num_rbf=4, cutoff_sr=4.5)
    assert wider.alpha == 5.0 / 4.5


def test_leftnet_frame_is_translation_invariant_after_centering():
    pos, idx_i, idx_j, vij, _ = _triangle()
    centered = subtract_graph_mean(pos, None)
    shifted = subtract_graph_mean(pos + torch.tensor([4.0, -2.0, 0.5]), None)
    frame, unit = leftnet_edge_frame(centered, vij, idx_i, idx_j, 1e-10)
    frame_s, unit_s = leftnet_edge_frame(shifted, vij, idx_i, idx_j, 1e-10)
    assert torch.allclose(frame, frame_s, atol=1e-12)
    assert torch.allclose(unit, unit_s, atol=1e-12)


def test_alphanet_frame_differs_from_leftnet_and_ignores_translation():
    pos, idx_i, idx_j, vij, dij = _triangle()
    centered = subtract_graph_mean(pos, None)
    left, _ = leftnet_edge_frame(centered, vij, idx_i, idx_j, 1e-10)
    alpha, unit = alphanet_edge_frame(pos, vij, dij, idx_i, idx_j, 1e-10)
    alpha_s, _ = alphanet_edge_frame(pos + 3.0, vij, dij, idx_i, idx_j, 1e-10)
    assert alpha.shape == left.shape == (idx_i.shape[0], 3, 3)
    assert not torch.allclose(alpha, left)
    assert torch.allclose(alpha, alpha_s, atol=1e-12)
    assert torch.allclose(unit, vij / (dij.unsqueeze(-1) + 1e-10), atol=1e-12)


def test_scalarize_abs_middle_axis_is_reflection_invariant():
    torch.manual_seed(0)
    vectors = torch.randn(4, 3, 5, dtype=torch.float64)
    frame = torch.randn(4, 3, 3, dtype=torch.float64)
    base = scalarize(vectors, frame)
    flipped = frame.clone()
    flipped[..., 1] *= -1
    assert torch.allclose(base, scalarize(vectors, flipped), atol=1e-12)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    rotated_vectors = torch.einsum("dc,ncf->ndf", rotation, vectors)
    rotated_frame = torch.einsum("dc,nca->nda", rotation, frame)
    assert torch.allclose(
        base, scalarize(rotated_vectors, rotated_frame), atol=1e-10
    )


def test_endpoint_order_swaps_the_two_halves():
    torch.manual_seed(1)
    hidden = 8
    pos, idx_i, idx_j, vij, dij = _triangle()
    scalar = torch.randn(pos.shape[0], hidden, dtype=torch.float64)
    radial = torch.randn(idx_i.shape[0], hidden, dtype=torch.float64)
    rbf = torch.randn(idx_i.shape[0], 4, dtype=torch.float64)
    window = torch.rand(idx_i.shape[0], dtype=torch.float64)
    frame, unit = alphanet_edge_frame(pos, vij, dij, idx_i, idx_j, 1e-10)
    source = LocalSubstructure(hidden, False, False, "source_target").double()
    target = LocalSubstructure(hidden, False, False, "target_source").double()
    target.load_state_dict(source.state_dict())
    weight_s = source(scalar, unit, frame, radial, rbf, window, idx_i, idx_j)
    weight_t = target(scalar, unit, frame, radial, rbf, window, idx_i, idx_j)
    assert torch.allclose(weight_s[:, :hidden], weight_t[:, hidden : 2 * hidden])
    assert torch.allclose(weight_s[:, hidden : 2 * hidden], weight_t[:, :hidden])
    assert torch.allclose(weight_s[:, 2 * hidden :], weight_t[:, 2 * hidden :])


def test_graphs_are_centered_separately():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 0, 1, 1])
    centered = subtract_graph_mean(pos, batch)
    assert torch.allclose(centered[:2].mean(dim=0), torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(centered[2:].mean(dim=0), torch.zeros(3, dtype=torch.float64))
    assert not torch.allclose(centered, pos - pos.mean(dim=0))


def test_unconnected_atom_gets_the_identity_node_frame():
    pos = torch.tensor(
        [[4.0, -1.0, 0.3], [0.0, 0.0, 0.0], [1.2, 0.1, 0.0]],
        dtype=torch.float64,
    )
    idx_i = torch.tensor([1, 2])
    idx_j = torch.tensor([2, 1])
    mean, count = neighbor_mean(pos, idx_i, idx_j)
    assert count[0].item() == 0
    assert count[1].item() == 1
    assert torch.allclose(mean[1], pos[2])
    frame = leftnet_node_frame(pos, idx_i, idx_j, 1e-10)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(frame[0], eye)
    assert not torch.allclose(frame[1], eye)


def test_centroid_on_the_atom_is_the_identity_frame():
    pos = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float64)
    frame = leftnet_node_frame(
        pos, torch.tensor([0]), torch.tensor([1]), 1e-10
    )
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(frame[0], eye)
    assert torch.allclose(frame[1], eye)


def test_alphanet_bond_axis_uses_the_vector_length():
    pos, idx_i, idx_j, vij, dij = _triangle()
    _, unit = alphanet_edge_frame(pos, vij, dij, idx_i, idx_j, 1e-10)
    length = torch.linalg.vector_norm(unit, dim=-1)
    assert torch.allclose(length, torch.ones_like(length), atol=1e-8)
    with pytest.raises(ValueError, match="Dij_sr"):
        alphanet_edge_frame(pos, vij, dij * 2, idx_i, idx_j, 1e-10)


def test_compiled_local_frame_input_validation():
    pos, i, j, vec, dist = _triangle()
    def validate(batch, i, j, vec, dist):
        require_same_graph(batch, i, j)
        require_matching_distance(vec, dist)
        return dist + 1

    compiled = torch.compile(validate, backend="eager", fullgraph=True)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    torch.testing.assert_close(compiled(batch, i, j, vec, dist), dist + 1)
    with pytest.raises(RuntimeError, match="one molecule"):
        compiled(torch.tensor([0, 1, 0]), i, j, vec, dist)
    with pytest.raises(RuntimeError, match="Euclidean norm"):
        compiled(batch, i, j, vec, dist + 1.)
    with pytest.raises(RuntimeError, match="cover every neighbor index"):
        compiled(batch, i + 2, j, vec, dist)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only local-frame fusion")
def test_compiled_local_frame_projection_honors_nested_hook():
    layer = LocalSubstructure(8, False, True, "source_target").cuda().double().eval()
    environment = torch.randn(4, 3, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    frame = torch.eye(3, device="cuda", dtype=torch.float64).expand(3, -1, -1)
    nodes = torch.tensor([2, 0, 3], device="cuda")
    inputs = (environment, *layer.lin.parameters())
    with layer.lin[1].register_forward_hook(lambda _, args, y: y * 1.7):
        expected = layer._endpoint(environment, frame, nodes)
        actual = torch.compile(layer._endpoint, backend="eager", fullgraph=True)(environment, frame, nodes)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)
