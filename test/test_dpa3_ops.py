"""DPA3 gates, symmetrizer, angle pairs, and residual combiners."""

import math

import torch
from numpy.testing import assert_allclose


def test_polynomial_switch_matches_deepmd_gate():
    from enerzyme.models.mp.basis.envelope import polynomial_transition

    distance = torch.tensor([0.0, 4.0, 5.5, 6.0, 7.0])
    got = polynomial_transition(distance, 6.0, 5.0)
    assert_allclose(got[:2].numpy(), [1.0, 1.0], atol=1e-6)
    assert_allclose(got[3:].numpy(), [0.0, 0.0], atol=1e-6)
    uu = (5.5 - 5.0) / (6.0 - 5.0)
    expected = uu**3 * (-6 * uu**2 + 15 * uu - 10) + 1
    assert_allclose(got[2].item(), expected, atol=1e-6)


def test_exponential_switch_limits_and_gradient():
    from enerzyme.models.mp.basis.envelope import exponential_switch

    distance = torch.tensor([0.0, 5.3, 6.0], requires_grad=True)
    gate = exponential_switch(distance, 6.0, 5.3, decay=20.0)
    assert gate[0].item() > 0.99
    assert gate[1].item() < math.exp(-1.0) + 1e-6
    assert gate[2].item() == 0.0
    grad = torch.autograd.grad(gate[:2].sum(), distance)[0]
    assert torch.isfinite(grad).all()


def test_silut_matches_value_and_first_derivative():
    from enerzyme.models.mp.nn.activation import silut, silut_constants

    threshold = 3.0
    slope, const = silut_constants(threshold)
    left = torch.tensor([threshold - 1e-4], requires_grad=True)
    right = torch.tensor([threshold + 1e-4], requires_grad=True)
    y_left = silut(left, threshold)
    y_right = silut(right, threshold)
    assert_allclose(y_left.detach().item(), torch.nn.functional.silu(left).item(), atol=1e-6)
    tail = math.tanh(slope * (1e-4)) + const
    assert_allclose(y_right.detach().item(), tail, atol=1e-5)
    g_left = torch.autograd.grad(y_left, left)[0]
    g_right = torch.autograd.grad(y_right, right)[0]
    assert_allclose(g_left.item(), g_right.item(), atol=2e-3)


def test_symmetrize_one_edge_and_rotation():
    from enerzyme.models.layers.symmetrize import dp_symmetrize

    invariant = torch.tensor([[1.0, 0.0]])
    vector = torch.tensor([[2.0, 0.0, 0.0]])
    owner = torch.tensor([0])
    gram = dp_symmetrize(invariant, vector, owner, 1, axis_neuron=1, inv_sqrt_neighbors=0.5)
    assert_allclose(gram.numpy(), [[1.0 / 3.0, 0.0]], atol=1e-6)

    features = torch.randn(5, 4)
    vectors = torch.randn(5, 3)
    owner = torch.tensor([0, 1, 0, 2, 1])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    plain = dp_symmetrize(features, vectors, owner, 3, 2, 0.25)
    rotated = dp_symmetrize(features, vectors @ rotation, owner, 3, 2, 0.25)
    assert_allclose(plain.numpy(), rotated.numpy(), atol=1e-5)


def test_angle_pairs_include_the_diagonal():
    from enerzyme.models.dpa3.geometry import angle_pairs

    idx_i = torch.tensor([0, 0, 1])
    dist = torch.tensor([1.0, 1.5, 3.5])
    pairs = angle_pairs(idx_i, dist, num_nodes=2, angular_cutoff=2.0)
    found = set(zip(pairs.edge_ij.tolist(), pairs.edge_ik.tolist(), pairs.center.tolist()))
    assert found == {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)}
    assert angle_pairs(idx_i, dist, 2, 0.5).edge_ij.numel() == 0


def test_residual_combiners():
    from enerzyme.models.dpa3.interaction import combine_updates

    identity = torch.tensor([2.0, 0.0])
    message = torch.tensor([2.0, 4.0])
    weight = torch.tensor([0.5, 0.25])
    assert_allclose(
        combine_updates([identity, message], "res_avg").numpy(),
        (identity + message).numpy() / math.sqrt(2),
    )
    assert_allclose(
        combine_updates([identity, message], "res_incr").numpy(),
        (identity + message).numpy(),
    )
    assert_allclose(
        combine_updates([identity, message], "res_residual", [weight]).numpy(),
        (identity + weight * message).numpy(),
    )


def test_repflow_smooth_and_compression_shapes():
    from enerzyme.models.mp.nn.activation import resolve_scalar_activation
    from enerzyme.models.dpa3.interaction import RepFlowBlock

    torch.manual_seed(0)
    block = RepFlowBlock(
        n_dim=8, e_dim=8, a_dim=8, axis_neuron=2, n_multi_edge_message=2,
        update_angle=True, update_style="res_residual", update_residual=0.1,
        update_residual_init="const", activation=resolve_scalar_activation("silu"),
        a_compress_rate=2, a_compress_e_rate=1, a_compress_use_split=False,
        sequential_update=False, smooth_edge_update=True,
        neighbor_scale=4.0, angular_scale=4.0,
    )
    node = torch.randn(3, 8)
    edge = torch.randn(4, 8)
    angle = torch.randn(2, 8)
    h = torch.randn(4, 3)
    switch = torch.rand(4)
    idx_i = torch.tensor([0, 0, 1, 2])
    idx_j = torch.tensor([1, 2, 0, 0])
    node_out, edge_out, angle_out = block(
        node, edge, angle, h, switch, switch, idx_i, idx_j,
        torch.tensor([0, 1]), torch.tensor([1, 0]), torch.tensor([0, 0]),
        torch.tensor([True, True, False, False]),
    )
    assert node_out.shape == (3, 8)
    assert edge_out.shape == (4, 8)
    assert angle_out.shape == (2, 8)
    assert torch.isfinite(node_out).all()
