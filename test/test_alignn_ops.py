"""ALIGNN line graph, Gaussian width, and edge-gated convolution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from numpy.testing import assert_allclose


def test_alignn_gaussian_uses_inverse_spacing_not_schnet_width():
    from enerzyme.models.layers.rbf import GaussianRBFLayer, alignn_gaussian_terms

    cutoff = 8.0
    bins = 80
    layer = GaussianRBFLayer(
        num_rbf=bins,
        cutoff_sr=cutoff,
        flavor="alignn",
        apply_cutoff_fn=False,
    )
    centers, coeff = alignn_gaussian_terms(0.0, cutoff, bins)
    delta = (centers[1] - centers[0]).item()
    assert abs(float(layer.coeff) - (-1.0 / delta)) < 1e-6
    schnet = -0.5 / delta**2
    assert abs(float(layer.coeff) - schnet) > 1.0
    distance = torch.tensor([0.0, cutoff / 2, cutoff])
    manual = torch.exp(coeff * (distance.unsqueeze(-1) - centers) ** 2)
    assert_allclose(layer._get_rbf(distance).detach().numpy(), manual.detach().numpy(), atol=1e-6)
    # A center of the expansion is 1, up to the stored spacing.
    assert abs(float(layer._get_rbf(centers[:1]).squeeze()[0]) - 1.0) < 1e-5
    try:
        alignn_gaussian_terms(-1.0, 1.0, 1)
    except ValueError as exc:
        assert "bins" in str(exc)
    else:
        raise AssertionError("a one-bin ALIGNN expansion must be rejected")


def test_line_graph_matches_head_to_tail_pairs_and_cosines():
    from enerzyme.models.alignn.geometry import directed_line_graph, line_graph_cosine

    # Star: center 0, neighbor 1 on +x, neighbor 2 on +y. Both directions.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    idx_i = torch.tensor([1, 0, 2, 0])
    idx_j = torch.tensor([0, 1, 0, 2])
    vij = pos[idx_j] - pos[idx_i]
    src, dst = directed_line_graph(idx_i, idx_j, 3)
    brute = []
    for e1 in range(4):
        for e2 in range(4):
            if int(idx_i[e1]) == int(idx_j[e2]):
                brute.append((e1, e2))
    got = sorted((int(a), int(b)) for a, b in zip(src.tolist(), dst.tolist()))
    assert got == sorted(brute)
    cosine = line_graph_cosine(vij, src, dst)
    # r = -vij. ALIGNN cosine is (-r_src) · r_dst. dim=-1 is the vector
    # length; Tensor.norm(-1) would be a scalar p-norm of the whole tensor.
    r = -vij
    manual = ((-r[src]) * r[dst]).sum(-1) / (
        r[src].norm(dim=-1) * r[dst].norm(dim=-1)
    )
    assert_allclose(cosine.detach().numpy(), manual.clamp(-1, 1).detach().numpy(), atol=1e-6)

    # Path 1 -> 0 -> 2 is a right angle. A bond with its reverse has cosine +1.
    right = []
    straight = []
    for e1, e2, value in zip(src.tolist(), dst.tolist(), cosine.tolist()):
        if {int(idx_j[e1]), int(idx_i[e2])} == {1, 2} and int(idx_i[e1]) == 0:
            right.append(value)
        if int(idx_j[e1]) == int(idx_i[e2]) and int(idx_i[e1]) == int(idx_j[e2]):
            straight.append(value)
    assert right
    assert straight
    assert_allclose(right, [0.0] * len(right), atol=1e-5)
    assert_allclose(straight, [1.0] * len(straight), atol=1e-5)

    kept_src, kept_dst = directed_line_graph(idx_i, idx_j, 3, backtracking=False, vij=vij)
    assert kept_src.numel() < src.numel()
    assert kept_src.numel() == len(right)
    kept_cos = line_graph_cosine(vij, kept_src, kept_dst)
    assert_allclose(kept_cos.detach().numpy(), torch.zeros(kept_src.shape[0]).numpy(), atol=1e-5)


def test_periodic_images_are_distinct_line_graph_nodes():
    from enerzyme.models.alignn.geometry import directed_line_graph, line_graph_cosine

    # Two images of atom 1 around atom 0, plus both reverses.
    idx_i = torch.tensor([0, 1, 0, 1])
    idx_j = torch.tensor([1, 0, 1, 0])
    vij = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    src, dst = directed_line_graph(idx_i, idx_j, 2, backtracking=False, vij=vij)
    # The geometric reverses are dropped. The 90-degree image pairs remain.
    cosine = line_graph_cosine(vij, src, dst)
    assert src.numel() == 4
    assert_allclose(cosine.abs().detach().numpy(), torch.zeros(4).numpy(), atol=1e-5)
    try:
        directed_line_graph(idx_i, idx_j, 2, backtracking=False, vij=None)
    except ValueError as exc:
        assert "vij" in str(exc)
    else:
        raise AssertionError("dropping backtracking without vij must be rejected")


def test_colinear_path_has_cosine_minus_one():
    from enerzyme.models.alignn.geometry import directed_line_graph, line_graph_cosine

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    idx_i = torch.tensor([1, 0, 2, 1])
    idx_j = torch.tensor([0, 1, 1, 2])
    vij = pos[idx_j] - pos[idx_i]
    src, dst = directed_line_graph(idx_i, idx_j, 3, backtracking=False, vij=vij)
    cosine = line_graph_cosine(vij, src, dst)
    # 0 -> 1 -> 2 and 2 -> 1 -> 0.
    assert src.numel() == 2
    assert_allclose(cosine.detach().numpy(), torch.tensor([-1.0, -1.0]).numpy(), atol=1e-5)


def _reference_conv(conv, node, edge, src, dst):
    updated = conv.src_update(node)
    if edge.shape[0] == 0:
        updated = F.silu(conv.norm_nodes(updated) if conv.norm_nodes is not None else updated)
        if conv.residual:
            updated = node + updated
        return updated, edge
    message = conv.src_gate(node)[src] + conv.dst_gate(node)[dst] + conv.edge_gate(edge)
    gate = torch.sigmoid(message)
    acc_h = torch.zeros_like(node)
    acc_g = torch.zeros_like(node)
    neighbor = conv.dst_update(node)
    for edge_id in range(edge.shape[0]):
        destination = int(dst[edge_id])
        acc_h[destination] = acc_h[destination] + gate[edge_id] * neighbor[int(src[edge_id])]
        acc_g[destination] = acc_g[destination] + gate[edge_id]
    updated = updated + acc_h / (acc_g + conv.gate_eps)
    node_norm = conv.norm_nodes(updated) if conv.norm_nodes is not None else updated
    edge_norm = conv.norm_edges(message) if conv.norm_edges is not None else message
    updated = F.silu(node_norm)
    updated_edge = F.silu(edge_norm)
    if conv.residual:
        updated = node + updated
        updated_edge = edge + updated_edge
    return updated, updated_edge


def test_edge_gated_conv_matches_per_channel_reference():
    from enerzyme.models.alignn.interaction import EdgeGatedGraphConv

    torch.manual_seed(0)
    conv = EdgeGatedGraphConv(4, "layer", residual=True, gate_eps=1e-6)
    conv.eval()
    node = torch.randn(5, 4)
    src = torch.tensor([0, 1, 2, 3, 4, 0, 2])
    dst = torch.tensor([1, 2, 3, 4, 0, 2, 0])
    edge = torch.randn(src.shape[0], 4)
    got_node, got_edge = conv(node, edge, src, dst)
    ref_node, ref_edge = _reference_conv(conv, node, edge, src, dst)
    assert_allclose(got_node.detach().numpy(), ref_node.detach().numpy(), atol=1e-5)
    assert_allclose(got_edge.detach().numpy(), ref_edge.detach().numpy(), atol=1e-5)

    # The denominator is per channel. A scalar gate would mix these columns.
    with torch.no_grad():
        conv.edge_gate.weight.zero_()
        conv.edge_gate.bias.copy_(torch.tensor([4.0, -4.0, 0.0, 1.0]))
        conv.src_gate.weight.zero_()
        conv.src_gate.bias.zero_()
        conv.dst_gate.weight.zero_()
        conv.dst_gate.bias.zero_()
    gated, _ = conv(node, edge, src, dst)
    ref_gated, _ = _reference_conv(conv, node, edge, src, dst)
    assert_allclose(gated.detach().numpy(), ref_gated.detach().numpy(), atol=1e-5)

    alone, alone_edge = conv(node[:1], node.new_zeros(0, 4), src.new_empty(0), dst.new_empty(0))
    assert alone.shape == (1, 4)
    assert alone_edge.shape == (0, 4)
    assert torch.isfinite(alone).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Inductor")
def test_compiled_gate_preserves_live_hooks_and_force_gradients():
    from enerzyme.models.alignn.interaction import EdgeGatedGraphConv

    torch.manual_seed(31)
    conv = EdgeGatedGraphConv(4, "layer").cuda().double().eval()
    node = torch.randn(3, 4, device="cuda", dtype=torch.double, requires_grad=True)
    src = torch.tensor([0, 1, 2, 1], device="cuda")
    dst = torch.tensor([1, 2, 1, 0], device="cuda")
    edge = torch.randn(4, 4, device="cuda", dtype=torch.double, requires_grad=True)
    calls = torch.zeros((), device="cuda")
    compiled = torch.compile(conv)

    def result(run):
        atom, bond = run(node, edge, src, dst)
        loss = atom.square().sum() + bond.square().sum()
        grads = torch.autograd.grad(loss, (node, edge, conv.edge_gate.weight))
        return (atom, bond, *grads)

    def modify_gate(_, inputs, output):
        calls.add_(1)
        return 0.5 * output

    def check():
        expected = result(conv)
        calls.zero_()
        actual = result(compiled)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)

    check()
    hook = conv.edge_gate.register_forward_hook(modify_gate)
    check()
    assert calls.item() == 1
    hook.remove()
    check()
    assert calls.item() == 0


def test_alignn_layer_orders_disagree_on_which_state_sees_the_angle():
    from enerzyme.models.alignn.interaction import ALIGNNConv

    torch.manual_seed(0)
    layer = ALIGNNConv(4, "layer")
    layer.eval()
    atom = torch.randn(3, 4)
    bond = torch.randn(3, 4)
    angle_a = torch.randn(2, 4)
    angle_b = angle_a + 1.5
    # Bond 0 has two incoming line-graph edges, so the gate does not cancel
    # and the bond state follows the angle. Atom 0 receives that bond and
    # bond 1, so the paper-order atom update can see it. A destination with
    # one incoming edge would divide the gate out.
    atom_src = torch.tensor([1, 2, 0])
    atom_dst = torch.tensor([0, 0, 1])
    line_src = torch.tensor([1, 2])
    line_dst = torch.tensor([0, 0])
    code_a = layer(atom, bond, angle_a, atom_src, atom_dst, line_src, line_dst, "code")
    code_b = layer(atom, bond, angle_b, atom_src, atom_dst, line_src, line_dst, "code")
    paper_a = layer(atom, bond, angle_a, atom_src, atom_dst, line_src, line_dst, "paper")
    paper_b = layer(atom, bond, angle_b, atom_src, atom_dst, line_src, line_dst, "paper")
    # Released order updates atoms before the line graph, so one layer's atoms
    # do not see the angle. The bond state does. Paper order runs the line
    # graph first, then the atom graph reads those bonds.
    assert_allclose(code_a[0].detach().numpy(), code_b[0].detach().numpy(), atol=1e-6)
    assert float((code_a[1] - code_b[1]).detach().abs().max()) > 1e-3
    assert float((code_a[2] - code_b[2]).detach().abs().max()) > 1e-2
    assert float((paper_a[0] - paper_b[0]).detach().abs().max()) > 1e-5
    try:
        layer(atom, bond, angle_a, atom_src, atom_dst, line_src, line_dst, "other")
    except ValueError as exc:
        assert "update_order" in str(exc)
    else:
        raise AssertionError("an unknown update order must be rejected")
