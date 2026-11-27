"""Receiver-wise aggregation of scalar and equivariant feature streams."""

import pytest
import torch
from torch.nn import functional as F

from enerzyme.models.mp.aggregate import Aggregate, EnvelopeGated, SoftmaxAttention
from enerzyme.models.mp.message.attention import dot_product_attention_logits


def test_density_sum_and_gradients_with_isolated_nodes():
    aggregate = Aggregate(mode="density", divisor=2.0)
    index = torch.tensor([1, 0, 1, 3])
    messages = torch.randn(4, 3, 2, dtype=torch.float64, requires_grad=True)
    density = torch.rand(4, 1, dtype=torch.float64, requires_grad=True)
    result = aggregate.reduce(messages, index, 5, density)
    expected = torch.stack(
        [
            messages[index == node].sum(0) / (1 + density[index == node].sum()) / 2
            for node in range(5)
        ]
    )
    torch.testing.assert_close(result, expected)
    probe = torch.randn_like(result)
    for a, b in zip(
        torch.autograd.grad(result, (messages, density), probe),
        torch.autograd.grad(expected, (messages, density), probe),
    ):
        torch.testing.assert_close(a, b)


def test_density_sum_empty_graph():
    result = Aggregate(mode="density").reduce(
        torch.empty(0, 3), torch.empty(0, dtype=torch.long), 0, torch.empty(0)
    )
    assert result.shape == (0, 3)


def test_ordered_denominator_preserves_nonleading_axis_and_second_derivatives():
    from enerzyme.models.mp.ops import segment_softmax

    index = torch.tensor([2, 0, 2, 0, 1])
    scores = torch.tensor([[.2, -.3, .5, .8, -.1], [.7, .4, .1, -.6, -.2]],
                          dtype=torch.float64, requires_grad=True)

    def derivatives(ordered):
        weights = segment_softmax(scores, index, 4, dim=-1, ordered_sum=ordered)
        gradient, = torch.autograd.grad(weights.square().sum(), scores, create_graph=True)
        curvature, = torch.autograd.grad(gradient.square().sum(), scores)
        return weights, gradient, curvature

    torch.testing.assert_close(derivatives(True), derivatives(False), rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA receiver reductions")
def test_double_cuda_envelope_training_uses_the_ordered_fp32_denominator():
    from enerzyme.models.mp.ops import segment_envelope_gated_softmax

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(11)
    edges, nodes = 8192, 31
    index = torch.randint(nodes - 1, (edges,), generator=gen, device=device)
    logits = torch.randn(edges, 1, 1, dtype=torch.float64, generator=gen, device=device)
    envelope = torch.rand(edges, dtype=torch.float64, generator=gen, device=device)
    bias = logits.new_zeros(1, 1)
    values = logits.new_ones(edges, 1, 1, 1, requires_grad=True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        reference = segment_envelope_gated_softmax(logits, envelope, index, nodes, bias)
        torch.use_deterministic_algorithms(False)
        result = EnvelopeGated(ordered_sum=True)(values, logits, envelope, index, nodes, bias)
        weights, = torch.autograd.grad(result.sum(), values)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.testing.assert_close(weights.flatten(), reference.flatten(), rtol=0, atol=0)


def test_dpa4_opts_in_to_ordered_sum_only_for_double_cuda_training(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from enerzyme.models.dpa4.so2 import SO2Convolution
    from enerzyme.models.mp import aggregate

    selected = []

    def record(logits, *args, ordered_sum):
        selected.append(ordered_sum)
        return logits

    monkeypatch.setattr(aggregate, "segment_envelope_gated_softmax", record)
    monkeypatch.setattr(aggregate, "scatter_sum", lambda values, *_: values)
    default = EnvelopeGated()
    legacy = EnvelopeGated()
    del legacy.ordered_sum
    dpa4 = SO2Convolution(lmax=1, mmax=1, channels=2).aggregate
    with FakeTensorMode():
        for module, training, device, dtype in (
            (default, True, "cuda", torch.float64), (legacy, True, "cuda", torch.float64),
            (dpa4, True, "cuda", torch.float64), (dpa4, False, "cuda", torch.float64),
            (dpa4, True, "cuda", torch.float32), (dpa4, True, "cpu", torch.float64),
        ):
            logits = torch.empty(3, 1, 1, dtype=dtype, device=device)
            module.train(training)(logits.unsqueeze(-1), logits, logits[:, 0, 0],
                                   torch.zeros(3, dtype=torch.long, device=device), 1, logits[0])
    assert selected == [False, False, True, False, False, False]


def test_gated_softmax_value_and_score_gradients():
    # Node 2 is isolated. One edge has a closed value gate but still
    # contributes to the softmax denominator, as in MatterSim.
    receiver = torch.tensor([1, 0, 1, 1, 0])
    sender = torch.tensor([0, 1, 0, 2, 2])
    query = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    values = torch.randn_like(query, requires_grad=True)
    gate = torch.tensor([[1.], [.5], [0.], [.25], [1.]], dtype=query.dtype, requires_grad=True)
    logits = dot_product_attention_logits(query[receiver], key[sender], scale=0.5)
    actual = SoftmaxAttention(clamp_min=1e-12)(values[sender], logits, receiver, 3, gate)
    reference_logits = torch.einsum("ehc,ehc->eh", query[receiver], key[sender]) / 2
    groups = []
    for node in range(3):
        selected = receiver == node
        weights = reference_logits[selected].softmax(dim=0) * gate[selected]
        groups.append((weights.unsqueeze(-1) * values[sender[selected]]).sum(dim=0))
    expected = torch.stack(groups)
    torch.testing.assert_close(actual, expected)
    inputs, probe = (query, key, values, gate), torch.randn_like(expected)
    for a, b in zip(torch.autograd.grad(actual, inputs, probe), torch.autograd.grad(expected, inputs, probe)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_envelope_attention_gradients_against_null_mass_formula(dtype):
    # The two focuses have three attention heads each. DPA4 applies their
    # mean per focus to every spherical coefficient and channel.
    receiver = torch.tensor([0, 1, 0, 1, 0])
    values = torch.randn(5, 4, 2, 3, dtype=dtype, requires_grad=True)
    logits = torch.randn(5, 2, 3, dtype=dtype, requires_grad=True)
    envelope = torch.tensor([.25, .75, 0., 1., .5], dtype=dtype, requires_grad=True)
    null_bias = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    eps = 1e-5
    actual = EnvelopeGated()(values, logits, envelope, receiver, 3, null_bias, eps)

    masses = logits.float().exp() * envelope.float().square().view(5, 1, 1)
    null_mass = F.softplus(null_bias.float()) + eps
    groups = []
    for node in range(3):
        selected = receiver == node
        group_mass = masses[selected]
        weights = group_mass / (group_mass.sum(dim=0) + null_mass)
        weights = weights.to(dtype).mean(dim=-1)[:, None, :, None]
        groups.append((weights * values[selected]).sum(dim=0))
    expected = torch.stack(groups)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    inputs, probe = (values, logits, envelope, null_bias), torch.randn_like(expected)
    for a, b in zip(torch.autograd.grad(actual, inputs, probe), torch.autograd.grad(expected, inputs, probe)):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
