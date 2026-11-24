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
