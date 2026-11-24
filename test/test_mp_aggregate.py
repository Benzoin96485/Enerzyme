"""Density-normalized aggregation across two feature streams."""

import torch
from enerzyme.models.mp.aggregate import Aggregate


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
