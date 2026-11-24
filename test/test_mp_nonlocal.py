"""Masked global attention agrees with a dense per-head computation."""

import math
import torch
from enerzyme.models.mp.nonlocal_ import GlobalAttention, Performer


def test_global_attention_values_and_gradients():
    layer = GlobalAttention(6, 2, 0).double()
    q, k, v = [
        torch.randn(2, n, 6, dtype=torch.float64, requires_grad=True) for n in (4, 5, 5)
    ]
    valid = torch.tensor([True, False, True, True, False])
    mask = valid.expand(2, 2, 4, -1)
    out = layer(q, k, v, mask)
    a, b, c = layer.qkv_projection(q, k, v)
    scores = (a @ b.transpose(-1, -2)) / math.sqrt(a.shape[-1])
    mix = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ c
    expected = layer.attn_out_proj(mix.transpose(1, 2).reshape(2, 4, 6))
    probe = torch.randn_like(out)
    torch.testing.assert_close(out, expected)
    variables = (q, k, v, *layer.parameters())
    for actual, ref in zip(
        torch.autograd.grad(out, variables, probe),
        torch.autograd.grad(expected, variables, probe),
    ):
        torch.testing.assert_close(actual, ref)


def test_performer_exact_masks_other_molecules():
    layer = Performer(3)
    q, k, v = [torch.randn(4, 3, dtype=torch.float64) for _ in range(3)]
    batch = torch.tensor([0, 0, 1, 1])
    dots = q @ k.T
    kernel = ((dots - dots.max()) / math.sqrt(3)).exp() * (
        batch[:, None] == batch[None, :]
    )
    expected = (kernel / (kernel.sum(-1, keepdim=True) + 1e-8)) @ v
    torch.testing.assert_close(layer(q, k, v, 2, batch), expected)
