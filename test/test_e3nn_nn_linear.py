"""Numerical / gradient parity for e3nn_nn linear bias application."""

import torch
import torch.nn.functional as F
from e3nn import o3

from enerzyme.models.e3nn_nn.linear import (
    ElementIrrepsLinear,
    IrrepsLinear,
    _add_0e_bias,
    _collect_0e_slices,
)


def _legacy_add_0e_bias_clone_per_slice(out, bias, out_slices, bias_slices):
    """Original ElementIrrepsLinear path: clone once per 0e slice."""
    if bias.ndim == 1:
        for sl, bsl in zip(out_slices, bias_slices):
            out = out.clone()
            out[:, sl] = out[:, sl] + bias[bsl].unsqueeze(0)
    else:
        for sl, bsl in zip(out_slices, bias_slices):
            out = out.clone()
            out[:, sl] = out[:, sl] + bias[:, bsl]
    return out


def _legacy_add_0e_bias_clone_once(out, bias, out_slices, bias_slices):
    """Post-refactor path: single clone then in-place slice updates."""
    out = out.clone()
    if bias.ndim == 1:
        for sl, bsl in zip(out_slices, bias_slices):
            out[:, sl] = out[:, sl] + bias[bsl].unsqueeze(0)
    else:
        for sl, bsl in zip(out_slices, bias_slices):
            out[:, sl] = out[:, sl] + bias[:, bsl]
    return out


def test_add_0e_bias_matches_legacy_paths_values_and_grads():
    torch.manual_seed(0)
    irreps = o3.Irreps("4x0e+3x1o+2x0e+1x2e")
    out_slices, bias_slices, bias_dim = _collect_0e_slices(irreps)
    assert bias_dim == 6 and len(out_slices) == 2

    n = 5
    base = torch.randn(n, irreps.dim, requires_grad=True)
    bias = torch.randn(bias_dim, requires_grad=True)
    bias_n = torch.randn(n, bias_dim, requires_grad=True)

    for b in (bias, bias_n):
        outs = []
        for fn in (
            _add_0e_bias,
            _legacy_add_0e_bias_clone_once,
            _legacy_add_0e_bias_clone_per_slice,
        ):
            x = base.detach().requires_grad_(True)
            bb = b.detach().requires_grad_(True)
            y = fn(x, bb, out_slices, bias_slices)
            y.sum().backward()
            outs.append((y.detach(), x.grad.detach(), bb.grad.detach()))
        ref_y, ref_gx, ref_gb = outs[0]
        for y, gx, gb in outs[1:]:
            assert torch.allclose(y, ref_y, atol=1e-7, rtol=1e-6)
            assert torch.allclose(gx, ref_gx, atol=1e-7, rtol=1e-6)
            assert torch.allclose(gb, ref_gb, atol=1e-7, rtol=1e-6)


def test_irreps_linear_bias_values_and_grads():
    torch.manual_seed(1)
    ir_in = o3.Irreps("8x0e+4x1o")
    ir_out = o3.Irreps("4x0e+2x1o+2x0e")
    layer = IrrepsLinear(ir_in, ir_out, bias=True)
    with torch.no_grad():
        layer.bias.uniform_(-0.2, 0.2)

    x = torch.randn(6, ir_in.dim, requires_grad=True)
    y = layer(x)
    assert y.shape == (6, ir_out.dim)
    # Non-0e channels unchanged vs linear-only; 0e channels shifted by bias.
    with torch.no_grad():
        y_nobias = layer.linear(x, layer.weight)
    out_slices, bias_slices, _ = _collect_0e_slices(ir_out)
    for sl, bsl in zip(out_slices, bias_slices):
        assert torch.allclose(y[:, sl], y_nobias[:, sl] + layer.bias[bsl], atol=1e-6)
    # vector channels: identical to linear (no bias)
    vec = slice(4, 4 + 2 * 3)  # after first 4x0e
    assert torch.allclose(y[:, vec], y_nobias[:, vec], atol=1e-6)

    loss = y.pow(2).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
    assert layer.bias.grad.abs().sum() > 0


def test_element_irreps_linear_bias_values_and_grads():
    torch.manual_seed(2)
    ir_in = o3.Irreps("8x0e+4x1o")
    ir_out = o3.Irreps("4x0e+2x1o+2x0e")
    num_elements = 5
    layer = ElementIrrepsLinear(ir_in, ir_out, num_elements=num_elements, bias=True)
    with torch.no_grad():
        layer.bias.uniform_(-0.2, 0.2)

    n = 7
    x = torch.randn(n, ir_in.dim, requires_grad=True)
    za = torch.tensor([1, 3, 0, 2, 4, 1, 0])
    attrs = F.one_hot(za, num_classes=num_elements).to(dtype=x.dtype)
    y = layer(x, attrs)

    weight = torch.einsum("ne,ew->nw", attrs, layer.weight)
    with torch.no_grad():
        y_nobias = layer.linear(x, weight)
        b = torch.einsum("ne,eb->nb", attrs, layer.bias)
    out_slices, bias_slices, _ = _collect_0e_slices(ir_out)
    for sl, bsl in zip(out_slices, bias_slices):
        assert torch.allclose(y[:, sl], y_nobias[:, sl] + b[:, bsl], atol=1e-6)
    vec = slice(4, 4 + 2 * 3)
    assert torch.allclose(y[:, vec], y_nobias[:, vec], atol=1e-6)

    y.pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias.grad).all()
    assert layer.bias.grad.abs().sum() > 0
