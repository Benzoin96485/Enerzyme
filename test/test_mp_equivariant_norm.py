"""Spherical-array normalization reproduces each upstream formula and its gradients."""

import importlib.util
from pathlib import Path

import pytest
import torch

from enerzyme.models.mp.equivariant.sph import layer_norm
from enerzyme.models.mp.nn.norm import EquivariantNorm


def upstream(version):
    source = (
        Path(__file__).parent
        / "fixtures"
        / f"equiformer_{version}_upstream"
        / "layer_norm.py"
    )
    spec = importlib.util.spec_from_file_location(f"reference_norm_{version}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "version,public,reference",
    [
        ("v2", "EquivariantLayerNormArray", "EquivariantLayerNormArray"),
        (
            "v2",
            "EquivariantLayerNormArraySphericalHarmonics",
            "EquivariantLayerNormArraySphericalHarmonics",
        ),
        (
            "v2",
            "EquivariantRMSNormArraySphericalHarmonicsV2",
            "EquivariantRMSNormArraySphericalHarmonicsV2",
        ),
        ("v3", "EquivariantSeparableLayerNorm", "EquivariantSeparableLayerNorm"),
        ("v3", "EquivariantMergeLayerNorm", "EquivariantMergeLayerNorm"),
    ],
)
@pytest.mark.parametrize("lmax,affine", [(0, True), (2, True), (3, False)])
def test_reference_values_and_gradients(version, public, reference, lmax, affine):
    torch.manual_seed(19)
    ref = getattr(upstream(version), reference)(lmax, 5, affine=affine).double()
    new = getattr(layer_norm, public)(lmax, 5, affine=affine).double()
    with torch.no_grad():
        for param in ref.parameters():
            param.normal_()
    new.load_state_dict(ref.state_dict())
    data = torch.randn(3, (lmax + 1) ** 2, 5, dtype=torch.float64)
    probe = torch.randn_like(data)
    values = []
    grads = []
    for module in (ref, new):
        x = data.clone().requires_grad_()
        y = module(x)
        values.append(y)
        grads.append(
            torch.autograd.grad(
                (y * probe).sum(), (x, *module.parameters()), allow_unused=True
            )
        )
    torch.testing.assert_close(values[0], values[1], rtol=1e-12, atol=1e-12)
    for left, right in zip(*grads):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["inside", "outside", "floor"])
def test_epsilon_placement(mode):
    eps = 0.25
    norm = EquivariantNorm(
        1, 2, kind="rms", scope="global", eps=eps, eps_mode=mode, affine=False
    )
    x = torch.arange(8, dtype=torch.float64).reshape(1, 4, 2) / 20.0
    moment = x.square().mean()
    if mode == "inside":
        expected = x / (moment + eps).sqrt()
    elif mode == "outside":
        expected = x / (moment.sqrt() + eps)
    else:
        expected = x / moment.clamp_min(eps).sqrt()
    torch.testing.assert_close(norm(x), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("variant", ["nequix", "eqnorm", "equflash"])
def test_flat_rms_matches_block_reductions_and_gradients(variant):
    from e3nn.o3 import Irreps
    from enerzyme.models.mp.equivariant.e3nn.rms_norm import (
        EqnormRMSNorm,
        EquivariantRMSNormSH,
        RMSLayerNorm,
    )

    irreps = Irreps("4x0e+2x1o+3x2e")
    classes = {
        "nequix": RMSLayerNorm,
        "eqnorm": EqnormRMSNorm,
        "equflash": EquivariantRMSNormSH,
    }
    layer = classes[variant](irreps, eps=1e-4, centering=True).double()
    with torch.no_grad():
        for param in layer.parameters():
            param.uniform_(-1.2, 1.2)
    x = torch.randn(5, irreps.dim, dtype=torch.float64, requires_grad=True)
    parts = [
        x[:, sl].reshape(len(x), mul, ir.dim)
        for sl, (mul, ir) in zip(irreps.slices(), irreps)
    ]
    parts[0] = parts[0] - parts[0].mean(dim=1, keepdim=True)
    if variant == "eqnorm":
        channels = [
            torch.stack(
                [v[:, c].square().mean(-1) for v in parts if v.size(1) > c]
            ).mean(0)
            for c in range(4)
        ]
        moment = torch.stack(channels).mean(0)
    elif variant == "nequix":
        moment = sum(v.square().sum(1).mean(-1) / v.shape[-1] for v in parts) / len(
            irreps
        )
    else:
        moment = sum(v.square().mean((1, 2)) for v in parts) / len(irreps)
    gain_blocks = (
        layer.affine_weight.split([mul for mul, _ in irreps])
        if variant == "equflash"
        else layer.affine_weight
    )
    reference = [
        v / (moment + 1e-4).sqrt()[:, None, None] * gain[None, :, None]
        for v, gain in zip(parts, gain_blocks)
    ]
    bias = layer.affine_bias[0] if variant == "nequix" else layer.affine_bias
    reference[0] = reference[0] + bias[None, :, None]
    expected = torch.cat([v.flatten(1) for v in reference], dim=1)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=5e-8, atol=5e-8)
    probe = torch.randn_like(x)
    variables = (x, *layer.parameters())
    for actual_grad, ref_grad in zip(
        torch.autograd.grad(actual, variables, probe),
        torch.autograd.grad(expected, variables, probe),
    ):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=5e-8, atol=5e-8)
