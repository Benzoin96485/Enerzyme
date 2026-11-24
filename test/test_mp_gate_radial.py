"""Reference coverage for shared radial kernels and spherical gates."""

import pytest
import torch
from e3nn.nn import FullyConnectedNet
from torch.nn import functional as F

from enerzyme.models.mp.equivariant.s2.activation import GateActivation as GateV2
from enerzyme.models.mp.equivariant.s2.activation_v3 import GateActivation as GateV3
from enerzyme.models.mp.nn.activation import SwiGLU
from enerzyme.models.mp.nn.radial_mlp import RadialMLP


def test_e3nn_radial_output_and_derivatives():
    kernel = RadialMLP([4, 7, 3], e3nn_style=True).double()
    ref = FullyConnectedNet([4, 7, 3], F.silu).double()
    ref.load_state_dict(kernel.net.state_dict())
    x = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    a, b = kernel(x), ref(x)
    torch.testing.assert_close(a, b)
    probe = torch.randn_like(a)
    for i, j in zip(
        torch.autograd.grad(a, (x, *kernel.parameters()), probe),
        torch.autograd.grad(b, (x, *ref.parameters()), probe),
    ):
        torch.testing.assert_close(i, j)


@pytest.mark.parametrize("gate_first", [False, True])
def test_swiglu_order(gate_first):
    x = torch.randn(2, 6, dtype=torch.float64, requires_grad=True)
    a, b = x.chunk(2, -1)
    expect = F.silu(a) * b if gate_first else a * F.silu(b)
    actual = SwiGLU(gate_first)(x)
    torch.testing.assert_close(actual, expect)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expect.sum(), x)[0]
    )


@pytest.mark.parametrize("m_major", [False, True])
def test_spherical_gates_match_degree_loop(m_major):
    # lmax=3, mmax=1 cuts off high |m| within each degree.
    degrees = [1, 2, 3] * 3 if m_major else [1] * 3 + [2] * 3 + [3] * 3
    x = torch.randn(4, 10, 5, requires_grad=True)
    gates = torch.randn(4, 3, 5, requires_grad=True)
    expected = torch.cat(
        [
            F.silu(x[:, :1]),
            *[
                x[:, i + 1 : i + 2] * gates[:, l - 1 : l].sigmoid()
                for i, l in enumerate(degrees)
            ],
        ],
        dim=1,
    )
    flat = gates.flatten(1)
    candidates = [GateV3(3, 1, m_major)(x, flat)]
    if not m_major:
        candidates.append(GateV2(3, 1, 5)(flat, x))
    probe = torch.randn_like(x)
    gradients = torch.autograd.grad(expected, (x, gates), probe)
    for out in candidates:
        torch.testing.assert_close(out, expected)
        for a, b in zip(torch.autograd.grad(out, (x, gates), probe), gradients):
            torch.testing.assert_close(a, b)
