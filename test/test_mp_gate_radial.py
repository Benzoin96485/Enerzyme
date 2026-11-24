"""Reference coverage for shared radial kernels and spherical nonlinearities."""

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


@pytest.mark.parametrize("family,m_major", [("v2", False), ("v3", False), ("v3", True)])
def test_separable_grid_preserves_scalar_and_grid_dropout_order(family, m_major):
    from enerzyme.models.mp.equivariant.s2.activation import SeparableS2Activation as V2
    from enerzyme.models.mp.equivariant.s2.activation_v3 import SeparableS2Activation as V3
    from enerzyme.models.mp.equivariant.s2.grid import SO3Grid

    grid_act = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Dropout(0.25))
    if family == "v2":
        grid = SO3Grid(2, 1).double()
        activation = V2(2, 1)
        scalar_act = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Dropout(0.125))
        activation.scalar_act = scalar_act
        activation.s2_act.act = grid_act
        scalar_shape = (3, 1, 4)
    else:
        activation = V3(2, 1, grid_resolution_list=[6, 5], use_m_primary=m_major).double()
        grid = activation.so3_grid
        activation.act = scalar_act = grid_act
        scalar_shape = (3, 4)

    tensors = torch.randn(3, grid.n_coeff, 4, dtype=torch.float64, requires_grad=True)
    scalars = torch.randn(scalar_shape, dtype=torch.float64, requires_grad=True)
    torch.manual_seed(73)
    scalar_output = scalar_act(scalars).reshape(3, 1, 4)
    samples = torch.einsum("gd,ndc->ngc", grid.to_grid_mat, tensors)
    coefficients = torch.einsum("dg,ngc->ndc", grid.from_grid_mat, grid_act(samples))
    expected = torch.cat((scalar_output, coefficients[:, 1:]), dim=1)

    torch.manual_seed(73)
    if family == "v2":
        actual = activation(input_scalars=scalars, input_tensors=tensors, SO3_grid={2: {1: grid}})
    else:
        actual = activation(inputs=tensors, scalars=scalars)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (scalars, tensors), probe)
    expected_grads = torch.autograd.grad(expected, (scalars, tensors), probe)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference)
