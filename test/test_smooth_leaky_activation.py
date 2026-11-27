import pytest
import torch

from enerzyme.models.activation import SmoothLeakyReLU


_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
))]


def _reference(x, slope):
    return ((1 + slope) / 2) * x + ((1 - slope) / 2) * x * (2 * torch.sigmoid(x) - 1)


@pytest.mark.parametrize("slope", [0.0, 0.2, 1.0, -0.25, 1.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", _DEVICES)
def test_attention_activation_derivatives(slope, dtype, device):
    x = torch.linspace(-40, 40, 204, dtype=dtype, device=device).reshape(17, 12).T.requires_grad_()
    upstream = torch.linspace(0.1, 1.5, x.numel(), dtype=dtype, device=device).reshape_as(x)
    activation = SmoothLeakyReLU(slope)
    assert not activation.state_dict()
    actual = activation(x)
    expected = _reference(x, slope)
    for order in range(3):
        if dtype == torch.float64:
            torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
        if order < 2:
            actual, = torch.autograd.grad(actual, x, upstream, create_graph=True)
            expected, = torch.autograd.grad(expected, x, upstream, create_graph=True)


@pytest.mark.parametrize("device", _DEVICES)
def test_attention_activation_compiled_second_derivative(device):
    x = torch.linspace(-12, 12, 91, dtype=torch.float64, device=device)
    slope = 0.2
    activation = SmoothLeakyReLU(slope)

    def second_derivative(value):
        gradient = torch.func.grad(lambda t: activation(t).sum())
        return torch.func.jvp(gradient, (value,), (torch.ones_like(value),))[1]

    compiled = torch.compile(second_derivative, backend="aot_eager", fullgraph=True)
    expect = torch.func.grad(torch.func.grad(lambda v: _reference(v, slope)))
    torch.testing.assert_close(compiled(x), torch.vmap(expect)(x))


def test_attention_activation_nonfinite_and_integer_inputs():
    activation = SmoothLeakyReLU(0.2)
    logits = torch.tensor([-torch.inf, torch.inf, torch.nan])
    torch.testing.assert_close(activation(logits), _reference(logits, 0.2), equal_nan=True)
    indices = torch.arange(-3, 4)
    torch.testing.assert_close(activation(indices), _reference(indices, 0.2))
