import pytest
import torch

from enerzyme.models.mp.ops import (
    GraphSoftmax,
    safe_norm,
    safe_normalize,
    scatter_max,
    scatter_mean,
    scatter_sum,
    segment_softmax,
    segment_sum_csr,
)
from enerzyme.models.mp.ops import scatter as scatter_mod


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


def _edges(n_nodes=7, n_edges=40, width=5, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    index = torch.randint(0, n_nodes, (n_edges,), generator=g)
    src = torch.randn(n_edges, 3, width, generator=g, dtype=dtype)
    return src, index


@pytest.mark.parametrize("with_null", [False, True])
def test_segment_softmax_with_envelope_matches_dense_values_and_gradients(with_null):
    logits = torch.tensor([[5., -2., 1., 3.], [-4., 2., 1., 1.]], dtype=torch.float64, requires_grad=True)
    envelope = torch.tensor([0.25, 0., 0.75, 0.], dtype=logits.dtype, requires_grad=True)
    index = torch.tensor([1, 0, 1, 2])
    null = torch.tensor([0.3, -0.4], dtype=logits.dtype, requires_grad=True) if with_null else None
    weights = segment_softmax(logits, index, 4, dim=1, envelope=envelope, null_logit=null)
    active = envelope > 0
    numer = logits.exp() * torch.where(active, envelope, 0.)
    denom = torch.stack([numer[:, index == group].sum(1) for group in range(4)], dim=1)
    if null is not None:
        denom = denom + null.exp().unsqueeze(1)
    expected = numer / denom.masked_fill(denom == 0, 1.)[:, index]
    torch.testing.assert_close(weights, expected)
    direction = torch.arange(weights.numel(), dtype=weights.dtype).reshape_as(weights)
    inputs = (logits, envelope, null) if with_null else (logits, envelope)
    actual_grad = torch.autograd.grad(weights, inputs, direction)
    expected_grad = torch.autograd.grad(expected, inputs, direction)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference)


def test_scatter_sum_backends_bitwise(dtype, monkeypatch):
    torch_scatter = pytest.importorskip("torch_scatter")
    src, index = _edges(dtype=dtype)
    fast = scatter_sum(src, index, 9)
    monkeypatch.setattr(scatter_mod, "TORCH_SCATTER_MAX_NUMEL", -1)
    native = scatter_sum(src, index, 9)
    reference = torch_scatter.scatter_sum(src, index, dim=0, dim_size=9)
    assert torch.equal(fast, native)
    assert torch.equal(fast, reference)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_scatter_sum_any_dim(dim):
    torch_scatter = pytest.importorskip("torch_scatter")
    src = torch.randn(6, 8, 4, dtype=torch.float64)
    size = src.shape[dim]
    index = torch.randint(0, 3, (size,))
    out = scatter_sum(src, index, 3, dim=dim)
    assert torch.equal(out, torch_scatter.scatter_sum(src, index, dim=dim, dim_size=3))


def test_scatter_sum_empty_and_inferred_size():
    src = torch.zeros(0, 4)
    out = scatter_sum(src, torch.zeros(0, dtype=torch.long), 5)
    assert out.shape == (5, 4) and not out.any()
    src, index = _edges(n_nodes=4)
    assert scatter_sum(src, index).shape[0] == int(index.max()) + 1


def test_scatter_mean_max():
    torch_scatter = pytest.importorskip("torch_scatter")
    src, index = _edges()
    assert torch.allclose(scatter_mean(src, index, 9), torch_scatter.scatter_mean(src, index, dim=0, dim_size=9))
    ref_max = torch_scatter.scatter_max(src, index, dim=0, dim_size=9)[0]
    assert torch.equal(scatter_max(src, index, 9), ref_max)


def test_segment_sum_csr_matches_scatter():
    src, index = _edges()
    order = torch.argsort(index, stable=True)
    sorted_src, sorted_index = src[order], index[order]
    counts = torch.bincount(sorted_index, minlength=9)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    assert torch.allclose(segment_sum_csr(sorted_src, ptr), scatter_sum(sorted_src, sorted_index, 9))


def test_scatter_sum_compiles():
    src, index = _edges(dtype=torch.float32)

    def fn(x):
        return scatter_sum(x * 2.0, index, 9)

    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert torch.allclose(compiled(src), fn(src))


def test_inductor_cpu_single_destination_scalar_sum():
    compiled = torch.compile(scatter_sum, backend="inductor")
    values = torch.tensor([1., -2., 0.5, 3., 4.], dtype=torch.float64, requires_grad=True)
    for index in (torch.zeros(5, dtype=torch.long), torch.tensor([1, 0, 0, 1, 0])):
        expected = torch.stack([values[index == i].sum() for i in range(index.max().item() + 1)])
        for size in (None, len(expected)):
            result = compiled(values, index, dim_size=size)
            torch.testing.assert_close(result, expected)
            probe = torch.arange(1, len(result) + 1, dtype=values.dtype)
            torch.testing.assert_close(torch.autograd.grad(result, values, probe)[0], probe[index])


def test_segment_softmax_normalises_and_matches_pyg_style():
    logits = torch.randn(30, 4, dtype=torch.float64)
    index = torch.randint(0, 6, (30,))
    weights = segment_softmax(logits, index, 6)
    totals = scatter_sum(weights, index, 6)
    present = torch.bincount(index, minlength=6) > 0
    assert torch.allclose(totals[present], torch.ones_like(totals[present]))
    graph = GraphSoftmax(eps=1e-16)(logits, index, num_nodes=6)
    assert torch.allclose(graph, weights, atol=1e-14)


def test_segment_softmax_exp_rescale_and_grad():
    logits = torch.randn(20, dtype=torch.float64, requires_grad=True)
    index = torch.randint(0, 4, (20,))
    rescale = torch.rand(20, dtype=torch.float64)
    a = segment_softmax(logits, index, 4, eps=1e-16, exp_rescale=rescale)
    b = segment_softmax(logits, index, 4, eps=1e-16, exp_rescale=rescale, detach_max=True)
    assert torch.allclose(a, b, atol=1e-15)
    ga, = torch.autograd.grad(a.square().sum(), logits)
    gb, = torch.autograd.grad(b.square().sum(), logits)
    assert torch.allclose(ga, gb, atol=1e-13)


@pytest.mark.parametrize("mode", ["sqrt_eps2", "clamp", "clamp_zero", "norm_clamp"])
def test_safe_norm_modes(mode):
    x = torch.randn(10, 3, dtype=torch.float64)
    x[0] = 0.0
    norm = safe_norm(x, mode=mode, eps=1e-12)
    assert torch.allclose(norm[1:], x[1:].norm(dim=-1), atol=1e-6)
    unit = safe_normalize(x, mode=mode, eps=1e-12)
    assert torch.isfinite(unit).all()
    assert torch.allclose(unit[1:].norm(dim=-1), torch.ones(9, dtype=torch.float64), atol=1e-6)
    if mode == "clamp_zero":
        assert norm[0] == 0 and not unit[0].any()
