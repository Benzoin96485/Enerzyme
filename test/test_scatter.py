"""Production scatter sums against torch_scatter on CPU and CUDA."""

import pytest
import torch

from enerzyme.models.mp.ops import scatter_sum


def _assert_sum(actual, reference, src, index, dim):
    if not src.is_cuda:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        return
    # CUDA atomics can sum the same destination in different orders. Compare
    # each result to a double-precision reference using the standard forward
    # error bound for n floating-point additions, gamma_(n-1) * sum(abs(x)).
    values = src.cpu().double()
    destinations = index.cpu()
    while destinations.ndim < values.ndim:
        destinations = destinations.unsqueeze(-1)
    destinations = destinations.expand_as(values)
    zeros = values.new_zeros(actual.shape)
    exact = zeros.scatter_add(dim, destinations, values)
    absolute_sum = zeros.scatter_add(dim, destinations, values.abs())
    count = zeros.scatter_add(dim, destinations, torch.ones_like(values))
    roundoff = (count - 1).clamp_min(0) * (torch.finfo(src.dtype).eps / 2)
    bound = absolute_sum * (roundoff / (1 - roundoff))
    for observed in (actual, reference):
        assert ((observed.cpu().double() - exact).abs() <= bound).all()


def _set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _import_torch_scatter():
    try:
        import torch_scatter  # noqa: F401
        return torch_scatter
    except Exception as e:
        return e


# --------------------------- tests ---------------------------

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_basic_equivalence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    _set_seed(123)

    # Typical PyG use case: src [E, F], index [E], dim=0, dim_size=N
    E, F, N = 128, 16, 57
    src = torch.randn(E, F, device=device, dtype=torch.float32)
    index = torch.randint(0, N, (E,), device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src, index, dim=0, dim_size=N)
    out_new = scatter_sum(src, index, dim=0, dim_size=N)

    _assert_sum(out_new, out_ref, src, index, 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_rank_broadcast_equivalence(device):
    """
    Per-batch receiver lists share the destination axis across features.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    _set_seed(7)

    # src [B, E, F], index [B, E] for dim=1
    B, E, F, N = 4, 50, 8, 31
    src = torch.randn(B, E, F, device=device)
    index = torch.randint(0, N, (B, E), device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src, index, dim=1, dim_size=N)
    out_new = torch.stack([scatter_sum(row, ids, N) for row, ids in zip(src, index)])

    _assert_sum(out_new, out_ref, src, index, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_negative_dim_equivalence(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    _set_seed(99)

    # scatter along last dim (dim=-1)
    B, F, N = 3, 11, 17
    src = torch.randn(B, F, device=device)
    index = torch.randint(0, N, (B, F), device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src, index, dim=-1, dim_size=N)
    out_new = torch.stack([scatter_sum(row, ids, N, dim=-1) for row, ids in zip(src, index)])

    _assert_sum(out_new, out_ref, src, index, -1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_infer_dim_size_equivalence(device):
    """
    When dim_size is None, both should infer as max(index)+1.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    _set_seed(2026)

    E, F = 200, 5
    # Make a non-trivial max(index)
    N_true = 73
    src = torch.randn(E, F, device=device)
    index = torch.randint(0, N_true, (E,), device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src, index, dim=0)  # infer dim_size
    out_new = scatter_sum(src, index, dim=0, dim_size=None)

    assert out_ref.shape == out_new.shape
    _assert_sum(out_new, out_ref, src, index, 0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_empty_input_equivalence(device):
    """
    Edge case: empty src/index (E=0). Behavior can be subtle across libs.
    We test a conservative case where dim_size is explicitly provided.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    E, F, N = 0, 4, 10
    src = torch.empty(E, F, device=device)
    index = torch.empty(E, device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src, index, dim=0, dim_size=N)
    out_new = scatter_sum(src, index, dim=0, dim_size=N)

    torch.testing.assert_close(out_new, out_ref, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_grad_equivalence(device):
    """
    Gradient check: compare grads w.r.t. src for a random loss.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch_scatter = _import_torch_scatter()
    if isinstance(torch_scatter, Exception):
        pytest.skip(f"torch_scatter not importable: {torch_scatter}")

    _set_seed(31415)

    E, F, N = 256, 32, 101
    src1 = torch.randn(E, F, device=device, requires_grad=True)
    src2 = src1.detach().clone().requires_grad_(True)  # same values, separate graph
    index = torch.randint(0, N, (E,), device=device, dtype=torch.long)

    out_ref = torch_scatter.scatter_sum(src1, index, dim=0, dim_size=N)
    out_new = scatter_sum(src2, index, dim=0, dim_size=N)

    # Random-ish loss that touches all outputs
    w = torch.randn_like(out_ref)
    loss_ref = (out_ref * w).sum()
    loss_new = (out_new * w).sum()

    loss_ref.backward()
    loss_new.backward()

    torch.testing.assert_close(src2.grad, src1.grad, rtol=0, atol=0)
