# test_scatter_sum_equivalence.py
import pytest
import torch


def scatter_sum_torch(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: int | None = None,
) -> torch.Tensor:
    """
    Torch-native replacement for torch_scatter.scatter_sum(src, index, dim, dim_size).

    Notes:
    - index is interpreted as destination indices along `dim`
    - index must be broadcastable to src; will be expanded to src shape
    - output shape equals src shape except along `dim` it becomes dim_size (or inferred)
    """
    if index.dtype != torch.long:
        index = index.long()

    dim = dim % src.dim()

    # Make index broadcastable to src, then expand to src shape.
    # Common GNN case: src [E, F], index [E] and dim=0.
    if index.dim() != src.dim():
        # Insert trailing singleton dims until ranks match.
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
    index = index.expand_as(src)

    if dim_size is None:
        if index.numel() == 0:
            dim_size = 0
        else:
            dim_size = int(index.max().item()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    return out.scatter_add(dim, index, src)


# --------------------------- helpers ---------------------------

def _maybe_cuda_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    out_new = scatter_sum_torch(src, index, dim=0, dim_size=N)

    torch.testing.assert_close(out_new, out_ref, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_sum_rank_broadcast_equivalence(device):
    """
    torch_scatter allows index rank < src rank.
    We verify our unsqueeze+expand matches behavior.
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
    out_new = scatter_sum_torch(src, index, dim=1, dim_size=N)

    torch.testing.assert_close(out_new, out_ref, rtol=0, atol=0)


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
    out_new = scatter_sum_torch(src, index, dim=-1, dim_size=N)

    torch.testing.assert_close(out_new, out_ref, rtol=0, atol=0)


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
    out_new = scatter_sum_torch(src, index, dim=0, dim_size=None)

    assert out_ref.shape == out_new.shape
    torch.testing.assert_close(out_new, out_ref, rtol=0, atol=0)


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
    out_new = scatter_sum_torch(src, index, dim=0, dim_size=N)

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
    out_new = scatter_sum_torch(src2, index, dim=0, dim_size=N)

    # Random-ish loss that touches all outputs
    w = torch.randn_like(out_ref)
    loss_ref = (out_ref * w).sum()
    loss_new = (out_new * w).sum()

    loss_ref.backward()
    loss_new.backward()

    torch.testing.assert_close(src2.grad, src1.grad, rtol=0, atol=0)