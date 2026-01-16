# bench_scatter_sum_speed.py
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch


def scatter_sum_torch(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Torch-native replacement for torch_scatter.scatter_sum using scatter_add."""
    if index.dtype != torch.long:
        index = index.long()
    dim = dim % src.dim()

    # Make index broadcastable to src, then expand to src shape.
    if index.dim() != src.dim():
        for _ in range(src.dim() - index.dim()):
            index = index.unsqueeze(-1)
    index = index.expand_as(src)

    if dim_size is None:
        dim_size = 0 if index.numel() == 0 else int(index.max().item()) + 1

    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = src.new_zeros(out_shape)
    return out.scatter_add(dim, index, src)


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_one(fn, iters: int, device: str) -> float:
    """Return average time per call in milliseconds."""
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


@dataclass
class Case:
    E: int          # number of edges/items
    F: int          # feature dimension
    N: int          # number of groups/nodes (dim_size)
    dim: int = 0    # scatter along dim 0 (typical PyG)


def run_bench(
    device: str,
    dtype: torch.dtype,
    cases: List[Case],
    warmup: int = 50,
    iters: int = 200,
    seed: int = 123,
):
    # Try importing torch_scatter
    try:
        import torch_scatter
        has_scatter = True
    except Exception as e:
        has_scatter = False
        scatter_import_error = e

    print(f"\n=== Device: {device} | dtype: {dtype} | warmup={warmup} iters={iters} ===")
    if not has_scatter:
        print(f"torch_scatter not importable, will benchmark only torch-native. Error: {scatter_import_error}")

    # Header
    print(f"{'E':>8} {'F':>6} {'N':>8} | {'torch_scatter(ms)':>16} {'torch_native(ms)':>16} {'speedup(sc/native)':>18}")
    print("-" * 78)

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    for c in cases:
        # Prepare inputs
        src = torch.randn(c.E, c.F, device=device, dtype=dtype)
        index = torch.randint(0, c.N, (c.E,), device=device, dtype=torch.long)

        # Define callables (capture tensors)
        def fn_native():
            return scatter_sum_torch(src, index, dim=c.dim, dim_size=c.N)

        # Warmup native
        for _ in range(warmup):
            fn_native()
        _sync(device)

        t_native = _time_one(fn_native, iters=iters, device=device)

        t_scatter = float("nan")
        if has_scatter:
            def fn_scatter():
                return torch_scatter.scatter_sum(src, index, dim=c.dim, dim_size=c.N)

            # Warmup scatter
            for _ in range(warmup):
                fn_scatter()
            _sync(device)

            t_scatter = _time_one(fn_scatter, iters=iters, device=device)

        speedup = (t_scatter / t_native) if (has_scatter and t_native > 0) else float("nan")

        def fmt(x):
            return f"{x:16.4f}" if x == x else f"{'N/A':>16}"

        print(f"{c.E:8d} {c.F:6d} {c.N:8d} | {fmt(t_scatter)} {t_native:16.4f} {speedup:18.3f}")


def main():
    # Typical GNN-ish shapes:
    cases = [
        Case(E=50_000,  F=16,  N=10_000),
        Case(E=200_000, F=16,  N=50_000),
        Case(E=200_000, F=64,  N=50_000),
        Case(E=1_000_000, F=16, N=200_000),
        Case(E=1_000_000, F=64, N=200_000),
    ]

    # CPU benches (float32)
    run_bench(device="cpu", dtype=torch.float32, cases=cases, warmup=20, iters=50)

    # GPU benches (float32 / bf16 optional)
    if torch.cuda.is_available():
        run_bench(device="cuda", dtype=torch.float32, cases=cases, warmup=50, iters=200)

        # Optional: bf16 (if you care)
        if torch.cuda.is_bf16_supported():
            run_bench(device="cuda", dtype=torch.bfloat16, cases=cases, warmup=50, iters=200)
    else:
        print("\nCUDA not available; skipping GPU benchmarks.")


if __name__ == "__main__":
    main()
