# Adapted from IQuestLab/UBio-MolFM (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""Optional Triton sparse QK / V kernels with PyTorch fallbacks.

N×K tiled kernels match Enerzyme ``f_sparse_idx_*`` padded neighborhoods.
Triton runs only when CUDA is available; otherwise PyTorch gather paths are used.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def triton_kernels_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def _use_kernel(requested, tensor):
    # Python autograd.Function kernels do not participate in torch.func
    # transforms. The native path also preserves float64 throughout.
    return (
        requested is not False and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not torch.compiler.is_compiling()
        and not torch._C._are_functorch_transforms_active()
        and triton_kernels_available()
    )


def sparse_qk_pytorch(
    query: Tensor,
    key: Tensor,
    idx: Tensor,
    gate: Tensor,
    scale: float,
) -> Tensor:
    """Reference sparse QK: ``gate * (q · k[j]) * scale`` over padded neighbors."""
    # query: [N, H, D], key: [M, H, D], idx: [N, K], gate: [N, K, H]
    gathered = key[idx]  # [N, K, H, D]
    dots = (query.unsqueeze(1) * gathered).sum(dim=-1)  # [N, K, H]
    return gate * dots * scale


def sparse_v_agg_pytorch(value: Tensor, alpha: Tensor, idx: Tensor) -> Tensor:
    """Reference sparse V aggregate: ``sum_k alpha[i,k] * value[idx[i,k]]``."""
    # value: [M, C, H] after internal convention of n_tiled (see sparse_v_agg)
    # Public API matches UBio ``sparse_v_agg_triton_n_tiled``:
    # value [M, Feat, H], alpha [N, K, H], idx [N, K] → [N, Feat, H]
    gathered = value[idx]  # [N, K, Feat, H]
    return (alpha.unsqueeze(2) * gathered).sum(dim=1)


def sparse_qk(
    query: Tensor,
    key: Tensor,
    idx: Tensor,
    gate: Tensor,
    scale: float = 1.0,
    use_triton: Optional[bool] = None,
) -> Tensor:
    if _use_kernel(use_triton, query) and key.dtype == query.dtype == gate.dtype:
        from ._e2former_qk import sparse_qk_triton_n_tiled

        return sparse_qk_triton_n_tiled(query, key, idx, gate, scale)
    return sparse_qk_pytorch(query, key, idx, gate, scale)


def sparse_v_agg(
    value: Tensor,
    alpha: Tensor,
    idx: Tensor,
    use_triton: Optional[bool] = None,
) -> Tensor:
    if _use_kernel(use_triton, value) and alpha.dtype == value.dtype:
        from ._e2former_value import sparse_v_agg_triton_n_tiled

        return sparse_v_agg_triton_n_tiled(value, alpha, idx)
    return sparse_v_agg_pytorch(value, alpha, idx)


def make_sparse_v_kernel(
    idx: Tensor, use_triton: Optional[bool] = None
) -> Callable:
    """Bind neighbor indices for SO2 TP ``triton_kernel(value=, alpha=)`` calls."""
    def _kernel(value: Tensor, alpha: Tensor) -> Tensor:
        return sparse_v_agg(value, alpha, idx, use_triton=use_triton)

    return _kernel
