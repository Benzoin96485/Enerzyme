# Adapted from IQuestLab/UBio-MolFM (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""Optional Triton sparse QK / V kernels with PyTorch fallbacks.

N×K tiled kernels match Enerzyme ``f_sparse_idx_*`` padded neighborhoods.
Triton runs only when CUDA is available; otherwise PyTorch gather paths are used.
"""

from __future__ import annotations

from functools import partial
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
    if use_triton is None:
        use_triton = triton_kernels_available()
    if use_triton:
        from .sparse_qk import sparse_qk_triton_n_tiled

        return sparse_qk_triton_n_tiled(query, key, idx, gate, scale)
    return sparse_qk_pytorch(query, key, idx, gate, scale)


def sparse_v_agg(
    value: Tensor,
    alpha: Tensor,
    idx: Tensor,
    use_triton: Optional[bool] = None,
) -> Tensor:
    if use_triton is None:
        use_triton = triton_kernels_available()
    if use_triton:
        from .sparse_v_agg import sparse_v_agg_triton_n_tiled

        return sparse_v_agg_triton_n_tiled(value, alpha, idx)
    return sparse_v_agg_pytorch(value, alpha, idx)


def make_sparse_v_kernel(
    idx: Tensor, use_triton: Optional[bool] = None
) -> Callable:
    """Bind neighbor indices for SO2 TP ``triton_kernel(value=, alpha=)`` calls."""
    if use_triton is None:
        use_triton = triton_kernels_available()

    def _kernel(value: Tensor, alpha: Tensor) -> Tensor:
        return sparse_v_agg(value, alpha, idx, use_triton=use_triton)

    return _kernel
