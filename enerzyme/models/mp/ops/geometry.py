"""Norms and unit vectors with explicit guards.

``mode`` chooses where the guard enters, because Cores were fitted with
different conventions and the guard changes values near zero:

* ``"sqrt_eps2"``: ``sqrt(|x|^2 + eps^2)`` (smooth, never zero; quaternion code)
* ``"clamp"``: ``sqrt(max(|x|^2, eps))``
* ``"clamp_zero"``: like ``"clamp"`` but exactly zero when ``|x|^2 <= eps``
* ``"norm_clamp"``: ``max(|x|, eps)``
"""

from __future__ import annotations

import torch
from torch import Tensor


def smooth_norm(x: Tensor, eps: float = 1e-7, *, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Euclidean norm with the squared guard used by quaternion edge frames."""
    return torch.sqrt((x * x).sum(dim=dim, keepdim=keepdim) + eps * eps)


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12, mode: str = "clamp_zero") -> Tensor:
    if mode == "norm_clamp":
        return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim).clamp_min(eps)
    if mode == "sqrt_eps2":
        return smooth_norm(x, eps, dim=dim, keepdim=keepdim)
    sq = x.square().sum(dim=dim, keepdim=keepdim)
    norm = sq.clamp_min(eps).sqrt()
    if mode == "clamp":
        return norm
    if mode == "clamp_zero":
        return torch.where(sq <= eps, torch.zeros_like(norm), norm)
    raise ValueError(f"unknown safe_norm mode {mode!r}")


def safe_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12, mode: str = "clamp_zero") -> Tensor:
    if mode == "clamp_zero":
        sq = x.square().sum(dim=dim, keepdim=True)
        norm = sq.clamp_min(eps).sqrt()
        return torch.where(sq <= eps, torch.zeros_like(x), x / norm)
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps, mode=mode)
