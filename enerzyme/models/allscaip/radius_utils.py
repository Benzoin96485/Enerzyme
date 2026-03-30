"""
Neighbor ranking and geometry helpers for biknn + radius graphs.
Derived from fairchem escaip (MIT): MinDScAIP / Ryan Liu.
"""
from __future__ import annotations

import torch


def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    vec_norm_sq = x.square().sum(dim=dim, keepdim=keepdim)
    vec_norm = vec_norm_sq.clamp_min(eps).sqrt()
    return torch.where(vec_norm_sq <= eps, torch.zeros_like(vec_norm), vec_norm)


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    vec_norm_sq = x.square().sum(dim=dim, keepdim=True)
    vec_norm = vec_norm_sq.clamp_min(eps).sqrt()
    return torch.where(vec_norm_sq <= eps, torch.zeros_like(x), x / vec_norm)


def envelope_fn(x: torch.Tensor, envelope: bool = True) -> torch.Tensor:
    if envelope:
        env = -x.pow(2) / (1 - x.pow(2))
    else:
        env = torch.zeros_like(x)
    return torch.where(x < 1, env, torch.full_like(x, float("-inf")))


def shifted_sine(x: torch.Tensor) -> torch.Tensor:
    return (
        0.5 * torch.where(x.abs() < torch.pi, torch.sin(0.5 * x), torch.sign(x)) + 0.5
    )


def soft_rank(dist: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sigmoid((dist[:, :, None] - dist[:, None, :]) / scale).sum(dim=-1)


def hard_rank(dist: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(dist)
    ranks[
        torch.arange(dist.size(0), device=dist.device)[:, None],
        torch.argsort(dist, dim=-1),
    ] = torch.arange(dist.size(-1), device=dist.device, dtype=dist.dtype)
    return ranks


def soft_rank_low_mem(
    dist: torch.Tensor,
    k: int,
    scale: float,
    delta: int = 20,
) -> torch.Tensor:
    sorted_dist, indicies = torch.sort(dist, dim=-1)
    kd = min(k + delta, sorted_dist.shape[-1])
    ranks_T = shifted_sine(
        (sorted_dist[:, :kd, None] - sorted_dist[:, None, :kd]) / scale
    ).sum(dim=-1)
    ranks = torch.full_like(dist, torch.inf)
    ranks[
        torch.arange(dist.size(0), device=dist.device)[:, None],
        indicies[:, :kd],
    ] = ranks_T
    return ranks
