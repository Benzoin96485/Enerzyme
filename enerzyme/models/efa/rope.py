"""Euclidean rotary positional encodings (ERoPE) and linear EFA aggregation.

Ported from thorben-frank/euclidean_fast_attention ``rope.py`` (MIT) for the
invariant (L=0) path used by Enerzyme: features shaped ``[N, F]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def frequency_init(
    num_features_qk: int,
    max_frequency: float,
    max_length: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """ERoPE frequencies ``ω = linspace(0, b_max, F/2) / r_max``."""
    if num_features_qk % 2 != 0:
        raise ValueError(
            f"num_features_qk must be even for ERoPE, got {num_features_qk}"
        )
    n_freq = num_features_qk // 2
    if n_freq > 1:
        freqs = torch.linspace(
            0.0, float(max_frequency), n_freq, dtype=dtype, device=device
        )
    else:
        freqs = torch.tensor(
            [float(max_frequency)], dtype=dtype, device=device
        )
    return freqs / float(max_length)


def calculate_rotary_position_embedding(
    x_proj: Tensor, theta: Tensor
) -> Tuple[Tensor, Tensor]:
    """Sin/cos for projections ``x_proj`` ``[..., M]`` and frequencies ``theta``.

    Returns ``sin, cos`` each shaped ``[..., M, 2 * len(theta)]``.
    """
    # (..., M, 1) * (K,) -> (..., M, K)
    angle = x_proj.unsqueeze(-1) * theta
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    return sin, cos


def apply_rotary_position_embedding(
    x: Tensor, sin: Tensor, cos: Tensor
) -> Tensor:
    """Apply ERoPE to scalar features.

    Parameters
    ----------
    x:
        ``[N, F]`` features (F even).
    sin, cos:
        ``[N, M, F]`` rotary factors from :func:`calculate_rotary_position_embedding`.

    Returns
    -------
    Tensor
        ``[N, M, F]`` RoPE-encoded features on the Lebedev grid.
    """
    if not (
        x.shape[-1] == sin.shape[-1] == cos.shape[-1] and x.shape[-1] % 2 == 0
    ):
        raise ValueError(
            "x, sin, and cos must share an even last dimension; "
            f"got {x.shape}, {sin.shape}, {cos.shape}"
        )
    x_g = x.unsqueeze(1)  # (N, 1, F) -> broadcast over M
    # Pair rotation: (-x_odd, x_even)
    y = torch.stack((-x_g[..., 1::2], x_g[..., ::2]), dim=-1).reshape(x_g.shape)
    return x_g * cos + y * sin


def linear_efa_aggregate(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    positions: Tensor,
    theta: Tensor,
    grid_u: Tensor,
    grid_w: Tensor,
    batch_seg: Tensor,
    *,
    num_graphs: Optional[int] = None,
) -> Tensor:
    """Unnormalized linear attention with ERoPE (scalar features).

    Parameters
    ----------
    q, k:
        ``[N, F_qk]`` (even ``F_qk``).
    v:
        ``[N, F_v]``.
    positions:
        ``[N, 3]`` absolute coordinates.
    theta:
        ``[F_qk // 2]`` frequencies.
    grid_u, grid_w:
        Lebedev points ``[M, 3]`` and weights ``[M]``.
    batch_seg:
        ``[N]`` molecule ids.
    num_graphs:
        Number of graphs; defaults to ``batch_seg.max() + 1``.

    Returns
    -------
    Tensor
        ``[N, F_v]`` EFA update (Lebedev-integrated).
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be [N, 3], got {tuple(positions.shape)}")
    batch_seg = batch_seg.long()
    if num_graphs is None:
        num_graphs = int(batch_seg.max().item()) + 1 if batch_seg.numel() else 1

    # Projections onto Lebedev directions: (N, M)
    x_proj = torch.einsum("nd,md->nm", positions, grid_u)
    sin, cos = calculate_rotary_position_embedding(x_proj, theta)
    q_r = apply_rotary_position_embedding(q, sin, cos)  # (N, M, Fqk)
    k_r = apply_rotary_position_embedding(k, sin, cos)  # (N, M, Fqk)
    q_r = q_r / (q_r.shape[-1] ** 0.5)

    n_atoms = positions.shape[0]
    m_grid = grid_u.shape[0]
    f_v = v.shape[-1]

    if num_graphs > 1:
        # (N, M, Fqk, Fv)
        kv = torch.einsum("nmk,nv->nmkv", k_r, v)
        # Segment sum over atoms per graph, then broadcast back to atoms.
        kv_flat = kv.reshape(n_atoms, -1)
        kv_sum = torch.zeros(
            num_graphs,
            kv_flat.shape[-1],
            dtype=kv_flat.dtype,
            device=kv_flat.device,
        )
        kv_sum.index_add_(0, batch_seg, kv_flat)
        kv = kv_sum[batch_seg].reshape(n_atoms, m_grid, k_r.shape[-1], f_v)
        y = torch.einsum("nmd,nmdv,m->nv", q_r, kv, grid_w)
    else:
        # Single-graph fast path: (M, Fqk, Fv)
        kv = torch.einsum("nmk,nv->mkv", k_r, v)
        y = torch.einsum("nmd,mdv,m->nv", q_r, kv, grid_w)
    return y
