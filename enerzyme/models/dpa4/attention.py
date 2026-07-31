"""Envelope-gated attention for DPA4 EMFA A3.

Reimplemented after deepmd-kit ``dpa4_nn.attention`` (Li et al., arXiv:2606.02419).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def segment_envelope_gated_softmax(
    logits: Tensor,
    edge_env: Tensor,
    dst: Tensor,
    n_nodes: int,
    z_bias_raw: Tensor,
    eps: float = 1e-6,
    src_weight: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
) -> Tensor:
    """Destination-wise envelope-gated softmax.

    Parameters
    ----------
    logits
        ``(E, F, H)``
    edge_env
        ``(E, 1)`` or ``(E,)``
    dst
        ``(E,)`` destination indices
    z_bias_raw
        ``(F, H)`` unconstrained null-mass bias (softplus applied)
    """
    n_edge, n_focus, n_head = logits.shape
    n_channel = n_focus * n_head
    compute_dtype = torch.float32
    logits_2d = logits.reshape(n_edge, n_channel).to(dtype=compute_dtype)
    edge_env_1d = edge_env.reshape(n_edge).to(dtype=compute_dtype)
    edge_positive = edge_env_1d > 0.0
    ones = torch.ones_like(edge_env_1d)
    log_weight = 2.0 * torch.log(torch.where(edge_positive, edge_env_1d, ones))
    active = edge_positive
    source_ratio = None
    if src_weight is not None:
        source_weight = src_weight.reshape(n_edge).to(dtype=compute_dtype)
        source_positive = source_weight > 0.0
        safe_source = torch.where(source_positive, source_weight, ones)
        source_scale = safe_source.detach()
        log_weight = log_weight + torch.log(source_scale)
        source_ratio = torch.where(
            source_positive,
            source_weight / source_scale,
            torch.zeros_like(source_weight),
        )
        active = active & source_positive
    if edge_mask is not None:
        mask = edge_mask.reshape(n_edge).to(dtype=compute_dtype)
        active = active & (mask > 0.0)

    effective_logits = logits_2d + log_weight[:, None]
    effective_logits = torch.where(
        active[:, None],
        effective_logits,
        torch.full_like(effective_logits, float("-inf")),
    )
    null_mass = F.softplus(z_bias_raw.to(dtype=compute_dtype)) + float(eps)
    null_logit = torch.log(null_mass).reshape(1, n_channel)

    group_max = null_logit.expand(n_nodes, n_channel).clone()
    group_max = group_max.scatter_reduce(
        0, dst[:, None].expand(-1, n_channel), effective_logits, reduce="amax", include_self=True
    )
    edge_max = group_max.index_select(0, dst)
    edge_exp = torch.exp(effective_logits - edge_max)
    if source_ratio is not None:
        edge_exp = edge_exp * source_ratio[:, None]
    denom_sum = torch.zeros(n_nodes, n_channel, dtype=compute_dtype, device=logits.device)
    denom_sum.index_add_(0, dst, edge_exp)
    denominator = denom_sum + torch.exp(null_logit - group_max)
    alpha = edge_exp / denominator.index_select(0, dst)
    return alpha.reshape(n_edge, n_focus, n_head).to(dtype=logits.dtype)
