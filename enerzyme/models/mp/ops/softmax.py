"""Segment softmax over the edges that share a receiver.

All variants subtract the per-segment maximum before ``exp``. They differ only in
how the denominator is guarded, whether the maximum is detached, and whether a
per-segment null logit takes part (DPA4 null mass); those choices are explicit
arguments so every Core keeps its exact numerics.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .scatter import _broadcast_index, scatter_reduce, scatter_sum


def segment_softmax(
    logits: Tensor,
    index: Tensor,
    num_segments: int,
    dim: int = 0,
    eps: float = 0.0,
    clamp_min: Optional[float] = None,
    detach_max: bool = False,
    exp_rescale: Optional[Tensor] = None,
    dropout: Optional[torch.nn.Module] = None,
    floor: Optional[float] = None,
    envelope: Optional[Tensor] = None,
    null_logit: Optional[Tensor] = None,
    *,
    ordered_sum: bool = False,
) -> Tensor:
    """Softmax of ``logits`` along ``dim`` within segments given by ``index``.

    ``eps`` is added to the denominator; ``clamp_min`` clamps it instead.
    ``exp_rescale`` multiplies ``exp(logit - max)`` before normalising
    (EquiformerV3 envelope); ``dropout`` is applied to the exponentials.
    ``floor`` initialises the running maximum (``None`` means empty segments
    produce zero maxima, matching PyG ``scatter(..., reduce="max")``).
    ``envelope`` is a nonnegative weight per edge or per logit. Zero weights
    exclude edges. ``null_logit`` adds a competing logit per segment, broadcast
    over the non-edge dimensions. The output contains only the edge weights.
    ``ordered_sum`` uses indexed accumulation for the denominator.
    """
    if logits.numel() == 0 and null_logit is None and envelope is None:
        return logits
    dim = dim + logits.dim() if dim < 0 else dim
    if envelope is not None:
        if envelope.ndim == 1:
            edge_shape = [1] * logits.ndim
            edge_shape[dim] = -1
            envelope = envelope.reshape(edge_shape)
        active = envelope > 0
        log_weight = torch.where(active, envelope, 1.0).log()
        logits = (logits + log_weight).masked_fill(~active, -torch.inf)
    source = logits.detach() if detach_max else logits
    if null_logit is not None:
        # Include the null alternative in the maximum even for empty segments.
        shape = list(logits.shape)
        shape[dim] = num_segments
        if null_logit.ndim == logits.ndim:
            null = null_logit.expand(shape)
        else:
            channels = list(logits.shape)
            del channels[dim]
            null = null_logit.expand(channels).unsqueeze(dim).expand(shape)
        initial = null.detach() if detach_max else null
        if floor is not None:
            initial = initial.clamp_min(floor)
        maxima = initial.scatter_reduce(dim, _broadcast_index(index, logits, dim),
                                        source, reduce="amax", include_self=True)
    elif floor is None:
        maxima = scatter_reduce(source, index, num_segments, dim, reduce="amax")
    else:
        maxima = scatter_reduce(source, index, num_segments, dim, reduce="amax", fill=floor, include_self=True)
    if envelope is not None and null_logit is None:
        maxima = maxima.masked_fill(maxima == -torch.inf, 0.0)
    out = (logits - maxima.index_select(dim, index)).exp()
    if exp_rescale is not None:
        out = out * exp_rescale
    if dropout is not None:
        out = dropout(out)
    if ordered_sum:
        # CUDA indexed accumulation orders duplicate receivers. DPA4 needs
        # that FP32 sum to reproduce its double-precision force gradients.
        rows = out.movedim(dim, 0) if dim else out
        denominator = rows.new_zeros((num_segments, *rows.shape[1:]))
        denominator.index_put_((index,), rows, accumulate=True)
        if dim:
            denominator = denominator.movedim(0, dim)
    else:
        denominator = scatter_sum(out, index, num_segments, dim)
    if null_logit is not None:
        denominator = denominator + (null - maxima).exp()
    elif envelope is not None:
        denominator = denominator.masked_fill(denominator == 0, 1.0)
    if clamp_min is not None:
        denominator = denominator.clamp_min(clamp_min)
    if eps:
        denominator = denominator + eps
    return out / denominator.index_select(dim, index)


class SoftCap(torch.nn.Module):
    def __init__(self, cap: float) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.tanh(inputs / self.cap) * self.cap

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cap={self.cap})"


class GraphSoftmax(torch.nn.Module):
    """Receiver softmax with optional softcap, envelope rescale and exp dropout (EquiformerV3)."""

    def __init__(self, eps: float = 1e-16, exp_dropout: float = 0.0, softcap: Optional[float] = None) -> None:
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = torch.nn.Dropout(exp_dropout) if exp_dropout > 0.0 else torch.nn.Identity()
        self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()

    def forward(self, src: Tensor, index: Tensor, num_nodes: Optional[int] = None, dim: int = 0,
                exp_rescale: Optional[Tensor] = None) -> Tensor:
        src = self.softcap(src)
        if num_nodes is None:
            num_nodes = int(index.max().item()) + 1 if index.numel() else 0
        return segment_softmax(
            src, index, num_nodes, dim=dim, eps=self.eps, detach_max=True,
            exp_rescale=exp_rescale, dropout=self.dropout,
        )

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


def segment_envelope_gated_softmax(
    logits: Tensor,
    edge_env: Tensor,
    dst: Tensor,
    n_nodes: int,
    z_bias_raw: Tensor,
    eps: float = 1e-6,
    src_weight: Optional[Tensor] = None,
    edge_mask: Optional[Tensor] = None,
    *,
    ordered_sum: bool = False,
) -> Tensor:
    """Destination softmax with envelope-weighted logits and a learned null mass (DPA4 EMFA A3).

    ``logits`` is ``(E, F, H)``, ``edge_env`` ``(E,)`` or ``(E, 1)``, ``z_bias_raw``
    ``(F, H)`` (softplus gives the null mass). Computed in float32.
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
        source_ratio = torch.where(source_positive, source_weight / source_scale, torch.zeros_like(source_weight))
        active = active & source_positive
    if edge_mask is not None:
        active = active & (edge_mask.reshape(n_edge).to(dtype=compute_dtype) > 0.0)

    effective_logits = logits_2d + log_weight[:, None]
    effective_logits = torch.where(active[:, None], effective_logits, torch.full_like(effective_logits, float("-inf")))
    null_mass = F.softplus(z_bias_raw.to(dtype=compute_dtype)) + float(eps)
    null_logit = torch.log(null_mass).reshape(1, n_channel)

    alpha = segment_softmax(
        effective_logits, dst, n_nodes, null_logit=null_logit,
        exp_rescale=source_ratio[:, None] if source_ratio is not None else None,
        ordered_sum=ordered_sum,
    )
    return alpha.reshape(n_edge, n_focus, n_head).to(dtype=logits.dtype)
