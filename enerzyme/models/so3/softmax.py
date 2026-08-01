"""Graph softmax with optional envelope rescale (EquiformerV3) and
envelope-gated destination softmax (DPA4 EMFA A3).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


class SoftCap(torch.nn.Module):
    def __init__(self, cap: float) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs / self.cap
        outputs = torch.nn.functional.tanh(outputs)
        return outputs * self.cap

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cap={self.cap})"


class GraphSoftmax(torch.nn.Module):
    """PyG-style softmax with envelope rescale / softcap / exp dropout.

    Reference: EquiformerV3 ``experimental/models/equiformer_v3/softmax.py``.
    """

    def __init__(
        self,
        eps: float = 1e-16,
        exp_dropout: float = 0.0,
        softcap: float | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.exp_dropout = exp_dropout
        self.dropout = (
            torch.nn.Dropout(exp_dropout)
            if self.exp_dropout > 0.0
            else torch.nn.Identity()
        )
        self.softcap = SoftCap(cap=softcap) if softcap is not None else torch.nn.Identity()

    def forward(
        self,
        src,
        index=None,
        ptr=None,
        num_nodes=None,
        dim: int = 0,
        exp_rescale=None,
    ):
        src = self.softcap(src)
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            count = ptr[1:] - ptr[:-1]
            ptr = ptr.view(size)
            src_max = segment(src.detach(), ptr, reduce="max")
            src_max = src_max.repeat_interleave(count, dim=dim)
            out = (src - src_max).exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out = self.dropout(out)
            out_sum = segment(out, ptr, reduce="sum") + self.eps
            out_sum = out_sum.repeat_interleave(count, dim=dim)
        elif index is not None:
            N = maybe_num_nodes(index, num_nodes)
            src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
            out = src - src_max.index_select(dim, index)
            out = out.exp()
            if exp_rescale is not None:
                out = out * exp_rescale
            out = self.dropout(out)
            out_sum = scatter(out, index, dim, dim_size=N, reduce="sum") + self.eps
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError
        return out / out_sum

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
) -> Tensor:
    """Destination-wise envelope-gated softmax (DPA4 EMFA A3).

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
