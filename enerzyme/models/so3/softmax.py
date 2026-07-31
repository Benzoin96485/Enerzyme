"""Graph softmax with optional envelope rescale (EquiformerV3)."""

from __future__ import annotations

import torch
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
