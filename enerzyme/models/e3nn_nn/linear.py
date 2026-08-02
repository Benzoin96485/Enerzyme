"""Flat-Irreps linear maps shared by e3nn-stack architectures (TACE, …)."""

from __future__ import annotations

from typing import List, Literal, Sequence, Union

import torch
from e3nn import o3
from torch import Tensor, nn


def _add_0e_bias(
    out: Tensor,
    bias: Tensor,
    out_slices: Sequence[slice],
    bias_slices: Sequence[slice],
) -> Tensor:
    """Add scalar (0e) bias without per-slice ``clone`` of ``out``.

    ``bias`` is ``[bias_dim]`` (broadcast over batch) or ``[N, bias_dim]``.
    """
    if not out_slices:
        return out
    bias_full = out.new_zeros(out.shape)
    if bias.ndim == 1:
        for sl, bsl in zip(out_slices, bias_slices):
            bias_full[:, sl] = bias[bsl]
    else:
        for sl, bsl in zip(out_slices, bias_slices):
            bias_full[:, sl] = bias[:, bsl]
    return out + bias_full


def _collect_0e_slices(irreps_out: o3.Irreps) -> tuple[List[slice], List[slice], int]:
    out_slices: List[slice] = []
    bias_slices: List[slice] = []
    bias_dim = 0
    acc = 0
    for mul, ir in irreps_out:
        dim = mul * ir.dim
        if ir.l == 0 and ir.p == 1:
            out_slices.append(slice(acc, acc + dim))
            bias_slices.append(slice(bias_dim, bias_dim + dim))
            bias_dim += dim
        acc += dim
    return out_slices, bias_slices, bias_dim


class IrrepsLinear(nn.Module):
    """e3nn Linear with optional scalar bias (parity-even 0e only)."""

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        *,
        bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight_numel = self.linear.weight_numel
        self.weight = nn.Parameter(torch.empty(self.weight_numel))
        torch.nn.init.normal_(self.weight)

        self._0e_slices, self._bias_slices, bias_dim = _collect_0e_slices(self.irreps_out)
        if bias and bias_dim > 0:
            self.bias = nn.Parameter(torch.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x, self.weight)
        if self.bias is not None:
            out = _add_0e_bias(out, self.bias, self._0e_slices, self._bias_slices)
        return out


class ElementIrrepsLinear(nn.Module):
    """Element-conditioned e3nn Linear (one weight set per element)."""

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        irreps_out: Union[str, o3.Irreps],
        *,
        num_elements: int,
        bias: bool = True,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_elements = num_elements
        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.weight = nn.Parameter(torch.empty(num_elements, self.linear.weight_numel))
        torch.nn.init.normal_(self.weight)

        self._0e_slices, self._bias_slices, bias_dim = _collect_0e_slices(self.irreps_out)
        if bias and bias_dim > 0:
            self.bias = nn.Parameter(torch.zeros(num_elements, bias_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, attrs: Tensor) -> Tensor:
        # attrs: [N, num_elements] one-hot
        weight = torch.einsum("ne,ew->nw", attrs, self.weight)
        out = self.linear(x, weight)
        if self.bias is not None:
            b = torch.einsum("ne,eb->nb", attrs, self.bias)
            out = _add_0e_bias(out, b, self._0e_slices, self._bias_slices)
        return out


class SkipIdentity(nn.Module):
    """Pad / truncate irreps features without a learned map."""

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        if self.irreps_in == self.irreps_out:
            self._mode = "id"
            return
        in_slices = list(self.irreps_in.slices())
        out_slices = list(self.irreps_out.slices())
        pairs = []
        used = set()
        for i, mul_ir in enumerate(self.irreps_in):
            for j, (target, tsl) in enumerate(zip(self.irreps_out, out_slices)):
                if j not in used and mul_ir == target:
                    pairs.append((in_slices[i], tsl))
                    used.add(j)
                    break
            else:
                raise ValueError(f"{self.irreps_in} not embeddable into {self.irreps_out}")
        self._pairs = pairs
        self._mode = "map"

    def forward(self, x: Tensor) -> Tensor:
        if self._mode == "id":
            return x
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        for isl, osl in self._pairs:
            out[:, osl] = x[:, isl]
        return out


def get_resnet_layer(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    bias: bool,
    num_elements: int,
    resnet_type: Literal["aware", "agnostic", "identity"] = "aware",
):
    """Element-aware / agnostic / identity residual map between irreps."""
    if resnet_type == "agnostic":
        return IrrepsLinear(irreps_in, irreps_out, bias=bias)
    if resnet_type == "identity":
        return SkipIdentity(irreps_in, irreps_out)
    return ElementIrrepsLinear(
        irreps_in, irreps_out, bias=bias, num_elements=num_elements
    )
