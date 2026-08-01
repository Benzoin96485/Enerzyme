"""Thin e3nn Linear / ElementLinear wrappers for TACE (no LoRA).

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import math
from typing import List, Optional, Union

import torch
from e3nn import o3
from torch import Tensor, nn


class MLPLinear(nn.Module):
    """Scalar linear used inside radial MLPs."""

    def __init__(self, in_dim: int, out_dim: int, alpha: float = 1.0, bias: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight * self.alpha
        if self.bias is None:
            return torch.mm(x, w)
        return torch.addmm(self.bias, x, w)


class ScalarMLP(nn.Module):
    def __init__(
        self,
        channels: List[int],
        bias: bool = False,
        act: Optional[str] = "silu",
        layer_norm: bool = False,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("ScalarMLP needs at least 2 channel sizes")
        layers: List[nn.Module] = []
        for i, (h_in, h_out) in enumerate(zip(channels[:-1], channels[1:])):
            gain = 1.0 if act is None or i == 0 else math.sqrt(2.0)
            layers.append(MLPLinear(h_in, h_out, alpha=gain / math.sqrt(h_in), bias=bias))
            if i < len(channels) - 2:
                if layer_norm:
                    layers.append(nn.LayerNorm(h_out))
                if act == "silu":
                    layers.append(nn.SiLU())
                elif act == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif act is not None:
                    raise ValueError(f"Unsupported act={act}")
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


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

        bias_dim = 0
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        for mul, ir in self.irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_dim, bias_dim + dim))
                bias_dim += dim
            acc += dim
        if bias and bias_dim > 0:
            self.bias = nn.Parameter(torch.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x, self.weight)
        if self.bias is not None:
            out = out.clone()
            for sl, bsl in zip(self._0e_slices, self._bias_slices):
                out[:, sl] = out[:, sl] + self.bias[bsl].unsqueeze(0)
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

        bias_dim = 0
        self._0e_slices = []
        self._bias_slices = []
        acc = 0
        for mul, ir in self.irreps_out:
            dim = mul * ir.dim
            if ir.l == 0 and ir.p == 1:
                self._0e_slices.append(slice(acc, acc + dim))
                self._bias_slices.append(slice(bias_dim, bias_dim + dim))
                bias_dim += dim
            acc += dim
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
            for sl, bsl in zip(self._0e_slices, self._bias_slices):
                out = out.clone()
                out[:, sl] = out[:, sl] + b[:, bsl]
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
        # Gather matching irrep slices
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
