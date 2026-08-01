"""Pure-e3nn tensor products used by spherical TACE (no OEQ/CUE/EQT).

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from .paths import generate_paths


class UUUTensorProduct(nn.Module):
    """Channel-coupled (uuu) tensor product for CgtpACE body-order recursion."""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Optional[str] = None,
        trainable: bool = False,
        warning: bool = False,
        identical_inputs: bool = False,
    ) -> None:
        super().__init__()
        instructions, actual_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            e3nn_mode="uuu",
            trainable=trainable,
            identical_inputs=identical_inputs,
        )
        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.irreps_out = actual_out
        self.weight_numel = self.tp.weight_numel
        if warning:
            warnings.warn(
                "TACE CgtpACE correlation>=3 uses pure e3nn uuu products "
                "(no Equitorch). Prefer correlation=2 for faster iteration.",
                stacklevel=2,
            )

    def forward(
        self, x: Tensor, y: Tensor, ws: Optional[Tensor] = None
    ) -> Tensor:
        return self.tp(x, y, ws)


class O3ScatterTensorProduct(nn.Module):
    """Weighted TP(node, Y_lm) followed by scatter to receivers."""

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        l1l2: Optional[str] = None,
    ) -> None:
        super().__init__()
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)
        instructions, actual_out = generate_paths(
            irreps_out=irreps_out,
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            l1l2=l1l2,
            e3nn_mode="uvu",
        )
        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            actual_out,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.irreps_out = actual_out
        self.weight_numel = self.tp.weight_numel

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        w: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        msg = self.tp(x[edge_index[0]], y, w)
        return scatter_sum(msg, edge_index[1], dim=0, dim_size=x.size(0))
