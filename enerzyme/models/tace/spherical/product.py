"""CgtpACE — recursive CGTP product basis for spherical TACE.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import torch
from e3nn import o3
from torch import Tensor, nn

from ..linear import ElementIrrepsLinear, IrrepsLinear
from ..paths import to_possible_tp_irreps
from ..tensor_product import UUUTensorProduct


class CgtpACE(nn.Module):
    def __init__(
        self,
        *,
        layer: int,
        num_layers: int,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        irreps_in: o3.Irreps,
        correlation: int,
        target_irreps: o3.Irreps,
        l1l2: Optional[str] = None,
        parity: bool = False,
        bias: bool = True,
        agnostic: bool = False,
    ) -> None:
        super().__init__()
        if parity and correlation > 2:
            raise ValueError(
                "CgtpACE with parity=True currently requires correlation < 3."
            )
        if correlation >= 3:
            warnings.warn(
                "CgtpACE correlation>=3 is expensive without fused backends.",
                stacklevel=2,
            )

        self.layer = layer
        self.correlation = correlation
        self.num_channel = num_channel
        self.num_elements = num_elements
        self.agnostic = agnostic
        self.last_layer = layer == num_layers - 1
        self.scale = 1.0 / math.sqrt(2.0)

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = o3.Irreps(
            [(num_channel, ir) for _, ir in self.irreps_in]
        )
        target_irreps = o3.Irreps(target_irreps)

        self.irreps_tp_out_list: List[o3.Irreps] = []
        for nu in range(2, correlation + 1):
            if nu == correlation:
                if self.last_layer:
                    self.irreps_tp_out_list.append(
                        (target_irreps * num_channel).regroup()
                    )
                else:
                    base = to_possible_tp_irreps(
                        self.irreps_hidden, self.irreps_hidden, parity, Lmax
                    )
                    self.irreps_tp_out_list.append((base * num_channel).regroup())
            else:
                self.irreps_tp_out_list.append(
                    to_possible_tp_irreps(
                        self.irreps_hidden, self.irreps_hidden, parity, lmax
                    )
                )

        if correlation == 1:
            coefs_base = o3.Irreps(
                [(num_channel, ir) for _, ir in self.irreps_in if ir.l <= Lmax]
            )
        else:
            coefs_base = (
                to_possible_tp_irreps(self.irreps_in, self.irreps_in, parity, Lmax)
                * num_channel
            ).regroup()
        if self.last_layer:
            self.irreps_coefs_out = (target_irreps * num_channel).regroup()
        else:
            self.irreps_coefs_out = coefs_base

        self.irreps_out = o3.Irreps(
            [(num_channel, ir) for _, ir in self.irreps_coefs_out]
        )

        coefs_cls = IrrepsLinear if agnostic else ElementIrrepsLinear
        coefs_kwargs = {"bias": bias}
        if not agnostic:
            coefs_kwargs["num_elements"] = num_elements

        self.coefs = nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps([(num_channel, ir) for _, ir in self.irreps_hidden]).simplify(),
                self.irreps_coefs_out,
                **coefs_kwargs,
            )
        )

        self.aces = nn.ModuleList()
        product_in1 = self.irreps_hidden
        for nu in range(2, correlation + 1):
            ace = UUUTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_hidden,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=l1l2,
                identical_inputs=(nu == 2),
                warning=(nu == 2 and correlation >= 3),
            )
            self.aces.append(ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(num_channel, ir) for _, ir in ace.irreps_out]
                    ).simplify(),
                    self.irreps_coefs_out,
                    **coefs_kwargs,
                )
            )
            product_in1 = ace.irreps_out

        if num_channel != self.irreps_in.count((0, 1)) and self.irreps_in != self.irreps_hidden:
            self.linear_up = IrrepsLinear(self.irreps_in, self.irreps_hidden, bias=bias)
        else:
            # Still project if irreps differ in structure
            if self.irreps_in.simplify() != self.irreps_hidden.simplify():
                self.linear_up = IrrepsLinear(
                    self.irreps_in, self.irreps_hidden, bias=bias
                )
            else:
                self.linear_up = nn.Identity()

        self.linear = IrrepsLinear(
            o3.Irreps([(num_channel, ir) for _, ir in self.irreps_coefs_out]),
            self.irreps_out,
            bias=bias,
        )

    def forward(
        self,
        node_feats: Tensor,
        node_attrs: Tensor,
        sc: Optional[Tensor] = None,
    ) -> Tensor:
        node_feats = self.linear_up(node_feats)
        if self.agnostic:
            outs = self.coefs[0](node_feats)
        else:
            outs = self.coefs[0](node_feats, node_attrs)

        corr = {1: node_feats}
        for nu in range(2, self.correlation + 1):
            corr[nu] = self.aces[nu - 2](corr[nu - 1], node_feats)
            if self.agnostic:
                outs = outs + self.coefs[nu - 1](corr[nu])
            else:
                outs = outs + self.coefs[nu - 1](corr[nu], node_attrs)

        outs = self.linear(outs)
        if sc is not None:
            outs = outs + sc
        return outs
