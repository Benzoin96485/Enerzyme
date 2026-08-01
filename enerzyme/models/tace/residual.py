"""Residual helpers for TACE BB skip connections."""

from __future__ import annotations

from typing import Literal, Union

from e3nn import o3

from .linear import ElementIrrepsLinear, IrrepsLinear, SkipIdentity


def get_resnet_layer(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    bias: bool,
    num_elements: int,
    resnet_type: Literal["aware", "agnostic", "identity"] = "aware",
):
    if resnet_type == "agnostic":
        return IrrepsLinear(irreps_in, irreps_out, bias=bias)
    if resnet_type == "identity":
        return SkipIdentity(irreps_in, irreps_out)
    return ElementIrrepsLinear(
        irreps_in, irreps_out, bias=bias, num_elements=num_elements
    )
