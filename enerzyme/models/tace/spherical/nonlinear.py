"""Spherical nonlinearities for TACE (sigmoid gate)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import nn

from ..linear import IrrepsLinear


def get_nonlinear_layer(
    nonlinear: Optional[str],
    irreps_mid: o3.Irreps,
    irreps_out: o3.Irreps,
    bias: bool = True,
) -> Tuple[nn.Module, nn.Module, o3.Irreps]:
    """Return (nonlinearity, post_linear, linear_down_irreps_out)."""
    if nonlinear is None or nonlinear in ("null", "none", "identity"):
        return nn.Identity(), nn.Identity(), irreps_mid

    # e.g. "sigmoid_gate" -> gate with sigmoid on scalars/gates
    if nonlinear.endswith("_gate") or nonlinear == "gate":
        act_name = nonlinear.split("_")[0] if "_" in nonlinear else "sigmoid"
        act = torch.nn.Sigmoid() if act_name == "sigmoid" else torch.nn.SiLU()

        even_scalar = o3.Irrep("0e")
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_mid if ir == even_scalar])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_mid if ir != even_scalar])
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        if len(irreps_gated) == 0:
            # only scalars: SiLU/sigmoid activation then linear
            from e3nn.nn import Activation

            nonlinearity = Activation(irreps_mid, [act] * len(irreps_mid))
            post = IrrepsLinear(nonlinearity.irreps_out, irreps_out, bias=bias)
            return nonlinearity, post, irreps_mid

        nonlinearity = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[act] * len(irreps_scalars),
            irreps_gates=irreps_gates,
            act_gates=[torch.nn.Sigmoid()] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        post = IrrepsLinear(nonlinearity.irreps_out, irreps_out, bias=bias)
        return nonlinearity, post, nonlinearity.irreps_in.simplify()

    raise ValueError(f"Unsupported nonlinear={nonlinear}")
