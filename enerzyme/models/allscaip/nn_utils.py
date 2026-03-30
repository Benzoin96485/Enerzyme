# SPDX-License-Identifier: MIT
# Adapted from fairchem AllScAIP (https://github.com/FAIR-Chem/fairchem), MIT License.
from __future__ import annotations

import torch.nn as nn


def _activation_module(name: str) -> nn.Module:
    n = name.lower()
    if n in ("silu", "swish"):
        return nn.SiLU()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU()
    if n == "tanh":
        return nn.Tanh()
    return nn.SiLU()


def get_feedforward(
    hidden_dim: int,
    activation_name: str,
    hidden_layer_multiplier: int,
    bias: bool = True,
    dropout: float = 0.0,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> nn.Module:
    act = _activation_module(activation_name)
    in_d = hidden_dim if input_dim is None else input_dim
    out_d = hidden_dim if output_dim is None else output_dim
    if hidden_layer_multiplier == 0:
        lin = nn.Linear(in_d, out_d, bias=bias)
        if dropout > 0:
            return nn.Sequential(lin, nn.Dropout(dropout))
        return lin
    mid = hidden_dim * hidden_layer_multiplier
    seq: list[nn.Module] = [nn.Linear(in_d, mid, bias=bias), act]
    if dropout > 0:
        seq.append(nn.Dropout(dropout))
    seq.append(nn.Linear(mid, out_d, bias=bias))
    if dropout > 0:
        seq.append(nn.Dropout(dropout))
    return nn.Sequential(*seq)


def get_normalization(norm: str, width: int) -> nn.Module:
    _ = norm
    return nn.LayerNorm(width)
