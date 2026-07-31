"""Radial channel MLP shared by Equiformer / EquiformerV2.

``RadialMLP`` is Linear → (LayerNorm → SiLU)* → Linear, with an optional
learnable offset on the last layer (Equiformer V1 ``RadialProfile``).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn import init


class RadialMLP(nn.Module):
    """Construct a radial function from a list of channel widths."""

    def __init__(
        self,
        channels_list: Sequence[int],
        use_layer_norm: bool = True,
        use_offset: bool = False,
    ) -> None:
        super().__init__()
        channels_list = list(channels_list)
        modules: List[nn.Module] = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            if (i == len(channels_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, channels_list[i], bias=use_biases))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(channels_list[-1]))
            fan_in = channels_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.offset, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        f_out = self.net(inputs)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
        return f_out


class RadialFunction(RadialMLP):
    """EquiformerV2-style radial MLP (LayerNorm + SiLU, no offset)."""

    def __init__(self, channels_list: Sequence[int]) -> None:
        super().__init__(channels_list, use_layer_norm=True, use_offset=False)


class RadialProfile(RadialMLP):
    """Equiformer V1-style radial MLP (optional offset on last layer)."""

    def __init__(
        self,
        ch_list: Sequence[int],
        use_layer_norm: bool = True,
        use_offset: bool = True,
    ) -> None:
        super().__init__(ch_list, use_layer_norm=use_layer_norm, use_offset=use_offset)
