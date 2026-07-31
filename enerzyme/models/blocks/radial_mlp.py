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



class RadialFunctionExpand(nn.Module):
    """EquiformerV3 radial MLP with optional L/m expand for feature scaling.

    When ``use_expand`` is True and ``use_rad_l_parametrization`` is True,
    all m within a type-L vector share one radial weight (expanded to
    ``(lmax+1)**2``).
    """

    def __init__(
        self,
        channels_list: Sequence[int],
        lmax: int | None = None,
        mmax: int | None = None,
        use_rad_l_parametrization: bool = True,
        use_expand: bool = True,
    ) -> None:
        super().__init__()
        channels_list = list(channels_list)
        modules: List[nn.Module] = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            if i == len(channels_list) - 1:
                break
            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())
        self.net = nn.Sequential(*modules)
        self.lmax = lmax
        self.mmax = mmax
        self.use_rad_l_parametrization = use_rad_l_parametrization
        self.use_expand = use_expand
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                expand_index = []
                offset = 0
                for m in range(self.mmax + 1):
                    index = torch.arange((self.lmax + 1 - m))
                    index = index + offset
                    expand_index.append(index)
                    if m > 0:
                        expand_index.append(index)
                    offset = offset + len(index)
                expand_index = torch.cat(expand_index, dim=0).long()
                self.register_buffer("expand_index", expand_index)
                self.num_m_components = offset
                assert channels_list[-1] % self.num_m_components == 0
            else:
                assert self.lmax == self.mmax
                expand_index = torch.zeros([((self.lmax + 1) ** 2)]).long()
                start_idx = 0
                for l in range(self.lmax + 1):
                    length = 2 * l + 1
                    expand_index[start_idx : (start_idx + length)] = l
                    start_idx = start_idx + length
                self.register_buffer("expand_index", expand_index)
                assert channels_list[-1] % (self.lmax + 1) == 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.net(inputs)
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                outputs = outputs.view(outputs.shape[0], self.num_m_components, -1)
            else:
                outputs = outputs.view(outputs.shape[0], (self.lmax + 1), -1)
            outputs = torch.index_select(outputs, dim=1, index=self.expand_index)
        return outputs
