"""Radial channel MLPs.

``RadialMLP`` is Linear → ([LayerNorm] → SiLU)* → Linear, with an optional
learnable offset on the last layer (Equiformer V1 ``RadialProfile``). It covers
the Equiformer family (LayerNorm), NequIP / DPA4 (plain SiLU stack, built with
``radial_mlp``) and EquiformerV3 (``RadialFunctionExpand`` adds the L/m expand).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn import init


def e3nn_radial_mlp(widths: Sequence[int], act=torch.nn.functional.silu) -> nn.Sequential:
    """Native biasless, fan-in-scaled layers with e3nn's normalized activation.

    Returning the native sequence keeps layer names and JIT compilation compatible
    with checkpoints from MACE, EquFlash, and EQNorm.
    """
    from e3nn.nn import FullyConnectedNet

    return FullyConnectedNet(list(widths), act)


class RadialMLP(nn.Module):
    """Construct a radial function from a list of channel widths."""

    def __init__(
        self,
        channels_list: Sequence[int],
        use_layer_norm: bool = True,
        use_offset: bool = False,
        bias: bool = True,
        *,
        e3nn_style: bool = False,
        activation=torch.nn.functional.silu,
    ) -> None:
        super().__init__()
        channels_list = list(channels_list)
        if e3nn_style:
            if use_offset:
                raise ValueError("e3nn radial networks require a bias-free output")
            self.net = e3nn_radial_mlp(channels_list, activation)
            self.offset = None
            return
        modules: List[nn.Module] = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            if (i == len(channels_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = bias
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


def radial_mlp(
    n_in: int,
    n_out: int,
    hidden: Sequence[int] | None = None,
    bias: bool = False,
    use_layer_norm: bool = False,
) -> RadialMLP:
    """``RadialMLP`` from input / output widths; ``hidden=[0]`` means no hidden layer."""
    hidden = [int(h) for h in hidden] if hidden is not None else []
    if hidden == [0]:
        hidden = []
    return RadialMLP([int(n_in), *hidden, int(n_out)], use_layer_norm=use_layer_norm, bias=bias)


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



class RadialFunctionExpand(RadialMLP):
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
        super().__init__(channels_list, use_layer_norm=True, use_offset=False, bias=True)
        channels_list = list(channels_list)
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
        outputs = super().forward(inputs)
        if self.use_expand:
            if not self.use_rad_l_parametrization:
                outputs = outputs.view(outputs.shape[0], self.num_m_components, -1)
            else:
                outputs = outputs.view(outputs.shape[0], (self.lmax + 1), -1)
            outputs = torch.index_select(outputs, dim=1, index=self.expand_index)
        return outputs


class _BiasFreeLinear(nn.Module):
    """``y = x W`` with ``W_ij ~ N(0, init_scale / fan_in)``."""

    def __init__(self, in_size: int, out_size: int, init_scale: float) -> None:
        super().__init__()
        if in_size < 1 or out_size < 1:
            raise ValueError(f"linear size must be positive, got {in_size} -> {out_size}")
        scale = math.sqrt(float(init_scale) / float(in_size))
        self.weights = nn.Parameter(torch.randn(in_size, out_size) * scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weights


class NequixRadialMLP(nn.Module):
    """Bias-free SiLU MLP. Hidden layers use ``init_scale``; the last does not."""

    def __init__(self, sizes: Sequence[int], init_scale: float = 4.0) -> None:
        super().__init__()
        widths = [int(size) for size in sizes]
        if len(widths) < 2:
            raise ValueError(f"radial MLP needs at least two widths, got {widths}")
        self.layers = nn.ModuleList(
            [
                _BiasFreeLinear(
                    widths[i],
                    widths[i + 1],
                    init_scale if i < len(widths) - 2 else 1.0,
                )
                for i in range(len(widths) - 1)
            ]
        )

    def forward(self, radial: torch.Tensor) -> torch.Tensor:
        hidden = radial
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last:
                hidden = torch.nn.functional.silu(hidden)
        return hidden
