"""SO(2) complex product basis (Edge Cluster Expansion building block).

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from ..e3nn_nn.tools import satisfy


class UUUSo2TensorProduct(torch.nn.Module):
    """Channel-coupled SO(2) tensor product via complex sum/diff multiplications."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channels: int,
        m1m2: Union[str, None] = None,
        internal_weights: bool = True,
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channels = num_channels
        self.m1m2 = m1m2
        self.instructions = []

        self.num_paths = 0
        weight_numel = 0
        for m3 in range(mmax + 1):
            paths = self.enumerate_paths(m3)
            self.instructions.append(paths)
            weight_numel += num_channels * (lmax + 1) * len(paths)
            self.num_paths += len(paths)
        self.weight_numel = weight_numel

        if internal_weights:
            self.weight = torch.nn.Parameter(torch.randn(1, self.weight_numel))
        else:
            self.register_buffer("weight", None)
        self.internal_weights = internal_weights

        output_scales = []
        n = lmax + 1
        scale0 = torch.full((n,), 1.0 / math.sqrt(len(self.instructions[0])))
        output_scales.append(scale0)
        for m3 in range(1, mmax + 1):
            scale = 1.0 / math.sqrt(len(self.instructions[m3]))
            output_scales.append(torch.full((2 * n,), scale))
        self.register_buffer(
            "output_scales", torch.cat(output_scales), persistent=False
        )

    def enumerate_paths(self, m3: int):
        paths = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                if satisfy(m1, m2, self.m1m2):
                    if m1 + m2 == m3:
                        paths.append((m1, m2, "sum"))
                    elif abs(m1 - m2) == m3:
                        paths.append((m1, m2, "diff"))
        return paths

    def rmul(self, x, y):
        return x * y

    def cmul(self, x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
        a = x[:, 0]
        b = x[:, 1]
        c = y[:, 0]
        d = y[:, 1]
        if mode == "sum":
            real = a * c - b * d
            imag = a * d + b * c
        else:
            real = a * c + b * d
            imag = b * c - a * d
        B = real.size(0)
        C = real.size(-1)
        n = self.lmax + 1
        real = real.reshape(B, n, C)
        imag = imag.reshape(B, n, C)
        return torch.stack([real, imag], dim=1)

    def to_list(self, x: torch.Tensor):
        B = x.size(0)
        out = []
        offset = 0
        n = self.lmax + 1
        out.append(x[:, offset : offset + n])
        offset += n
        for _m in range(1, self.mmax + 1):
            xm = x[:, offset : offset + 2 * n]
            xm = xm.view(B, 2, n, self.num_channels)
            out.append(xm)
            offset += 2 * n
        return out

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        xs = self.to_list(x)
        ys = self.to_list(y)
        ws = self.weight if self.internal_weights else weight
        C = self.num_channels
        outputs = []
        w_offset = 0
        n = self.lmax + 1
        m0 = 0.0
        w_numel = C * n

        for m1, m2, mode in self.instructions[0]:
            w = ws[:, w_offset : w_offset + w_numel].view(-1, n, C)
            w_offset += w_numel
            if m1 == 0 and m2 == 0:
                z = self.rmul(xs[0], ys[0])
                m0 = m0 + z * w
            elif m1 > 0 and m2 > 0:
                z = self.cmul(xs[m1], ys[m2], "diff")
                m0 = m0 + z[:, 0] * w
        outputs.append(m0)

        for m3 in range(1, self.mmax + 1):
            real = 0.0
            imag = 0.0
            for m1, m2, mode in self.instructions[m3]:
                w = ws[:, w_offset : w_offset + w_numel]
                w_offset += w_numel
                w = w.view(-1, 1, n, C)
                if m1 == 0:
                    z = xs[m1].unsqueeze(1) * ys[m2]
                elif m2 == 0:
                    z = xs[m1] * ys[m2].unsqueeze(1)
                else:
                    if m1 < m2 and mode == "diff":
                        z = self.cmul(ys[m2], xs[m1], mode)
                    else:
                        z = self.cmul(xs[m1], ys[m2], mode)
                out = z * w
                real = real + out[:, 0]
                imag = imag + out[:, 1]
            outputs.append(real)
            outputs.append(imag)

        out = torch.cat(outputs, dim=1)
        return out * self.output_scales.view(1, -1, 1)


class ComplexProductBasis(torch.nn.Module):
    """Thin wrapper used as Edge Cluster Expansion product."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        m1m2: Union[str, None] = ">=",
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.m1m2 = m1m2
        self.tp = UUUSo2TensorProduct(
            self.mmax,
            self.lmax,
            self.num_channel,
            m1m2=self.m1m2,
            internal_weights=False,
        )
        self.weight_numel = self.tp.weight_numel

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, ws: torch.Tensor
    ) -> torch.Tensor:
        return self.tp(x, y, ws)
