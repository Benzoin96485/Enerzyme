"""SO(2) Gate nonlinearity for TECE edge operations.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

import torch

from .so2_utils import so2_expand_index


class SO2Gate(torch.nn.Module):
    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        scalar_act,
        tensor_act,
        channel_wise: bool = False,
        gate_m0: bool = True,
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.gate_m0 = gate_m0
        self.num_m0_components = lmax + 1
        gate_start = 0 if gate_m0 else 1

        if not channel_wise:
            if gate_start <= mmax:
                self.num_components, expand_index = so2_expand_index(
                    mmax, lmax, start=gate_start
                )
            else:
                self.num_components = 0
                expand_index = torch.empty(0, dtype=torch.long)
        else:
            expand_index = []
            offset = 0
            for m in range(gate_start, mmax + 1):
                index = torch.arange(lmax + 1) + offset
                expand_index.append(index)
                if m > 0:
                    expand_index.append(index)
                offset = offset + len(index)
            if expand_index:
                expand_index = torch.cat(expand_index, dim=0).long()
            else:
                expand_index = torch.empty(0, dtype=torch.long)
            self.num_components = offset

        self.register_buffer("expand_index", expand_index, persistent=False)
        self.scalar_act = scalar_act
        self.tensor_act = tensor_act

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        g = self.tensor_act(g).view(B, self.num_components, self.num_channel)
        g = torch.index_select(g, dim=1, index=self.expand_index)
        if self.gate_m0:
            return g * x
        x_m0 = self.scalar_act(x[:, : self.num_m0_components])
        x_m = g * x[:, self.num_m0_components :]
        return torch.cat((x_m0, x_m), dim=1)
