"""Focus-stream gated activations for packed / m-major SO(3) features (DPA4)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .indexing import build_m_major_l_index, map_degree_idx


class FocusLinear(nn.Module):
    """Linear applied independently per focus stream: ``(..., F, Cin)`` → ``(..., F, Cout)``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.weight = nn.Parameter(
            torch.empty(self.n_focus, self.in_channels, self.out_channels)
        )
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.n_focus, self.out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.einsum("...fi,fio->...fo", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class SO3GatedActivation(nn.Module):
    """Gated activation for packed or m-major reduced layouts (DPA4 / SeZM).

    Layouts: ``ndfc`` ``(N, D, F, C)``, ``nfdc`` ``(N, F, D, C)``,
    ``fndc`` ``(F, N, D, C)``.
    """

    def __init__(
        self,
        lmax: int,
        channels: int,
        n_focus: int = 1,
        mmax: Optional[int] = None,
        layout: str = "ndfc",
        mlp_bias: bool = False,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.channels = int(channels)
        self.n_focus = int(n_focus)
        self.mmax = None if mmax is None else int(mmax)
        self.layout = layout
        if activation not in {"silu", "swish"}:
            raise ValueError(f"Unsupported gated activation: {activation}")
        if self.lmax > 0:
            if self.mmax is None:
                expand = map_degree_idx(self.lmax)[1:] - 1
            else:
                expand = build_m_major_l_index(self.lmax, self.mmax)[1:] - 1
            self.register_buffer(
                "expand_index", torch.as_tensor(expand, dtype=torch.long)
            )
            self.gate_linear = FocusLinear(
                self.channels,
                self.lmax * self.channels,
                n_focus=self.n_focus,
                bias=mlp_bias,
            )
            nn.init.normal_(self.gate_linear.weight, std=0.01)
        else:
            self.register_buffer("expand_index", torch.zeros(0, dtype=torch.long))
            self.gate_linear = None

    def _to_ndfc(self, x: Tensor) -> Tensor:
        if self.layout == "ndfc":
            return x
        if self.layout == "nfdc":
            return x.permute(0, 2, 1, 3)
        if self.layout == "fndc":
            return x.permute(1, 2, 0, 3)
        raise ValueError(self.layout)

    def _from_ndfc(self, x: Tensor) -> Tensor:
        if self.layout == "ndfc":
            return x
        if self.layout == "nfdc":
            return x.permute(0, 2, 1, 3)
        if self.layout == "fndc":
            return x.permute(2, 0, 1, 3)
        raise ValueError(self.layout)

    def forward(self, x: Tensor, gate: Optional[Tensor] = None) -> Tensor:
        x_ndfc = self._to_ndfc(x)
        if gate is not None:
            g_ndfc = self._to_ndfc(gate)
            g0 = g_ndfc[:, 0]
            x0 = x_ndfc[:, 0] * F.silu(g0)
            if self.lmax == 0:
                return self._from_ndfc(x0.unsqueeze(1))
            gate_logits = self.gate_linear(g0).view(
                g0.shape[0], self.n_focus, self.lmax, self.channels
            )
            gates = torch.sigmoid(gate_logits).index_select(2, self.expand_index)
            xt = x_ndfc[:, 1:] * gates.permute(0, 2, 1, 3)
            return self._from_ndfc(torch.cat([x0.unsqueeze(1), xt], dim=1))

        x0 = F.silu(x_ndfc[:, 0])
        if self.lmax == 0:
            return self._from_ndfc(x0.unsqueeze(1))
        gate_logits = self.gate_linear(x_ndfc[:, 0]).view(
            x_ndfc.shape[0], self.n_focus, self.lmax, self.channels
        )
        gates = torch.sigmoid(gate_logits).index_select(2, self.expand_index)
        xt = x_ndfc[:, 1:] * gates.permute(0, 2, 1, 3)
        return self._from_ndfc(torch.cat([x0.unsqueeze(1), xt], dim=1))
