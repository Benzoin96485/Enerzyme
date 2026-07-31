"""Activations and SO(3) linears for DPA4.

Reimplemented after deepmd-kit ``dpa4_nn.{activation,so3}`` (Li et al., arXiv:2606.02419).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .indexing import build_m_major_l_index, map_degree_idx


class SO3Linear(nn.Module):
    """Degree-wise channel mixing on packed ``(N, D, F, C)`` features."""

    def __init__(
        self,
        lmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        mlp_bias: bool = False,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.mlp_bias = bool(mlp_bias)
        weight = torch.empty(self.lmax + 1, self.in_channels, self.n_focus * self.out_channels)
        if init_std is not None:
            if init_std == 0.0:
                nn.init.zeros_(weight)
            else:
                nn.init.normal_(weight, std=float(init_std))
        else:
            nn.init.xavier_uniform_(weight.view(self.lmax + 1, -1))
        self.weight = nn.Parameter(weight)
        if self.mlp_bias:
            self.bias = nn.Parameter(torch.zeros(self.n_focus * self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "expand_index", torch.as_tensor(map_degree_idx(self.lmax), dtype=torch.long)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, D, F, Cin)
        w = self.weight.view(
            self.lmax + 1, self.in_channels, self.n_focus, self.out_channels
        )
        w = w.index_select(0, self.expand_index)  # (D, Cin, F, Cout)
        w = w.permute(0, 2, 1, 3)  # (D, F, Cin, Cout)
        out = torch.matmul(x.unsqueeze(-2), w.unsqueeze(0)).squeeze(-2)
        if self.bias is not None:
            bias = self.bias.view(1, 1, self.n_focus, self.out_channels)
            out = torch.cat([out[:, :1] + bias, out[:, 1:]], dim=1)
        return out


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
        # x: (..., F, Cin) with F on second-to-last among leading dims — expect (N, F, Cin)
        # or (F, E, Cin) etc. Contract last dim with weight batched on focus.
        # General: treat focus as axis -2.
        out = torch.einsum("...fi,fio->...fo", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class GatedActivation(nn.Module):
    """Gated activation for packed or m-major reduced layouts."""

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
            raise ValueError(f"Unsupported DPA4 gated activation: {activation}")
        if self.lmax > 0:
            if self.mmax is None:
                expand = map_degree_idx(self.lmax)[1:] - 1
            else:
                expand = build_m_major_l_index(self.lmax, self.mmax)[1:] - 1
            self.register_buffer("expand_index", torch.as_tensor(expand, dtype=torch.long))
            self.gate_linear = FocusLinear(
                self.channels, self.lmax * self.channels, n_focus=self.n_focus, bias=mlp_bias
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
            g0 = g_ndfc[:, 0]  # (N, F, C)
            x0 = x_ndfc[:, 0] * F.silu(g0)
            if self.lmax == 0:
                out = x0.unsqueeze(1)
                return self._from_ndfc(out)
            gate_logits = self.gate_linear(g0)  # (N, F, lmax*C)
            gate_logits = gate_logits.view(
                g0.shape[0], self.n_focus, self.lmax, self.channels
            )
            gates = torch.sigmoid(gate_logits)
            gates = gates.index_select(2, self.expand_index)  # (N, F, D-1, C)
            xt = x_ndfc[:, 1:] * gates.permute(0, 2, 1, 3)
            out = torch.cat([x0.unsqueeze(1), xt], dim=1)
            return self._from_ndfc(out)

        x0 = F.silu(x_ndfc[:, 0])
        if self.lmax == 0:
            return self._from_ndfc(x0.unsqueeze(1))
        gate_logits = self.gate_linear(x_ndfc[:, 0])
        gate_logits = gate_logits.view(
            x_ndfc.shape[0], self.n_focus, self.lmax, self.channels
        )
        gates = torch.sigmoid(gate_logits).index_select(2, self.expand_index)
        xt = x_ndfc[:, 1:] * gates.permute(0, 2, 1, 3)
        return self._from_ndfc(torch.cat([x0.unsqueeze(1), xt], dim=1))


class SwiGLU(nn.Module):
    """SwiGLU over the last dimension, split into value and gate halves."""

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even last dimension")
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)
