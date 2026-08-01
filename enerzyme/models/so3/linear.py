"""Degree-wise linear maps on spherical harmonic channel embeddings.

``SO3_LinearV2`` is adapted from EquiformerV2 (Liao et al., ICLR 2024; MIT).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import Parameter

from .embedding import SO3_Embedding


class SO3_LinearV2(torch.nn.Module):
    """Per-degree linear map: shared weight for all ``m`` of a given ``l``."""

    def __init__(
        self, in_features: int, out_features: int, lmax: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, input_embedding: SO3_Embedding) -> SO3_Embedding:
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        out = torch.einsum("bmi,moi->bmo", input_embedding.embedding, weight)
        if self.bias is not None:
            out = out.clone()
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias.view(1, 1, self.out_features)

        out_embedding = SO3_Embedding(
            0,
            input_embedding.lmax_list.copy(),
            self.out_features,
            device=input_embedding.device,
            dtype=input_embedding.dtype,
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(
            input_embedding.lmax_list.copy(), input_embedding.lmax_list.copy()
        )
        return out_embedding

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, lmax={self.lmax})"
        )


class SO3Linear(torch.nn.Module):
    """Per-degree linear on raw SH tensors ``[N, (lmax+1)**2, C]`` (EquiformerV3).

    Distinct from ``SO3_LinearV2``, which wraps ``SO3_Embedding``.
    """

    def __init__(self, in_features, out_features, lmax, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.weight = Parameter(torch.randn((self.lmax + 1), out_features, in_features))
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = Parameter(torch.zeros(1, 1, out_features)) if bias else None
        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        outputs = torch.einsum("bmi, moi -> bmo", inputs, weight)
        if self.bias is not None:
            outputs = outputs.clone()
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.bias
        return outputs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, lmax={self.lmax}, "
            f"bias={(self.bias is not None)})"
        )


class SO3FocusLinear(torch.nn.Module):
    """Degree-wise linear on packed ``(N, D, F, C)`` features (DPA4 / SeZM).

    Weights are shared across ``m`` within each ``l``, with an explicit focus
    stream axis ``F``. Distinct from :class:`SO3Linear` (``[N, D, C]``,
    EquiformerV3) and :class:`SO3_LinearV2` (``SO3_Embedding`` wrapper).
    """

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
        from .indexing import map_degree_idx

        self.lmax = int(lmax)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.mlp_bias = bool(mlp_bias)
        weight = torch.empty(
            self.lmax + 1, self.in_channels, self.n_focus * self.out_channels
        )
        if init_std is not None:
            if init_std == 0.0:
                torch.nn.init.zeros_(weight)
            else:
                torch.nn.init.normal_(weight, std=float(init_std))
        else:
            torch.nn.init.xavier_uniform_(weight.view(self.lmax + 1, -1))
        self.weight = Parameter(weight)
        if self.mlp_bias:
            self.bias = Parameter(torch.zeros(self.n_focus * self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "expand_index",
            torch.as_tensor(map_degree_idx(self.lmax), dtype=torch.long),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lmax={self.lmax}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"n_focus={self.n_focus})"
        )
