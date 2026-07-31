"""Degree-wise linear maps on spherical harmonic channel embeddings.

``SO3_LinearV2`` is adapted from EquiformerV2 (Liao et al., ICLR 2024; MIT).
"""

from __future__ import annotations

import math

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
