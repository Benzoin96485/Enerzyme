# Adapted from liyy2/E2Former (MIT) https://github.com/liyy2/E2Former
"""SO(3) helpers used by E2Former attention (raw SH-array layout)."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..so3.linear import SO3Linear


class SO3Linear2Scalar(torch.nn.Module):
    """Map SH coeffs ``[N, (lmax+1)^2, C]`` to invariant vectors ``[N, out]``.

    Port of upstream ``SO3_Linear2Scalar_e2former`` used by QK attention alphas.
    """

    def __init__(self, in_features: int, out_features: int, lmax: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        half = out_features // 2

        self.weight = nn.Parameter(torch.randn((lmax + 1), half, in_features))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(1, 1, half))

        self.weight2 = nn.Parameter(torch.randn((lmax + 1), half, in_features))
        torch.nn.init.uniform_(self.weight2, -bound, bound)

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for lval in range(lmax + 1):
            start_idx = lval**2
            length = 2 * lval + 1
            expand_index[start_idx : (start_idx + length)] = lval
        self.register_buffer("expand_index", expand_index)

        self.final_linear = nn.Sequential(
            nn.Linear(half * (lmax + 1), out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, input_embedding: torch.Tensor) -> torch.Tensor:
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        x = input_embedding.reshape([output_shape.numel()] + [l_sum, hidden])

        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        out = torch.einsum("bmi, moi -> bmo", x, weight)
        out = out.clone()
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias

        weight2 = torch.index_select(self.weight2, dim=0, index=self.expand_index)
        out2 = torch.einsum("bmi, moi -> bmo", x, weight2)
        out2 = out2.clone()
        out2[:, 0:1, :] = out2.narrow(1, 0, 1)

        chunks = []
        for lval in range(self.lmax + 1):
            chunks.append(
                torch.sum(
                    out[:, lval**2 : (lval + 1) ** 2]
                    * out2[:, lval**2 : (lval + 1) ** 2],
                    dim=1,
                )
            )
        return self.final_linear(torch.cat(chunks, dim=-1)).reshape(
            output_shape + (self.out_features,)
        )


# Alias for call sites that mirror upstream naming.
SO3_Linear_e2former = SO3Linear
