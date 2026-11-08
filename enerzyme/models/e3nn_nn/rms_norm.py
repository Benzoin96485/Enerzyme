"""Degree-balanced equivariant RMS normalization for flat e3nn features.

One invariant scale is shared by every degree. Even scalars are centered
together. Affine weights are per channel. This is the normalization used
inside EquFlashV2 interaction and feed-forward blocks; other flat-Irreps
cores can reuse it.
"""

from __future__ import annotations

import torch
from e3nn.o3 import Irreps
from torch import Tensor, nn


class EquivariantRMSNormSH(nn.Module):
    """RMSNorm over a flat ``mul_ir`` irreps tensor.

    Component ``c`` of irrep ``(mul, ir)`` is weighted by
    ``1 / (ir.dim * mul * (lmax + 1))`` before the global root-mean-square,
    so higher degrees do not dominate the scale.
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-5,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
    ) -> None:
        super().__init__()
        self.irreps = Irreps(irreps).simplify()
        if len(self.irreps) == 0:
            raise ValueError("EquivariantRMSNormSH requires a non-empty irreps")
        self.eps = float(eps)
        self.affine = bool(affine)
        self.centering = bool(centering)
        self.std_balance_degrees = bool(std_balance_degrees)
        self.lmax = max(ir.l for _, ir in self.irreps)

        balance = []
        scalar_index = []
        affine_index = []
        offset = 0
        channel = 0
        for mul, ir in self.irreps:
            width = mul * ir.dim
            if self.std_balance_degrees:
                weight = 1.0 / (ir.dim * mul * (self.lmax + 1))
            else:
                weight = 1.0 / self.irreps.dim
            balance.extend([weight] * width)
            if ir.l == 0 and ir.p == 1:
                scalar_index.extend(range(offset, offset + width))
            for _ in range(mul):
                affine_index.extend([channel] * ir.dim)
                channel += 1
            offset += width
        self.register_buffer("balance_weight", torch.tensor(balance), persistent=False)
        self.register_buffer(
            "scalar_index",
            torch.tensor(scalar_index, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "affine_index",
            torch.tensor(affine_index, dtype=torch.long),
            persistent=False,
        )
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.irreps.num_irreps))
            n_scalar = len(scalar_index)
            if self.centering and n_scalar:
                self.affine_bias = nn.Parameter(torch.zeros(n_scalar))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"feature width {features.shape[-1]} != irreps.dim {self.irreps.dim}"
            )
        values = features
        if self.centering and self.scalar_index.numel():
            scalars = values.index_select(-1, self.scalar_index)
            scalars = scalars - scalars.mean(dim=-1, keepdim=True)
            centered = values.clone()
            centered[:, self.scalar_index] = scalars
            values = centered
        square = values.pow(2) * self.balance_weight
        scale = (square.sum(dim=-1, keepdim=True) + self.eps).pow(-0.5)
        if self.affine:
            scale = scale * self.affine_weight[self.affine_index]
        out = values * scale
        if self.affine and self.affine_bias is not None:
            biased = out.clone()
            biased[:, self.scalar_index] = out[:, self.scalar_index] + self.affine_bias
            out = biased
        return out
