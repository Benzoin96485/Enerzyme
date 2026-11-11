"""Degree-balanced equivariant RMS normalization for flat e3nn features.

:class:`EquivariantRMSNormSH` shares one scale across degrees and centers all
even scalars together. EquFlashV2 uses it.

:class:`RMSLayerNorm` is the eSEN / EquiformerV2 form used by Nequix: each
irrep block is weighted by ``1 / (dim * n_blocks)``, the block norms are
summed, and affine scales are per multiplicity. Centering, when enabled, is
the mean over channels inside each scalar block.
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


class RMSLayerNorm(nn.Module):
    """eSEN equivariant RMSNorm on a flat irreps tensor.

    Matches the Nequix layer norm (centering off in the published model):
    each ``(mul, ir)`` block contributes
    ``mean_m(sum_u x^2) / (dim * n_blocks)``, those contributions are summed,
    and the inverse square root scales every component. Affine weights have
    one value per channel of each block. Scalar biases exist only when
    ``centering`` is on.
    """

    def __init__(
        self,
        irreps,
        eps: float = 1e-12,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
    ) -> None:
        super().__init__()
        self.irreps = Irreps(irreps)
        if len(self.irreps) == 0:
            raise ValueError("RMSLayerNorm requires a non-empty irreps")
        self.eps = float(eps)
        self.affine = bool(affine)
        self.centering = bool(centering)
        self.std_balance_degrees = bool(std_balance_degrees)
        if not self.std_balance_degrees:
            raise NotImplementedError("RMSLayerNorm requires std_balance_degrees")

        slices = list(self.irreps.slices())
        self._slices = tuple((sl.start, sl.stop) for sl in slices)
        if self.affine:
            self.affine_weight = nn.ParameterList(
                [nn.Parameter(torch.ones(mul)) for mul, _ in self.irreps]
            )
            if self.centering:
                self.affine_bias = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(mul)) if ir.l == 0 else None
                        for mul, ir in self.irreps
                    ]
                )
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

    def forward(self, node_input: Tensor) -> Tensor:
        if node_input.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"feature width {node_input.shape[-1]} != irreps.dim {self.irreps.dim}"
            )
        chunks = []
        norms = []
        n_blocks = len(self.irreps)
        for (mul, ir), (start, stop) in zip(self.irreps, self._slices):
            values = node_input.narrow(-1, start, stop - start).reshape(
                *node_input.shape[:-1], mul, ir.dim
            )
            if self.centering and ir.l == 0:
                values = values - values.mean(dim=-2, keepdim=True)
            chunks.append(values)
            weight = 1.0 / (ir.dim * n_blocks)
            # Sum channels, then average components. The 1/n_blocks lets the
            # later sum over blocks recover a uniform mixture.
            block_norm = (values.square() * weight).sum(dim=-2, keepdim=True)
            norms.append(block_norm.mean(dim=-1, keepdim=True))
        norm = torch.cat(norms, dim=-1).sum(dim=-1, keepdim=True)
        norm = torch.pow(norm + self.eps, -0.5)
        outputs = []
        for index, ((mul, ir), values) in enumerate(zip(self.irreps, chunks)):
            if self.affine_weight is not None:
                values = values * self.affine_weight[index].view(
                    *([1] * (values.ndim - 2)), mul, 1
                )
            values = values * norm
            if self.affine_bias is not None and ir.l == 0:
                bias = self.affine_bias[index]
                values = values + bias.view(*([1] * (values.ndim - 2)), mul, 1)
            outputs.append(values.reshape(*node_input.shape[:-1], mul * ir.dim))
        return torch.cat(outputs, dim=-1)
