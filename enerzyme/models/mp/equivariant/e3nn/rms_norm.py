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


from ...nn.norm import flat_rms_scale


def multiplicity_index(irreps):
    """Map flat components to indices in a concatenated list of channel gains."""
    widths = torch.tensor([ir.dim for mul, ir in irreps for _ in range(mul)])
    return torch.arange(irreps.num_irreps).repeat_interleave(widths)


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
        offset = 0
        for mul, ir in self.irreps:
            width = mul * ir.dim
            if self.std_balance_degrees:
                weight = 1.0 / (ir.dim * mul * (self.lmax + 1))
            else:
                weight = 1.0 / self.irreps.dim
            balance.extend([weight] * width)
            if ir.l == 0 and ir.p == 1:
                scalar_index.extend(range(offset, offset + width))
            offset += width
        self.register_buffer("balance_weight", torch.tensor(balance), persistent=False)
        self.register_buffer(
            "scalar_index",
            torch.tensor(scalar_index, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "affine_index",
            multiplicity_index(self.irreps),
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
            centered[..., self.scalar_index] = scalars
            values = centered
        scale = flat_rms_scale(values, self.balance_weight, self.eps)
        if self.affine:
            scale = scale * self.affine_weight[self.affine_index]
        out = values * scale
        if self.affine and self.affine_bias is not None:
            biased = out.clone()
            biased[..., self.scalar_index] = (
                out[..., self.scalar_index] + self.affine_bias
            )
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
        self.scalar_slices = tuple(
            sl for (_, ir), sl in zip(self.irreps, slices) if ir.l == 0
        )
        self.register_buffer(
            "affine_index", multiplicity_index(self.irreps), persistent=False
        )
        # The original block reduction sums over channels and averages over m.
        divisors = [
            ir.dim**2 * len(self.irreps)
            for mul, ir in self.irreps
            for _ in range(mul * ir.dim)
        ]
        self.register_buffer(
            "component_divisors", torch.tensor(divisors), persistent=False
        )
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
        values = node_input
        if self.centering and self.scalar_slices:
            values = node_input.clone()
            for sl in self.scalar_slices:
                block = node_input[..., sl]
                values[..., sl] = block - block.mean(dim=-1, keepdim=True)
        balance = self.component_divisors.to(dtype=values.dtype).reciprocal()
        scale = flat_rms_scale(values, balance, self.eps)
        if self.affine_weight is not None:
            gain = torch.cat(list(self.affine_weight)).index_select(
                0, self.affine_index
            )
            values = values * gain
        out = values * scale
        if self.affine_bias is not None:
            biases = (bias for bias in self.affine_bias if bias is not None)
            for sl, bias in zip(self.scalar_slices, biases):
                out[..., sl] = out[..., sl] + bias
        return out


def one_irrep_per_degree(irreps) -> Irreps:
    """Require ``0e/0o, 1e/1o, ...`` with no missing or repeated degree."""
    parsed = Irreps(irreps)
    if len(parsed) == 0:
        raise ValueError("EQNorm irreps must be non-empty")
    if parsed.lmax + 1 != len(parsed):
        raise ValueError(
            "EQNorm expects one irrep for every degree from 0 through lmax, "
            f"got {parsed}"
        )
    for degree, (mul, ir) in enumerate(parsed):
        if ir.l != degree:
            raise ValueError(
                "EQNorm expects irreps ordered by degree, "
                f"but position {degree} is {mul}x{ir} in {parsed}"
            )
        if mul < 1:
            raise ValueError(f"EQNorm irrep multiplicity must be >= 1, got {parsed}")
    return parsed


class EqnormRMSNorm(nn.Module):
    """Degree-balanced RMSNorm used inside every EQNorm convolution.

    ``centering`` subtracts the channel mean of the scalar block before the
    norm. The published convolutions leave centering off, so there is no
    scalar bias. Affine scales are one value per channel of each degree and
    are shared by the ``2l + 1`` components.
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
        self.irreps = one_irrep_per_degree(irreps)
        self.lmax = int(self.irreps.lmax)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.centering = bool(centering)
        self.std_balance_degrees = bool(std_balance_degrees)
        if not self.std_balance_degrees:
            raise NotImplementedError("EQNorm RMSNorm requires std_balance_degrees")
        stops = [sl.stop for sl in self.irreps.slices()]
        self.slices = (0, *stops)
        self.channels = [mul for mul, _ in self.irreps]
        self.max_dim = max(self.channels)
        if self.affine:
            self.affine_weight = nn.ParameterList(
                [nn.Parameter(torch.ones(mul)) for mul, _ in self.irreps]
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.irreps[0].mul))
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

        balance_degree = torch.zeros((self.lmax + 1) ** 2, 1)
        balance_channel = torch.zeros(self.max_dim, 1)
        for degree in range(self.lmax + 1):
            start = degree**2
            length = 2 * degree + 1
            balance_degree[start : start + length, :] = 1.0 / length
            balance_channel[: self.irreps[degree].mul, :] += 1
        balance = torch.einsum("ai,bi->ab", balance_degree, 1.0 / balance_channel)
        self.register_buffer("balance_weight", balance)
        self.register_buffer(
            "affine_index", multiplicity_index(self.irreps), persistent=False
        )
        balance_index = [
            m * self.max_dim + channel
            for mul, ir in self.irreps
            for channel in range(mul)
            for m in range(ir.l**2, (ir.l + 1) ** 2)
        ]
        self.register_buffer(
            "balance_index", torch.tensor(balance_index), persistent=False
        )

    def forward(self, node_input: Tensor) -> Tensor:
        if node_input.ndim != 2 or node_input.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"EQNorm RMSNorm expects [N, {self.irreps.dim}], "
                f"got {tuple(node_input.shape)}"
            )
        if self.centering:
            scalars = node_input[:, self.slices[0] : self.slices[1]]
            scalars = scalars - scalars.mean(dim=1, keepdim=True)
            if self.lmax == 0:
                feature = scalars
            else:
                feature = torch.cat((scalars, node_input[:, self.slices[1] :]), dim=1)
        else:
            feature = node_input

        if self.affine_weight is not None:
            balance = self.balance_weight.flatten().index_select(0, self.balance_index)
            scale = flat_rms_scale(feature, balance, self.eps, self.max_dim)
            weights = torch.cat(list(self.affine_weight)).index_select(
                0, self.affine_index
            )
            feature = feature * scale * weights
        if self.affine_bias is not None:
            width = self.slices[1] - self.slices[0]
            feature = feature.clone()
            feature[:, :width] = feature[:, :width] + self.affine_bias.view(1, -1)
        return feature
