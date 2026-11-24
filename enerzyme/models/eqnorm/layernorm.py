"""EQNorm equivariant RMS normalization.

This is the variable-multiplicity norm in
``yzchen08/eqnorm`` ``layernorm.RMSLayerNorm`` (MIT). It is the fairchem
Equiformer component norm, generalized to a different channel count at each
degree: each spherical component is weighted by ``1 / (2l + 1)``, each
channel is divided by how many degrees actually contain it, and those
channel norms are averaged over the widest multiplicity.

:class:`enerzyme.models.mp.equivariant.e3nn.RMSLayerNorm` is a different formula (the
eSEN / Nequix block mean) and is not a substitute here. The layout this
norm accepts is one irrep per degree, ``l = 0, 1, ..., lmax``, in that
order. The published schedules satisfy that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from e3nn.o3 import Irreps
from torch import Tensor, nn


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
            start = degree ** 2
            length = 2 * degree + 1
            balance_degree[start : start + length, :] = 1.0 / length
            balance_channel[: self.irreps[degree].mul, :] += 1
        balance = torch.einsum("ai,bi->ab", balance_degree, 1.0 / balance_channel)
        self.register_buffer("balance_weight", balance)

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

        weights = None
        if self.affine_weight is not None:
            weights = torch.cat(
                [
                    weight.view(-1, 1).repeat(1, 2 * degree + 1).reshape(1, -1)
                    for degree, weight in enumerate(self.affine_weight)
                ],
                dim=1,
            )

        padded = []
        for degree in range(self.lmax + 1):
            block = feature[:, self.slices[degree] : self.slices[degree + 1]]
            block = block.reshape(-1, self.channels[degree], 2 * degree + 1).transpose(1, 2)
            block = F.pad(block, (0, self.max_dim - block.shape[-1]))
            padded.append(block)
        stacked = torch.cat(padded, dim=1)
        reduced = (stacked.square() * self.balance_weight.unsqueeze(0)).sum(dim=1)
        scale = (reduced.mean(dim=1, keepdim=True) + self.eps).pow(-0.5)
        if weights is not None:
            feature = feature * scale * weights
        if self.affine_bias is not None:
            width = self.slices[1] - self.slices[0]
            feature = feature.clone()
            feature[:, :width] = feature[:, :width] + self.affine_bias.view(1, -1)
        return feature
