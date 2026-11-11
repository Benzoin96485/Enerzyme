"""Allegro tensor track.

Spherical harmonics are weighted per channel (and, by default, per irrep),
summed onto the center atom, and contracted with the directed pair features.
The contraction is an e3nn ``uuu`` tensor product: channel ``u`` of the pair
meets only channel ``u`` of the environment. That is the same coupling as
``allegro.nn.Contracter`` with ``path_channel_coupling=True``. Flat e3nn layout
is used instead of the upstream strided kernel, so Allegro checkpoints do not
load here. Triton and cuequivariance kernels are not required.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3
from torch import Tensor, nn


def scatter_sum(src: Tensor, index: Tensor, n_nodes: int) -> Tensor:
    """Sum ``src`` onto ``n_nodes`` rows. Empty ``index`` yields zeros."""
    out = src.new_zeros((n_nodes,) + tuple(src.shape[1:]))
    if index.numel() == 0:
        return out
    out.index_add_(0, index.long(), src)
    return out


class WeightedSphericalChannels(nn.Module):
    """Lift multiplicity-1 spherical harmonics to ``mul`` channels.

    ``weight_individual_irreps=True`` gives each irrep its own weight, matching
    ``MakeWeightedChannels``. The flat weight vector is ``[edge, channel, irrep]``.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        mul: int,
        weight_individual_irreps: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        if any(count != 1 for count, _ in self.irreps_in):
            raise ValueError(
                f"spherical-harmonic irreps must have multiplicity 1, got {self.irreps_in}"
            )
        if int(mul) < 1:
            raise ValueError(f"mul must be >= 1, got {mul}")
        self.mul = int(mul)
        self.weight_individual_irreps = bool(weight_individual_irreps)
        self.num_irreps = len(self.irreps_in)
        self.weight_numel = (
            self.num_irreps * self.mul
            if self.weight_individual_irreps
            else self.mul
        )
        self.irreps_out = o3.Irreps([(self.mul, ir) for _, ir in self.irreps_in])

    def forward(self, basis: Tensor, weights: Tensor) -> Tensor:
        if basis.ndim != 2 or basis.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"basis must be [edges, {self.irreps_in.dim}], got {tuple(basis.shape)}"
            )
        if weights.shape[-1] != self.weight_numel or weights.shape[0] != basis.shape[0]:
            raise ValueError(
                f"weights must be [{basis.shape[0]}, {self.weight_numel}], "
                f"got {tuple(weights.shape)}"
            )
        if basis.shape[0] == 0:
            return basis.new_zeros((0, self.irreps_out.dim))
        pieces = []
        if self.weight_individual_irreps:
            scale = weights.reshape(basis.shape[0], self.mul, self.num_irreps)
            for irrep_i, slc in enumerate(self.irreps_in.slices()):
                component = basis[:, slc].unsqueeze(1)
                pieces.append(
                    (scale[:, :, irrep_i].unsqueeze(-1) * component).reshape(
                        basis.shape[0], -1
                    )
                )
        else:
            scale = weights.reshape(basis.shape[0], self.mul, 1)
            for slc in self.irreps_in.slices():
                component = basis[:, slc].unsqueeze(1)
                pieces.append((scale * component).reshape(basis.shape[0], -1))
        return torch.cat(pieces, dim=-1)


class ChannelTensorProduct(nn.Module):
    """Channel-diagonal Clebsch–Gordan product.

    ``path_channel_coupling=True`` learns one path weight per channel.
    ``False`` shares that weight across channels. Normalization is e3nn
    ``component`` with no extra path average, which multiplies each path by
    ``sqrt(2 l_out + 1)`` as in Allegro's Wigner-3j contraction. Weights are
    initialized uniformly on ``[-sqrt(3), sqrt(3)]``.
    """

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        mul: int,
        path_channel_coupling: bool = True,
    ) -> None:
        super().__init__()
        self.base_in1 = o3.Irreps(irreps_in1)
        self.base_in2 = o3.Irreps(irreps_in2)
        self.base_out = o3.Irreps(irreps_out)
        self.mul = int(mul)
        if self.mul < 1:
            raise ValueError(f"mul must be >= 1, got {mul}")
        self.path_channel_coupling = bool(path_channel_coupling)
        self.irreps_in1 = o3.Irreps([(self.mul, ir) for _, ir in self.base_in1])
        self.irreps_in2 = o3.Irreps([(self.mul, ir) for _, ir in self.base_in2])
        self.irreps_out = o3.Irreps([(self.mul, ir) for _, ir in self.base_out])
        instructions = []
        for i_out, (_, ir_out) in enumerate(self.base_out):
            for i_1, (_, ir_1) in enumerate(self.base_in1):
                for i_2, (_, ir_2) in enumerate(self.base_in2):
                    if ir_out in ir_1 * ir_2:
                        instructions.append((i_1, i_2, i_out, "uuu", True))
        if not instructions:
            raise ValueError(
                "channel tensor product has no paths from "
                f"{self.base_in1} x {self.base_in2} into {self.base_out}"
            )
        self.tp = o3.TensorProduct(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            instructions,
            shared_weights=True,
            internal_weights=self.path_channel_coupling,
            irrep_normalization="component",
            path_normalization="none",
        )
        self.num_paths = len(instructions)
        if self.path_channel_coupling:
            nn.init.uniform_(self.tp.weight, -math.sqrt(3.0), math.sqrt(3.0))
            self.path_weight = None
        else:
            self.path_weight = nn.Parameter(torch.empty(self.num_paths))
            nn.init.uniform_(self.path_weight, -math.sqrt(3.0), math.sqrt(3.0))

    def _shared_channel_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        chunks = [
            self.path_weight[path].to(dtype=dtype, device=device).repeat(self.mul)
            for path in range(self.num_paths)
        ]
        return torch.cat(chunks, dim=0)

    def forward(self, pair: Tensor, environment: Tensor) -> Tensor:
        if pair.shape[-1] != self.irreps_in1.dim or environment.shape[-1] != self.irreps_in2.dim:
            raise ValueError(
                "tensor-product inputs have widths "
                f"{pair.shape[-1]} and {environment.shape[-1]}, expected "
                f"{self.irreps_in1.dim} and {self.irreps_in2.dim}"
            )
        if pair.shape[0] != environment.shape[0]:
            raise ValueError(
                f"pair and environment batches differ: {pair.shape[0]} vs {environment.shape[0]}"
            )
        if pair.shape[0] == 0:
            return pair.new_zeros((0, self.irreps_out.dim))
        if self.path_channel_coupling:
            return self.tp(pair, environment)
        weight = self._shared_channel_weight(pair.dtype, pair.device)
        return self.tp(pair, environment, weight)
