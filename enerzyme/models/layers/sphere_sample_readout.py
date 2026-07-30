"""Sphere-sample readout generalizing eSCN EnergyBlock / ForceBlock.

Samples full spherical node features on fixed S² points, applies an MLP per
sample, then integrates (mean) to per-atom scalars — the same pattern as
Passaro & Zitnick (2023) energy heads, but with Enerzyme ``output_fields`` so
any atomic scalar property can be predicted. Optional ``vector_output_fields``
mirror the paper force head (scalar × sample direction).

Expects ``atom_sphere_feature`` in full degree-major layout
``(N, (lmax+1)**2, C)`` after message ``rotate_inv``. Reduced ``mmax`` is an
internal edge-frame SO(2) detail and is not present on this tensor.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Set

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, SiLU

from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from ..blocks.mlp import DenseLayer, ResidualLayer, ResidualMLP
from . import BaseFFLayer


def calc_sphere_points(num_points: int, device=None, dtype=torch.float32) -> Tensor:
    """Fibonacci-lattice sphere samples with density weights (fairchem SCN)."""
    golden_ratio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device, dtype=dtype).view(-1, 1)
    theta = 2 * math.pi * i / golden_ratio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    return points * (scalar.view(-1, 1))


class SphereSampleReadout(BaseFFLayer):
    """Integrate spherical atom features over S² samples into named fields."""

    def __init__(
        self,
        output_fields: Set[str],
        built_layers: List[Module],
        head_type: Literal["dense", "residual_layer", "residual_mlp", "escn_mlp"] = "escn_mlp",
        num_sphere_samples: int = 128,
        lmax: Optional[int] = None,
        sphere_channels: Optional[int] = None,
        vector_output_fields: Optional[Set[str]] = None,
        keep_feature: bool = False,
        shallow_ensemble_size: int = 1,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        **head_params,
    ) -> None:
        vector_output_fields = set(vector_output_fields or [])
        out = set(output_fields) | vector_output_fields
        if keep_feature:
            out = out | {"atom_feature", "atom_sphere_feature"}
        super().__init__(
            input_fields={"atom_sphere_feature"},
            output_fields=out,
        )
        if shallow_ensemble_size != 1:
            raise NotImplementedError(
                "SphereSampleReadout currently supports shallow_ensemble_size=1 only"
            )
        self.ordered_scalar_fields = sorted(list(output_fields))
        self.ordered_vector_fields = sorted(list(vector_output_fields))
        self.num_sphere_samples = num_sphere_samples
        self.head_type = head_type
        self.keep_feature = keep_feature
        self.activation_fn = activation_fn
        self.activation_params = activation_params
        self.head_params = head_params

        core = None
        for layer in reversed(built_layers):
            if hasattr(layer, "sphere_channels") and hasattr(layer, "lmax_list"):
                core = layer
                break
        if sphere_channels is None:
            if core is None:
                raise ValueError(
                    "sphere_channels must be set or Core with sphere_channels must precede "
                    "SphereSampleReadout in the layer stack"
                )
            sphere_channels = core.sphere_channels
        if lmax is None:
            if core is None:
                raise ValueError(
                    "lmax must be set or Core with lmax_list must precede SphereSampleReadout"
                )
            lmax = max(core.lmax_list)
        self.lmax = int(lmax)
        self.sphere_channels = int(sphere_channels)
        self.num_coefficients = (self.lmax + 1) ** 2
        self.dim_feature_in = self.sphere_channels

        points = calc_sphere_points(num_sphere_samples, device="cpu", dtype=torch.float32)
        self.register_buffer("sphere_points", points, persistent=True)
        # Match fairchem eSCN: o3.spherical_harmonics(degrees, points, False)
        with torch.no_grad():
            y = o3.spherical_harmonics(
                list(range(0, self.lmax + 1)),
                self.sphere_points,
                False,
            )
        self.register_buffer("sphharm_weights", y, persistent=True)

        n_scalar = len(self.ordered_scalar_fields)
        n_vector = len(self.ordered_vector_fields)
        self.scalar_head = self._make_head(n_scalar) if n_scalar else None
        self.vector_heads = ModuleList(
            [self._make_head(1) for _ in range(n_vector)]
        )

    def _make_head(self, dim_out: int) -> Module:
        if dim_out == 0:
            return None
        if self.head_type == "escn_mlp":
            return Sequential(
                torch.nn.Linear(self.dim_feature_in, self.dim_feature_in),
                SiLU(),
                torch.nn.Linear(self.dim_feature_in, self.dim_feature_in),
                SiLU(),
                torch.nn.Linear(self.dim_feature_in, dim_out, bias=False),
            )
        if self.head_type == "dense":
            return DenseLayer(
                dim_feature_in=self.dim_feature_in,
                dim_feature_out=dim_out,
                shallow_ensemble_size=1,
                **self.head_params,
            )
        if self.head_type == "residual_layer":
            return Sequential(
                ResidualLayer(
                    dim_feature_in=self.dim_feature_in,
                    dim_feature_out=self.dim_feature_in,
                    activation_fn=self.activation_fn,
                    activation_params=self.activation_params,
                    **self.head_params,
                ),
                DenseLayer(
                    dim_feature_in=self.dim_feature_in,
                    dim_feature_out=dim_out,
                    shallow_ensemble_size=1,
                    **self.head_params,
                ),
            )
        if self.head_type == "residual_mlp":
            return ResidualMLP(
                dim_feature_in=self.dim_feature_in,
                dim_feature_out=dim_out,
                shallow_ensemble_size=1,
                activation_fn=self.activation_fn,
                activation_params=self.activation_params,
                **self.head_params,
            )
        raise ValueError(f"Unknown head_type: {self.head_type}")

    def _sample(self, atom_sphere_feature: Tensor) -> Tensor:
        """Return (N, S, C) samples from (N, num_coeff, C)."""
        if atom_sphere_feature.ndim != 3:
            raise ValueError(
                f"atom_sphere_feature must be (N, coeff, C), got {tuple(atom_sphere_feature.shape)}"
            )
        n, n_coeff, c = atom_sphere_feature.shape
        if n_coeff < self.num_coefficients:
            raise ValueError(
                f"atom_sphere_feature has {n_coeff} coeffs but lmax={self.lmax} needs "
                f"{self.num_coefficients}"
            )
        # Use leading coefficients for this lmax (single-resolution layout)
        feat = atom_sphere_feature[:, : self.num_coefficients, :]
        y = self.sphharm_weights.to(dtype=feat.dtype, device=feat.device)
        # (N, C, coeff) x (S, coeff) -> via einsum nac,sa->nsc
        return torch.einsum("nac,sa->nsc", feat, y)

    def get_output(self, atom_sphere_feature: Tensor) -> Dict[str, Tensor]:
        samples = self._sample(atom_sphere_feature)  # (N, S, C)
        n, s, c = samples.shape
        flat = samples.reshape(n * s, c)
        out: Dict[str, Tensor] = {}

        if self.scalar_head is not None:
            pred = self.scalar_head(flat).view(n, s, -1)
            pred = pred.mean(dim=1)  # (N, F)
            for i, name in enumerate(self.ordered_scalar_fields):
                out[name] = pred[:, i]

        points = self.sphere_points.to(dtype=samples.dtype, device=samples.device)
        for i, name in enumerate(self.ordered_vector_fields):
            # scalar per sample → (N, S, 1) then * direction
            sc = self.vector_heads[i](flat).view(n, s, 1)
            vec = (sc * points.view(1, s, 3)).mean(dim=1)  # (N, 3)
            out[name] = vec

        if self.keep_feature:
            # l=0 slice as atom_feature for downstream optional use
            out["atom_sphere_feature"] = atom_sphere_feature
            out["atom_feature"] = atom_sphere_feature[:, 0, :]
        return out
