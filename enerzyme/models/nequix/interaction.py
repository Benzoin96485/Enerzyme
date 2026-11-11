"""Nequix interaction block.

Separable ``uvu`` convolution with spherical harmonics (multiplicity 1), a
bias-free radial MLP, a species-independent linear residual, optional eSEN
RMSNorm, and a SiLU / tanh gate. This is the JAX kernel path in
https://github.com/atomicarchitects/nequix (MIT), written on e3nn instead of
OpenEquivariance.

The radial MLP emits one weight per input channel of each Clebsch-Gordan
path (``Irreps.num_irreps``). With ``mul(Y) = 1`` that is exactly the e3nn
``uvu`` weight count.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..e3nn_nn import tp_out_irreps_with_instructions
from ..e3nn_nn.activation import Activation
from ..e3nn_nn.rms_norm import RMSLayerNorm


def _keep(irreps: o3.Irreps, keep: Sequence[str]) -> o3.Irreps:
    wanted = {o3.Irrep(item) for item in keep}
    return o3.Irreps([(mul, ir) for mul, ir in irreps if ir in wanted])


def _drop(irreps: o3.Irreps, drop: Sequence[str]) -> o3.Irreps:
    rejected = {o3.Irrep(item) for item in drop}
    return o3.Irreps([(mul, ir) for mul, ir in irreps if ir not in rejected])


def _split_scalars(irreps: o3.Irreps, n_extra: int) -> Tuple[o3.Irreps, o3.Irreps]:
    """Cut scalar irreps so the first ``n_extra`` channels stay data."""
    extra: List[Tuple[int, o3.Irrep]] = []
    gates: List[Tuple[int, o3.Irrep]] = []
    remaining = int(n_extra)
    for mul, ir in irreps:
        if ir.l != 0:
            raise ValueError(f"scalar split received {mul}x{ir}")
        if remaining >= mul:
            extra.append((mul, ir))
            remaining -= mul
        elif remaining > 0:
            extra.append((remaining, ir))
            gates.append((mul - remaining, ir))
            remaining = 0
        else:
            gates.append((mul, ir))
    if remaining != 0:
        raise ValueError(
            f"need {n_extra} scalar channels to split, irreps only have {irreps.dim}"
        )
    return o3.Irreps(extra), o3.Irreps(gates)


def _require_scalar_prefix(irreps: o3.Irreps) -> None:
    """Gate channels are a suffix of the scalar prefix, then ``l > 0``."""
    seen_nonscalar = False
    for _, ir in irreps:
        if ir.l == 0:
            if seen_nonscalar:
                raise ValueError(
                    "Nequix gate scalars must form a prefix so the gate "
                    f"channels are the trailing scalars. Got {irreps}"
                )
        else:
            seen_nonscalar = True


def _scalar_acts(irreps: o3.Irreps):
    acts = []
    for _, ir in irreps:
        if ir.p == 1:
            acts.append(F.silu)
        elif ir.p == -1:
            acts.append(torch.tanh)
        else:
            raise ValueError(f"scalar activation needs definite parity, got {ir}")
    return acts


class NequixGate(nn.Module):
    """SiLU on even scalars, tanh on odd scalars, SiLU gates on ``l > 0``.

    Gate channels are the trailing scalar channels. Activations are
    second-moment normalized, matching ``e3nn_jax.gate(..., normalize_act=True)``.
    """

    def __init__(self, irreps_in) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)
        _require_scalar_prefix(irreps_in)
        self.irreps_in = irreps_in
        scalars = _keep(irreps_in, ("0e", "0o"))
        vectors = _drop(irreps_in, ("0e", "0o"))
        n_gates = vectors.num_irreps
        if scalars.dim < n_gates:
            raise ValueError(
                f"{scalars.dim} scalar channels cannot gate {n_gates} non-scalar irreps"
            )
        extra, gates = _split_scalars(scalars, scalars.dim - n_gates)
        self.extra_dim = extra.dim
        self.gate_dim = gates.dim
        self.act_extra: Optional[Activation]
        self.act_gates: Optional[Activation]
        if extra.dim:
            self.act_extra = Activation(extra, _scalar_acts(extra))
        else:
            self.act_extra = None
        if gates.dim:
            self.act_gates = Activation(gates, _scalar_acts(gates))
            self.mul: Optional[nn.Module] = o3.ElementwiseTensorProduct(gates, vectors)
        else:
            self.act_gates = None
            self.mul = None
        produced = o3.Irreps("")
        if self.act_extra is not None:
            produced = produced + self.act_extra.irreps_out
        if self.mul is not None:
            produced = produced + self.mul.irreps_out
        self.irreps_out = produced.simplify()

    def forward(self, features: Tensor) -> Tensor:
        parts = []
        if self.act_extra is not None:
            parts.append(self.act_extra(features.narrow(-1, 0, self.extra_dim)))
        if self.mul is not None:
            gates = self.act_gates(features.narrow(-1, self.extra_dim, self.gate_dim))
            vectors = features.narrow(
                -1,
                self.extra_dim + self.gate_dim,
                features.shape[-1] - self.extra_dim - self.gate_dim,
            )
            parts.append(self.mul(gates, vectors))
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-1)


class _BiasFreeLinear(nn.Module):
    """``y = x W`` with ``W_ij ~ N(0, init_scale / fan_in)``."""

    def __init__(self, in_size: int, out_size: int, init_scale: float) -> None:
        super().__init__()
        if in_size < 1 or out_size < 1:
            raise ValueError(f"linear size must be positive, got {in_size} -> {out_size}")
        scale = math.sqrt(float(init_scale) / float(in_size))
        self.weights = nn.Parameter(torch.randn(in_size, out_size) * scale)

    def forward(self, features: Tensor) -> Tensor:
        return features @ self.weights


class NequixRadialMLP(nn.Module):
    """Bias-free SiLU MLP. Hidden layers use ``init_scale``; the last does not."""

    def __init__(self, sizes: Sequence[int], init_scale: float = 4.0) -> None:
        super().__init__()
        widths = [int(size) for size in sizes]
        if len(widths) < 2:
            raise ValueError(f"radial MLP needs at least two widths, got {widths}")
        self.layers = nn.ModuleList(
            [
                _BiasFreeLinear(
                    widths[i],
                    widths[i + 1],
                    init_scale if i < len(widths) - 2 else 1.0,
                )
                for i in range(len(widths) - 1)
            ]
        )

    def forward(self, radial: Tensor) -> Tensor:
        hidden = radial
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last:
                hidden = F.silu(hidden)
        return hidden


def pre_gate_irreps(irreps_out: o3.Irreps) -> o3.Irreps:
    """Block output plus one even scalar gate per non-scalar channel."""
    irreps_out = o3.Irreps(irreps_out).simplify()
    n_odd_scalar = irreps_out.count("0o")
    if n_odd_scalar:
        raise ValueError(
            "Nequix gates are trailing even scalars; odd scalar outputs "
            f"({n_odd_scalar}x0o) are not part of this block"
        )
    n_gates = irreps_out.num_irreps - irreps_out.count("0e")
    return (irreps_out + o3.Irreps(f"{n_gates}x0e")).regroup()


class NequixConvolution(nn.Module):
    """One Nequix interaction: conv, linear residual, RMSNorm, gate.

    Path instructions come from ``tp_out_irreps_with_instructions``. That
    helper sorts duplicate blocks without merging them, which is the layout
    ``linear_2`` reads.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_sh,
        *,
        num_rbf: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_num_neighbors: float,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        if avg_num_neighbors <= 0:
            raise ValueError(f"avg_num_neighbors must be positive, got {avg_num_neighbors}")
        if radial_mlp_layers < 0:
            raise ValueError(f"radial_mlp_layers must be >= 0, got {radial_mlp_layers}")
        # Regroup so the gate layout (even scalars, then higher l) is the
        # tensor layout the next block reads. Duplicate tensor-product blocks
        # stay unmerged; linear_2 reads that sorted list.
        self.irreps_in = o3.Irreps(irreps_in).regroup()
        requested = o3.Irreps(irreps_out).regroup()
        irreps_sh = o3.Irreps(irreps_sh)
        paths, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, irreps_sh, requested
        )
        if not instructions:
            raise ValueError(
                f"no tensor-product paths {self.irreps_in} x {irreps_sh} -> {requested}"
            )
        self.tp = o3.TensorProduct(
            self.irreps_in,
            irreps_sh,
            paths,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        if self.tp.weight_numel != paths.num_irreps:
            raise RuntimeError(
                "uvu weight count "
                f"{self.tp.weight_numel} != path channels {paths.num_irreps}"
            )
        hidden = [int(radial_mlp_size)] * int(radial_mlp_layers)
        self.radial_mlp = NequixRadialMLP(
            [int(num_rbf), *hidden, self.tp.weight_numel],
            init_scale=float(mlp_init_scale),
        )
        self.pre_gate = pre_gate_irreps(requested)
        # Keep duplicate path blocks unmerged. ``o3.Linear`` normalizes each
        # block separately, so regrouping them (the torch port) is a different
        # layer from the JAX / OpenEquivariance convolution that was trained.
        self.linear_1 = o3.Linear(self.irreps_in, self.irreps_in, biases=False)
        self.linear_2 = o3.Linear(paths, self.pre_gate, biases=False)
        self.skip = o3.Linear(self.irreps_in, self.pre_gate, biases=False)
        if self.linear_2.irreps_in != paths:
            raise RuntimeError(
                "linear_2 must read the sorted unmerged tensor-product blocks "
                f"{paths}, got {self.linear_2.irreps_in}"
            )
        self.layer_norm: Optional[RMSLayerNorm]
        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                self.pre_gate,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None
        self.gate = NequixGate(self.pre_gate)
        n_data = requested.count("0e")
        n_gates = requested.num_irreps - n_data
        if self.gate.extra_dim != n_data or self.gate.gate_dim != n_gates:
            raise RuntimeError(
                f"gate slices are {self.gate.extra_dim} data + {self.gate.gate_dim} "
                f"gates, expected {n_data} data scalars and {n_gates} gates"
            )
        if self.gate.irreps_out != requested:
            raise RuntimeError(
                f"gate produced {self.gate.irreps_out}, expected {requested}"
            )
        self.irreps_out = requested
        self.inv_sqrt_neighbors = 1.0 / math.sqrt(float(avg_num_neighbors))

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        radial_basis: Tensor,
        edge_sh: Tensor,
    ) -> Tensor:
        messages = self.linear_1(features)
        weights = self.radial_mlp(radial_basis)
        if idx_i.numel() == 0:
            aggregated = features.new_zeros(features.shape[0], self.tp.irreps_out.dim)
        else:
            sent = self.tp(messages[idx_j], edge_sh, weights)
            aggregated = scatter_sum(sent, idx_i, dim=0, dim_size=features.shape[0])
        aggregated = aggregated * self.inv_sqrt_neighbors
        hidden = self.linear_2(aggregated) + self.skip(features)
        if self.layer_norm is not None:
            hidden = self.layer_norm(hidden)
        return self.gate(hidden)
