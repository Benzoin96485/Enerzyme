"""EQNorm convolution and dual-branch interaction.

The message is the ``yzchen08/eqnorm`` ``E3Conv`` (MIT): a bias-free
equivariant linear, a radial ``e3nn`` network on the Bessel basis, a ``uvu``
tensor product with real spherical harmonics, a residual linear, and
:class:`~enerzyme.models.eqnorm.layernorm.EqnormRMSNorm` with centering off.
Receivers are ``idx_i`` and senders are ``idx_j``. ``edge_sh`` is ``Y(R_j - R_i)``.

Path instructions stay in creation order. That is the radial-weight layout
of the reference ``get_path``. :func:`enerzyme.models.mp.equivariant.e3nn.tp_out_irreps_with_instructions`
sorts the same paths, which would move those weights; Nequix wants the
sorted layout, so this module keeps its own enumeration.

Each interaction runs two convolutions. The tensor convolution writes
scalars, gate channels, and higher-order irreps, then an ``e3nn`` gate
updates the tensor state. The scalar convolution reads that state and
writes only even scalars. It does not write the tensor state back.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import Tensor, nn

from ..mp.ops import scatter_sum
from ..nequip.interaction import tp_path_exists
from .layernorm import EqnormRMSNorm, one_irrep_per_degree


def eqnorm_tp_paths(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
) -> Tuple[o3.Irreps, List[Tuple[int, int, int, str, bool]]]:
    """Clebsch–Gordan paths in the reference creation order.

    The returned irreps are sorted. Instructions are not reordered, so a
    radial network trained against ``yzchen08/eqnorm`` addresses the same
    weight slots.
    """
    irreps_mid: List[Tuple[int, o3.Irrep]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (_, ir_sh) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_sh:
                if ir_out in irreps_out:
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, permut, _ = irreps_mid.sort()
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    return irreps_mid, instructions


def _acts(irreps: o3.Irreps):
    acts = []
    for _, ir in irreps:
        if ir.p == 1:
            acts.append(F.silu)
        elif ir.p == -1:
            acts.append(torch.tanh)
        else:
            raise ValueError(f"EQNorm gate needs a definite parity, got {ir}")
    return acts


class EqnormConv(nn.Module):
    """One EQNorm convolution, before the gate or the scalar residual."""

    def __init__(
        self,
        irreps_in,
        irreps_sh,
        irreps_out,
        *,
        num_rbf: int,
        invariant_layers: int,
        invariant_neurons: int,
        normalization: str,
        avg_num_neighbors: Optional[float],
        use_sc: bool = True,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.normalization = str(normalization)
        self.use_sc = bool(use_sc)
        if self.normalization not in ("constant", "degree"):
            raise ValueError(
                "normalization must be 'constant' or 'degree', "
                f"got {normalization!r}"
            )
        if self.normalization == "constant":
            if avg_num_neighbors is None or float(avg_num_neighbors) <= 0:
                raise ValueError(
                    "constant neighbor normalization needs a positive "
                    f"avg_num_neighbors, got {avg_num_neighbors}"
                )
            self.avg_num_neighbors = float(avg_num_neighbors)
        else:
            self.avg_num_neighbors = None
        if int(invariant_layers) < 0:
            raise ValueError(f"invariant_layers must be >= 0, got {invariant_layers}")
        if int(invariant_layers) > 0 and int(invariant_neurons) < 1:
            raise ValueError(
                "invariant_neurons must be >= 1 when invariant_layers > 0, "
                f"got {invariant_neurons}"
            )
        irreps_mid, instructions = eqnorm_tp_paths(
            self.irreps_in, self.irreps_sh, self.irreps_out
        )
        if not instructions:
            raise ValueError(
                "EQNorm convolution has no tensor-product paths from "
                f"{self.irreps_in} x {self.irreps_sh} into {self.irreps_out}"
            )
        self.linear_1 = o3.Linear(self.irreps_in, self.irreps_in, biases=False)
        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_sh,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        hidden = [int(invariant_neurons)] * int(invariant_layers)
        self.fc = FullyConnectedNet(
            [int(num_rbf), *hidden, self.tp.weight_numel],
            F.silu,
        )
        # Merge duplicate blocks. The reference ``linear_2`` reads the
        # simplified irreps, which is a different normalization from leaving
        # the sorted path blocks unmerged.
        self.linear_2 = o3.Linear(
            irreps_mid.simplify(),
            self.irreps_out,
            biases=False,
        )
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out, biases=False)
            self.ln = EqnormRMSNorm(self.irreps_out, centering=False)
        else:
            self.sc = None
            self.ln = None

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        radial_basis: Tensor,
        edge_sh: Tensor,
        envelope: Optional[Tensor] = None,
    ) -> Tensor:
        skip = self.sc(features) if self.sc is not None else None
        mixed = self.linear_1(features)
        n_atoms = features.shape[0]
        if idx_i.numel() == 0:
            aggregated = features.new_zeros(n_atoms, self.tp.irreps_out.dim)
        else:
            messages = self.tp(mixed[idx_j], edge_sh, self.fc(radial_basis))
            if self.normalization == "constant":
                messages = messages / math.sqrt(self.avg_num_neighbors)
            if envelope is not None:
                window = envelope.reshape(-1, 1).to(dtype=messages.dtype)
                messages = messages * window
            aggregated = scatter_sum(messages, idx_i, n_atoms)
        if self.normalization == "degree":
            degree = features.new_zeros(n_atoms)
            if idx_i.numel():
                degree.index_add_(
                    0,
                    idx_i,
                    torch.ones(idx_i.shape[0], dtype=features.dtype, device=features.device),
                )
            degree = degree.clamp_min(1).sqrt().unsqueeze(-1)
            aggregated = aggregated / degree
        out = self.linear_2(aggregated)
        if skip is not None:
            out = self.ln(out + skip)
        return out


class EqnormInteraction(nn.Module):
    """Tensor update plus a scalar message that does not enter the tensor state."""

    def __init__(
        self,
        irreps_in,
        irreps_hidden,
        irreps_sh,
        irreps_scalar,
        *,
        num_rbf: int,
        invariant_layers: int,
        invariant_neurons: int,
        normalization: str,
        avg_num_neighbors: Optional[float],
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = one_irrep_per_degree(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_scalar = o3.Irreps(irreps_scalar).simplify()
        scalars, gates, nonscalars = _gated_targets(
            self.irreps_in, self.irreps_hidden, self.irreps_sh
        )
        conv_out = (scalars + gates + nonscalars).sort().irreps.simplify()
        self.tensor_conv = EqnormConv(
            self.irreps_in,
            self.irreps_sh,
            conv_out,
            num_rbf=num_rbf,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            normalization=normalization,
            avg_num_neighbors=avg_num_neighbors,
        )
        if nonscalars.dim:
            self.gate = Gate(
                scalars,
                _acts(scalars),
                gates,
                _acts(gates),
                nonscalars,
            )
        else:
            self.gate = None
        produced = self.gate.irreps_out if self.gate is not None else scalars.simplify()
        expected = (scalars + nonscalars).simplify()
        if self.gate is not None and self.gate.irreps_in != self.tensor_conv.irreps_out:
            raise RuntimeError(
                f"gate reads {self.gate.irreps_in}, "
                f"convolution writes {self.tensor_conv.irreps_out}"
            )
        if produced != expected:
            raise RuntimeError(
                f"gate produced {produced}, expected the hidden irreps {expected}"
            )
        self.irreps_tensor = produced
        self.scalar_conv = EqnormConv(
            self.irreps_tensor,
            self.irreps_sh,
            self.irreps_scalar,
            num_rbf=num_rbf,
            invariant_layers=invariant_layers,
            invariant_neurons=invariant_neurons,
            normalization=normalization,
            avg_num_neighbors=avg_num_neighbors,
        )

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        radial_basis: Tensor,
        edge_sh: Tensor,
        envelope: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        hidden = self.tensor_conv(
            features, idx_i, idx_j, radial_basis, edge_sh, envelope
        )
        if self.gate is not None:
            hidden = self.gate(hidden)
        scalars = self.scalar_conv(
            hidden, idx_i, idx_j, radial_basis, edge_sh, envelope
        )
        return hidden, scalars


def _gated_targets(
    irreps_in: o3.Irreps,
    irreps_hidden: o3.Irreps,
    irreps_sh: o3.Irreps,
) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    scalars = []
    nonscalars = []
    for mul, ir in irreps_hidden:
        if not tp_path_exists(irreps_in, irreps_sh, ir):
            continue
        if ir.l == 0:
            scalars.append((mul, ir))
        else:
            nonscalars.append((mul, ir))
    irreps_scalars = o3.Irreps(scalars)
    irreps_nonscalars = o3.Irreps(nonscalars)
    if irreps_scalars.dim == 0:
        raise ValueError(
            f"no scalar path from {irreps_in} x {irreps_sh} into {irreps_hidden}"
        )
    gate_ir = o3.Irrep("0e" if tp_path_exists(irreps_in, irreps_sh, "0e") else "0o")
    n_gates = irreps_nonscalars.num_irreps
    irreps_gates = o3.Irreps([(n_gates, gate_ir)]) if n_gates else o3.Irreps("")
    return irreps_scalars, irreps_gates, irreps_nonscalars
