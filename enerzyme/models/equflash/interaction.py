"""EquFlash interaction: uvu tensor product, equivariant gate, optional FFN.

The convolution is the NequIP / SevenNet message: a radial network produces
weights for a Clebsch–Gordan product of neighbor features with spherical
harmonics, and the messages are summed onto the receiver. Instructions come
from the shared :func:`enerzyme.models.e3nn_nn.tp_out_irreps_with_instructions`.
The gate is :class:`e3nn.nn.Gate` (second-moment-normalized activations), which
is the nonlinearity used by the reference EquFlash stack.

Cuequivariance and FlashTP are optional accelerators in SamsungDS/GGNN. This
module implements the same uvu product with e3nn so the Core does not take
those dependencies.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from torch import Tensor, nn

from ..activation import resolve_scalar_activation
from ..e3nn_nn import tp_out_irreps_with_instructions
from ..functional import segment_sum_coo


def resolve_activation(name: str):
    """EquFlash name for :func:`enerzyme.models.activation.resolve_scalar_activation`."""
    return resolve_scalar_activation(name)


def build_equivariant_gate(
    irreps_out: o3.Irreps,
    scalar_even: str = "silu",
    scalar_odd: str = "tanh",
    gate_even: str = "silu",
    gate_odd: str = "tanh",
) -> Gate:
    """e3nn gate whose output irreps match ``irreps_out`` for even/odd acts.

    Even scalar channels use ``scalar_even``. Non-scalar irreps are multiplied
    by extra scalar gates. Those gates are even parity when the block already
    has a ``0e`` channel, matching the EquFlash gate.
    """
    irreps_out = o3.Irreps(irreps_out)
    scalars: List[tuple] = []
    gated: List[tuple] = []
    for mul, ir in irreps_out:
        if ir.l == 0:
            scalars.append((mul, ir))
        else:
            gated.append((mul, ir))
    irreps_scalars = o3.Irreps(scalars)
    irreps_gated = o3.Irreps(gated)
    even_scalar = resolve_activation(scalar_even)
    odd_scalar = resolve_activation(scalar_odd)
    act_scalars = [
        even_scalar if ir.p == 1 else odd_scalar for _, ir in irreps_scalars
    ]
    gate_parity = 1 if o3.Irrep("0e") in irreps_scalars else -1
    irreps_gates = o3.Irreps([(mul, (0, gate_parity)) for mul, _ in irreps_gated])
    even_gate = resolve_activation(gate_even)
    odd_gate = resolve_activation(gate_odd)
    act_gates = [even_gate if ir.p == 1 else odd_gate for _, ir in irreps_gates]
    return Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)


class EquFlashFeedForward(nn.Module):
    """Linear, gated nonlinearity, linear, plus a residual on the same irreps."""

    def __init__(
        self,
        irreps: o3.Irreps,
        scalar_even: str,
        scalar_odd: str,
        gate_even: str,
        gate_odd: str,
        bias: bool,
    ) -> None:
        super().__init__()
        irreps = o3.Irreps(irreps)
        self.gate = build_equivariant_gate(
            irreps, scalar_even, scalar_odd, gate_even, gate_odd
        )
        self.linear_1 = o3.Linear(irreps, self.gate.irreps_in, biases=bias)
        self.linear_2 = o3.Linear(self.gate.irreps_out, irreps, biases=bias)

    def forward(self, features: Tensor) -> Tensor:
        updated = self.linear_2(self.gate(self.linear_1(features)))
        return features + updated


class EquFlashConvolution(nn.Module):
    """Weighted uvu tensor product of node features with spherical harmonics."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        radial_hidden: Sequence[int],
        num_rbf: int,
        act_radial: str,
        denominator: float,
        train_denominator: bool,
    ) -> None:
        super().__init__()
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            o3.Irreps(irreps_in), o3.Irreps(irreps_sh), o3.Irreps(irreps_out)
        )
        self.tp = o3.TensorProduct(
            irreps_in,
            irreps_sh,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.irreps_out = self.tp.irreps_out
        hidden = [int(width) for width in radial_hidden]
        if any(width < 1 for width in hidden):
            raise ValueError(f"radial hidden widths must be positive, got {hidden}")
        self.weight_nn = FullyConnectedNet(
            [num_rbf, *hidden, self.tp.weight_numel],
            act=resolve_activation(act_radial),
        )
        self.denominator = nn.Parameter(
            torch.tensor(float(denominator)), requires_grad=train_denominator
        )

    def forward(
        self,
        features: Tensor,
        sph: Tensor,
        rbf: Tensor,
        idx_sender: Tensor,
        idx_receiver: Tensor,
    ) -> Tensor:
        n_atoms = features.shape[0]
        if idx_sender.numel() == 0:
            return features.new_zeros(n_atoms, self.irreps_out.dim)
        weights = self.weight_nn(rbf)
        messages = self.tp(features[idx_sender], sph, weights)
        aggregated = segment_sum_coo(messages, idx_receiver, dim_size=n_atoms)
        return aggregated / self.denominator


class EquFlashInteraction(nn.Module):
    """One EquFlash block: self-connection, convolution, gate, optional FFN."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        radial_hidden: Sequence[int],
        num_rbf: int,
        act_radial: str,
        scalar_even: str,
        scalar_odd: str,
        gate_even: str,
        gate_odd: str,
        denominator: float,
        train_denominator: bool,
        bias: bool,
        feed_forward: bool,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        requested = o3.Irreps(irreps_out)
        self.gate = build_equivariant_gate(
            requested, scalar_even, scalar_odd, gate_even, gate_odd
        )
        self.irreps_out = self.gate.irreps_out
        self.skip = o3.Linear(self.irreps_in, self.gate.irreps_in, biases=bias)
        self.pre_linear = o3.Linear(self.irreps_in, self.irreps_in, biases=bias)
        self.conv = EquFlashConvolution(
            self.irreps_in,
            irreps_sh,
            requested,
            radial_hidden,
            num_rbf,
            act_radial,
            denominator,
            train_denominator,
        )
        self.post_linear = o3.Linear(
            self.conv.irreps_out, self.gate.irreps_in, biases=bias
        )
        self.feed_forward = (
            EquFlashFeedForward(
                self.irreps_out, scalar_even, scalar_odd, gate_even, gate_odd, bias
            )
            if feed_forward
            else None
        )

    def forward(
        self,
        features: Tensor,
        sph: Tensor,
        rbf: Tensor,
        idx_sender: Tensor,
        idx_receiver: Tensor,
    ) -> Tensor:
        residual = self.skip(features)
        hidden = self.pre_linear(features)
        messages = self.conv(hidden, sph, rbf, idx_sender, idx_receiver)
        activated = self.gate(self.post_linear(messages) + residual)
        if self.feed_forward is not None:
            activated = self.feed_forward(activated)
        return activated
