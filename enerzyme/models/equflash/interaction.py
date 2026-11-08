"""EquFlashV2 blocks on flat e3nn irreps.

Channelwise convolution, bilinear / scalar gates, and the metaformer
interaction and feed-forward blocks. Radial weights use the shared
``RadialMLP`` (SiLU, no layer norm). Tensor products are e3nn ``uvu`` /
``uuu`` maps, not cuequivariance kernels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..blocks.radial_mlp import RadialMLP
from ..e3nn_nn.rms_norm import EquivariantRMSNormSH
from .irreps import scale_irreps


def _linear(irreps_in, irreps_out) -> o3.Linear:
    return o3.Linear(o3.Irreps(irreps_in), o3.Irreps(irreps_out), biases=False)


class IrrepsEmbed(nn.Module):
    """Copy matching ``(mul, ir)`` blocks and zero-fill irreps that are new.

    Extra input irreps are dropped. This is the identity skip when a layer
    grows from scalars to ``l > 0`` or shrinks back to scalars.
    """

    def __init__(self, irreps_in, irreps_out) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in).simplify()
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        available: Dict[o3.Irrep, List] = {}
        for sl, (mul, ir) in zip(self.irreps_in.slices(), self.irreps_in):
            available.setdefault(ir, []).append((mul, sl))
        pairs = []
        for sl, (mul, ir) in zip(self.irreps_out.slices(), self.irreps_out):
            choices = available.get(ir, [])
            match = next((i for i, (m, _) in enumerate(choices) if m == mul), None)
            if match is None:
                if choices:
                    raise ValueError(
                        f"Cannot embed {mul}x{ir} into {self.irreps_out}; "
                        f"input multiplicities are {[m for m, _ in choices]}"
                    )
                continue
            _, src = choices.pop(match)
            pairs.append((src.start, src.stop, sl.start, sl.stop))
        self._pairs = tuple(pairs)

    def forward(self, features: Tensor) -> Tensor:
        if not self._pairs and self.irreps_in == self.irreps_out:
            return features
        out = features.new_zeros(features.shape[0], self.irreps_out.dim)
        for src_start, src_stop, dst_start, dst_stop in self._pairs:
            out[:, dst_start:dst_stop] = features[:, src_start:src_stop]
        return out


def _skip(kind: str, irreps_in, irreps_out) -> nn.Module:
    if kind == "identity":
        return IrrepsEmbed(irreps_in, irreps_out)
    if kind == "linear":
        return _linear(irreps_in, irreps_out)
    raise ValueError(f"Unknown skip type {kind!r}; expected 'identity' or 'linear'")


def _uvu_instructions(irreps_x, irreps_filter, irreps_out):
    irreps_x = o3.Irreps(irreps_x)
    irreps_filter = o3.Irreps(irreps_filter)
    irreps_out = o3.Irreps(irreps_out)
    raw = []
    for i, (mul_x, ir_x) in enumerate(irreps_x):
        for j, (_, ir_f) in enumerate(irreps_filter):
            for produced in ir_x * ir_f:
                for k, (mul_o, ir_o) in enumerate(irreps_out):
                    if ir_o == produced and mul_o == mul_x:
                        raw.append((i, j, k))
                        break
    if not raw:
        raise ValueError(
            f"No channelwise paths {irreps_x} x {irreps_filter} -> {irreps_out}"
        )
    used = sorted({k for _, _, k in raw})
    remap = {old: new for new, old in enumerate(used)}
    instructions = [(i, j, remap[k], "uvu", True) for i, j, k in raw]
    actual = o3.Irreps([irreps_out[k] for k in used])
    return instructions, actual


class ChannelwiseConv(nn.Module):
    """Edge convolution ``sum_j Y(r_ij) \\otimes_channel h_j`` with radial weights.

    ``conv_type`` ``full`` and ``efficient`` both use this e3nn ``uvu`` product.
    The cuequivariance fused kernels are not required. Messages are summed onto
    receivers ``idx_i`` from senders ``idx_j`` and divided by ``denominator``.
    """

    def __init__(
        self,
        irreps_x,
        irreps_filter,
        irreps_out,
        num_rbf: int,
        radial_hidden: Sequence[int],
        denominator: float = 1.0,
        train_denominator: bool = False,
    ) -> None:
        super().__init__()
        instructions, actual = _uvu_instructions(irreps_x, irreps_filter, irreps_out)
        self.tp = o3.TensorProduct(
            o3.Irreps(irreps_x),
            o3.Irreps(irreps_filter),
            actual,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.irreps_in = o3.Irreps(irreps_x)
        self.irreps_out = actual
        hidden = [int(width) for width in radial_hidden]
        if any(width < 1 for width in hidden):
            raise ValueError(f"radial_hidden must be positive, got {radial_hidden}")
        self.radial = RadialMLP(
            [int(num_rbf), *hidden, self.tp.weight_numel],
            use_layer_norm=False,
            bias=True,
        )
        self.denominator = nn.Parameter(
            torch.tensor(float(denominator)),
            requires_grad=bool(train_denominator),
        )

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        edge_sh: Tensor,
    ) -> Tensor:
        if idx_i.numel() == 0:
            return features.new_zeros(features.shape[0], self.irreps_out.dim)
        weight = self.radial(rbf)
        message = self.tp(features[idx_j], edge_sh, weight)
        aggregated = scatter_sum(message, idx_i, dim=0, dim_size=features.shape[0])
        return aggregated / self.denominator


class ScalarGate(nn.Module):
    """SiLU on even scalars, tanh on odd scalars, SiLU gates on ``l > 0``."""

    def __init__(self, irreps) -> None:
        super().__init__()
        from ..e3nn_nn.activation import Gate

        irreps = o3.Irreps(irreps).simplify()
        odd, even, gated = [], [], []
        for mul, ir in irreps:
            if ir.l == 0 and ir.p == -1:
                odd.append((mul, ir))
            elif ir.l == 0:
                even.append((mul, ir))
            else:
                gated.append((mul, ir))
        scalars = o3.Irreps(odd + even).simplify()
        irreps_gated = o3.Irreps(gated).simplify()
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()
        scalar_acts = [torch.tanh if ir.p == -1 else F.silu for _, ir in scalars]
        gate_acts = [F.silu] if irreps_gates.dim else []
        self.gate = Gate(
            scalars,
            scalar_acts,
            irreps_gates,
            gate_acts,
            irreps_gated,
        )
        self.irreps_in = self.gate.irreps_in
        self.irreps_out = self.gate.irreps_out

    def forward(self, features: Tensor) -> Tensor:
        return self.gate(features)


def _uuu_instructions(irreps_a, irreps_b, irreps_out, *, min_dim: bool = False):
    irreps_a = o3.Irreps(irreps_a)
    irreps_b = o3.Irreps(irreps_b)
    irreps_out = o3.Irreps(irreps_out)
    raw = []
    for i, (mul_a, ir_a) in enumerate(irreps_a):
        for j, (mul_b, ir_b) in enumerate(irreps_b):
            if mul_a != mul_b:
                continue
            if min_dim and ir_a.dim < ir_b.dim:
                continue
            for produced in ir_a * ir_b:
                for k, (mul_o, ir_o) in enumerate(irreps_out):
                    if ir_o == produced and mul_o == mul_a:
                        raw.append((i, j, k))
                        break
    if not raw:
        return [], o3.Irreps("")
    used = sorted({k for _, _, k in raw})
    remap = {old: new for new, old in enumerate(used)}
    instructions = [(i, j, remap[k], "uuu", True) for i, j, k in raw]
    actual = o3.Irreps([irreps_out[k] for k in used])
    return instructions, actual


class BiLinearGate(nn.Module):
    """Channelwise bilinear gate.

    Leading ``0e`` channels are activated and passed through. A SiLU scalar
    produced by the preceding linear weights a ``uuu`` product between two
    copies of the higher-order features. The second copy's scalar slots are
    ones, so the product contains both a gated linear term and a bilinear term.
    """

    def __init__(self, irreps) -> None:
        super().__init__()
        irreps = o3.Irreps(irreps).simplify()
        for _, ir in irreps:
            if ir.l == 0 and ir.p == -1:
                raise NotImplementedError("BiLinearGate does not support 0o features")
        if len(irreps) == 0 or irreps[0].ir != o3.Irrep("0e"):
            raise ValueError(f"BiLinearGate expects a leading 0e block, got {irreps}")
        self.scalar = o3.Irreps([irreps[0]])
        self.higher = o3.Irreps(list(irreps)[1:]).simplify()
        self.dim_act = self.scalar.dim
        instructions, actual = _uuu_instructions(
            self.higher, irreps, self.higher, min_dim=True
        )
        if instructions:
            self.tp = o3.TensorProduct(
                self.higher,
                irreps,
                actual,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )
            self.produced = actual
        else:
            self.tp = None
            self.produced = o3.Irreps("")
        self.weight_numel = 0 if self.tp is None else self.tp.weight_numel
        doubled = (self.higher + self.higher).simplify() if self.higher.dim else o3.Irreps("")
        gate = (
            o3.Irreps(f"{self.weight_numel}x0e")
            if self.weight_numel
            else o3.Irreps("")
        )
        self.irreps_in = (gate + self.scalar + doubled).simplify()
        self.irreps_out = (self.scalar + self.produced).simplify()
        self._higher_layout = tuple((mul, ir.dim) for mul, ir in self.higher)

    def forward(self, features: Tensor) -> Tensor:
        gate_dim = self.weight_numel
        act = features[:, gate_dim : gate_dim + self.dim_act]
        act = F.silu(act)
        if self.tp is None:
            return act
        weights = F.silu(features[:, :gate_dim])
        rest = features[:, gate_dim + self.dim_act :]
        first_parts = []
        second_parts = []
        offset = 0
        batch = features.shape[0]
        for mul, dim in self._higher_layout:
            width = 2 * mul * dim
            block = rest[:, offset : offset + width].reshape(batch, 2 * mul, dim)
            first_parts.append(block[:, :mul].reshape(batch, mul * dim))
            second_parts.append(block[:, mul:].reshape(batch, mul * dim))
            offset += width
        left = torch.cat(first_parts, dim=-1)
        ones = features.new_ones(batch, self.dim_act)
        right = torch.cat([ones, torch.cat(second_parts, dim=-1)], dim=-1)
        mixed = self.tp(left, right, weights)
        return torch.cat([act, mixed], dim=-1)


class SymmetricContraction(nn.Module):
    """Species-agnostic polynomial of the node features, degrees 1..``degree``.

    Degree 1 is an equivariant linear map. Higher degrees are ``uuu`` products
    of the running contraction with the original features. This is the e3nn
    stand-in for the cuequivariance symmetric contraction (one species).
    """

    def __init__(self, irreps, contraction_degree: int = 3) -> None:
        super().__init__()
        irreps = o3.Irreps(irreps).simplify()
        if contraction_degree < 1:
            raise ValueError(f"contraction_degree must be >= 1, got {contraction_degree}")
        self.irreps_in = irreps
        self.irreps_out = irreps
        self.lin = _linear(irreps, irreps)
        self.products = nn.ModuleList()
        instructions, actual = _uuu_instructions(irreps, irreps, irreps)
        if actual != irreps and contraction_degree > 1 and instructions:
            # Keep the contracted irreps equal to the input so the next degree
            # can reuse the same instruction list. Missing outputs stay zero
            # only when the product cannot make them; require a full cover.
            raise ValueError(
                f"Symmetric contraction of {irreps} does not cover every output"
            )
        for _ in range(int(contraction_degree) - 1):
            if not instructions:
                break
            self.products.append(
                o3.TensorProduct(
                    irreps,
                    irreps,
                    irreps,
                    instructions,
                    internal_weights=True,
                    shared_weights=True,
                )
            )

    def forward(self, features: Tensor) -> Tensor:
        out = self.lin(features)
        running = features
        for product in self.products:
            running = product(running, features)
            out = out + running
        return out


def make_nonlinear(kind: str, irreps, scale: float = 1.0, kwargs: Optional[dict] = None):
    kwargs = dict(kwargs or {})
    target = scale_irreps(irreps, scale)
    if kind == "gate":
        return ScalarGate(target)
    if kind == "bilinear_gate":
        return BiLinearGate(target)
    if kind == "symmetric_contraction":
        return SymmetricContraction(
            target, contraction_degree=int(kwargs.get("contraction_degree", 3))
        )
    raise ValueError(
        f"Unknown nonlinear_type {kind!r}; expected gate, bilinear_gate, "
        "or symmetric_contraction"
    )


def _project(kind: str, irreps_in, irreps_out) -> nn.Module:
    if kind == "identity":
        if o3.Irreps(irreps_in).simplify() != o3.Irreps(irreps_out).simplify():
            raise ValueError(
                f"identity projection needs {irreps_in} == {irreps_out}"
            )
        return nn.Identity()
    if kind == "linear":
        return _linear(irreps_in, irreps_out)
    raise ValueError(f"Unknown projection {kind!r}; expected 'identity' or 'linear'")


class InteractionBlock(nn.Module):
    """Norm, channelwise convolution, nonlinearity, residual."""

    def __init__(
        self,
        irreps_x,
        irreps_filter,
        irreps_out,
        *,
        num_rbf: int,
        radial_hidden: Sequence[int],
        denominator: float,
        train_denominator: bool,
        sc_type: str,
        nonlinear_type: str,
        nonlinear_scale: float,
        nonlinear_kwargs: Optional[dict],
        after_nonlinear: str,
    ) -> None:
        super().__init__()
        self.norm = EquivariantRMSNormSH(irreps_x)
        self.skip = _skip(sc_type, irreps_x, irreps_out)
        self.linear_in = _linear(irreps_x, irreps_x)
        self.conv = ChannelwiseConv(
            irreps_x,
            irreps_filter,
            irreps_out,
            num_rbf=num_rbf,
            radial_hidden=radial_hidden,
            denominator=denominator,
            train_denominator=train_denominator,
        )
        self.nonlinear = make_nonlinear(
            nonlinear_type, irreps_out, nonlinear_scale, nonlinear_kwargs
        )
        self.linear_mid = _linear(self.conv.irreps_out, self.nonlinear.irreps_in)
        self.linear_out = _project(
            after_nonlinear, self.nonlinear.irreps_out, irreps_out
        )

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        edge_sh: Tensor,
    ) -> Tensor:
        residual = self.skip(features)
        hidden = self.linear_in(self.norm(features))
        hidden = self.conv(hidden, idx_i, idx_j, rbf, edge_sh)
        hidden = self.linear_out(self.nonlinear(self.linear_mid(hidden)))
        return hidden + residual


class NequIPInteractionBlock(nn.Module):
    """Convolution residual into the nonlinearity, without a leading norm."""

    def __init__(
        self,
        irreps_x,
        irreps_filter,
        irreps_out,
        *,
        num_rbf: int,
        radial_hidden: Sequence[int],
        denominator: float,
        train_denominator: bool,
        sc_type: str,
        nonlinear_type: str,
        nonlinear_scale: float,
        nonlinear_kwargs: Optional[dict],
        after_nonlinear: str = "linear",
    ) -> None:
        super().__init__()
        del after_nonlinear
        self.linear_in = _linear(irreps_x, irreps_x)
        self.conv = ChannelwiseConv(
            irreps_x,
            irreps_filter,
            irreps_out,
            num_rbf=num_rbf,
            radial_hidden=radial_hidden,
            denominator=denominator,
            train_denominator=train_denominator,
        )
        self.nonlinear = make_nonlinear(
            nonlinear_type, irreps_out, nonlinear_scale, nonlinear_kwargs
        )
        self.skip = _skip(sc_type, irreps_x, self.nonlinear.irreps_in)
        self.linear_mid = _linear(self.conv.irreps_out, self.nonlinear.irreps_in)
        if self.nonlinear.irreps_out.simplify() == o3.Irreps(irreps_out).simplify():
            self.align: nn.Module = nn.Identity()
        else:
            self.align = _linear(self.nonlinear.irreps_out, irreps_out)

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        edge_sh: Tensor,
    ) -> Tensor:
        residual = self.skip(features)
        hidden = self.linear_in(features)
        hidden = self.conv(hidden, idx_i, idx_j, rbf, edge_sh)
        hidden = self.nonlinear(self.linear_mid(hidden) + residual)
        return self.align(hidden)


class FeedForwardBlock(nn.Module):
    """Norm, nonlinearity, residual. No edges."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        *,
        sc_type: str,
        nonlinear_type: str,
        nonlinear_scale: float,
        nonlinear_kwargs: Optional[dict],
    ) -> None:
        super().__init__()
        self.norm = EquivariantRMSNormSH(irreps_in)
        self.skip = _skip(sc_type, irreps_in, irreps_out)
        self.nonlinear = make_nonlinear(
            nonlinear_type, irreps_out, nonlinear_scale, nonlinear_kwargs
        )
        self.linear_1 = _linear(irreps_in, self.nonlinear.irreps_in)
        self.linear_2 = _linear(self.nonlinear.irreps_out, irreps_out)

    def forward(self, features: Tensor) -> Tensor:
        residual = self.skip(features)
        hidden = self.linear_2(self.nonlinear(self.linear_1(self.norm(features))))
        return residual + hidden


INTERACTION_TYPES = {
    "equflash": InteractionBlock,
    "nequip": NequIPInteractionBlock,
}
