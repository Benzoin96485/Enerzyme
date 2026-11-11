"""NequIP interaction and convolution.

The tensor product, species self-connection, and gate follow
``nequip.nn.InteractionBlock`` / ``ConvNetLayer`` (mir-group/nequip, MIT) and
Batzner et al., Nat. Commun. 2022. Messages are
``TP(h_src, Y(r_src - r_dst))`` with weights from a scalar radial MLP, summed
onto the receiver, then mixed by an equivariant linear map. A self-connection
is a fully connected tensor product with the fixed species embedding.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from torch import Tensor, nn

from ..activation import resolve_scalar_activation
from ..e3nn_nn import tp_out_irreps_with_instructions
from ..so3.radial import RadialMLP


def _parity_key(parity: int) -> str:
    return "e" if int(parity) == 1 else "o"


def _resolve_acts(spec: Optional[Dict[str, str]]) -> Dict[str, str]:
    resolved = {"e": "silu", "o": "tanh"}
    if spec is None:
        return resolved
    unknown = set(spec) - {"e", "o"}
    if unknown:
        raise ValueError(
            f"nonlinearity keys must be 'e' and 'o', got {sorted(unknown)}"
        )
    resolved.update(spec)
    for name in resolved.values():
        resolve_scalar_activation(name)
    return resolved


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    """Whether any Clebsch–Gordan path can produce ``ir_out``."""
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def hidden_irreps(
    num_features: int | Sequence[int],
    l_max: int,
    parity: bool,
) -> o3.Irreps:
    """Feature irreps used by the intermediate NequIP convolutions.

    ``parity=True`` keeps both mirror parities at every ``l``. ``parity=False``
    keeps the natural parity ``p = (-1)^l`` only, which is the current
    foundation-model default.
    """
    if isinstance(num_features, int):
        counts = [int(num_features)] * (int(l_max) + 1)
    else:
        counts = [int(count) for count in num_features]
        if len(counts) != int(l_max) + 1:
            raise ValueError(
                f"num_features must be an int or a list of length l_max+1="
                f"{int(l_max) + 1}, got {list(num_features)}"
            )
    if any(count < 0 for count in counts):
        raise ValueError(f"num_features must be non-negative, got {counts}")
    irreps: List[tuple] = []
    for ell, mul in enumerate(counts):
        if mul == 0:
            continue
        if parity:
            parities = (1, -1)
        else:
            parities = (1,) if ell % 2 == 0 else (-1,)
        for p in parities:
            irreps.append((mul, (ell, p)))
    return o3.Irreps(irreps)


class NequIPInteraction(nn.Module):
    """One message-passing convolution, before the equivariant nonlinearity."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        num_rbf: int,
        radial_mlp_depth: int,
        radial_mlp_width: int,
        avg_num_neighbors: float,
        use_sc: bool,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.use_sc = bool(use_sc)
        if avg_num_neighbors <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
            )
        self.linear_1 = o3.Linear(
            self.irreps_in,
            self.irreps_in,
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_in, self.irreps_edge_attr, self.irreps_out
        )
        if len(instructions) == 0:
            raise ValueError(
                "NequIP convolution has no tensor-product paths from "
                f"{self.irreps_in} x {self.irreps_edge_attr} into {self.irreps_out}"
            )
        self.tp = o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )
        # Adjacent identical irreps are merged by simplify(); the flat layout
        # matches the sorted tensor-product output, as in NequIP's linear_2.
        self.linear_2 = o3.Linear(
            irreps_mid.simplify(),
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )
        if radial_mlp_depth < 0:
            raise ValueError(
                f"radial_mlp_depth must be >= 0, got {radial_mlp_depth}"
            )
        if radial_mlp_depth > 0 and radial_mlp_width < 1:
            raise ValueError(
                f"radial_mlp_width must be >= 1 when depth > 0, got {radial_mlp_width}"
            )
        hidden = [int(radial_mlp_width)] * int(radial_mlp_depth)
        self.radial_mlp = RadialMLP(
            int(num_rbf), self.tp.weight_numel, hidden=hidden, bias=False
        )
        self.sc: Optional[o3.FullyConnectedTensorProduct]
        if self.use_sc:
            self.sc = o3.FullyConnectedTensorProduct(
                self.irreps_in, o3.Irreps(irreps_node_attr), self.irreps_out
            )
        else:
            self.sc = None
        self.register_buffer(
            "neighbor_scale",
            torch.tensor(1.0 / math.sqrt(float(avg_num_neighbors))),
            persistent=False,
        )

    def forward(
        self,
        features: Tensor,
        node_attrs: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_attr: Tensor,
        edge_embed: Tensor,
    ) -> Tensor:
        if self.sc is not None:
            skip = self.sc(features, node_attrs)
        else:
            skip = None
        mixed = self.linear_1(features) * self.neighbor_scale.to(dtype=features.dtype)
        n_atoms = features.shape[0]
        if edge_src.numel() == 0:
            aggregated = features.new_zeros(n_atoms, self.tp.irreps_out.dim)
        else:
            weights = self.radial_mlp(edge_embed)
            messages = self.tp(mixed[edge_src], edge_attr, weights)
            aggregated = features.new_zeros(n_atoms, messages.shape[-1])
            aggregated.index_add_(0, edge_dst, messages)
        out = self.linear_2(aggregated)
        if skip is not None:
            out = out + skip
        return out


class NequIPConv(nn.Module):
    """Convolution plus gate (or norm activation) and an optional residual."""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        feature_irreps_hidden: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        irreps_node_attr: o3.Irreps,
        num_rbf: int,
        radial_mlp_depth: int,
        radial_mlp_width: int,
        avg_num_neighbors: float,
        use_sc: bool,
        resnet: bool,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Optional[Dict[str, str]] = None,
        nonlinearity_gates: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        hidden = o3.Irreps(feature_irreps_hidden)
        edge = o3.Irreps(irreps_edge_attr)
        scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in hidden
                if ir.l == 0 and tp_path_exists(self.irreps_in, edge, ir)
            ]
        )
        gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in hidden
                if ir.l > 0 and tp_path_exists(self.irreps_in, edge, ir)
            ]
        )
        if scalars.dim == 0 and gated.dim == 0:
            raise ValueError(
                "NequIP layer has no reachable features from "
                f"{self.irreps_in} and edge {edge} into {hidden}"
            )
        scalar_acts = _resolve_acts(nonlinearity_scalars)
        gate_acts = _resolve_acts(nonlinearity_gates)
        if nonlinearity_type == "gate":
            if gated.dim > 0:
                gate_ir = (
                    "0e" if tp_path_exists(self.irreps_in, edge, "0e") else "0o"
                )
                irreps_gates = o3.Irreps([(mul, gate_ir) for mul, _ in gated])
            else:
                irreps_gates = o3.Irreps("")
            self.nonlin = Gate(
                irreps_scalars=scalars,
                act_scalars=[
                    resolve_scalar_activation(scalar_acts[_parity_key(ir.p)])
                    for _, ir in scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[
                    resolve_scalar_activation(gate_acts[_parity_key(ir.p)])
                    for _, ir in irreps_gates
                ],
                irreps_gated=gated,
            )
            conv_out = self.nonlin.irreps_in
        elif nonlinearity_type == "norm":
            conv_out = (scalars + gated).simplify()
            self.nonlin = NormActivation(
                irreps_in=conv_out,
                scalar_nonlinearity=resolve_scalar_activation(scalar_acts["e"]),
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )
        else:
            raise ValueError(
                f"nonlinearity_type must be 'gate' or 'norm', got {nonlinearity_type!r}"
            )
        self.interaction = NequIPInteraction(
            irreps_in=self.irreps_in,
            irreps_out=conv_out,
            irreps_edge_attr=edge,
            irreps_node_attr=irreps_node_attr,
            num_rbf=num_rbf,
            radial_mlp_depth=radial_mlp_depth,
            radial_mlp_width=radial_mlp_width,
            avg_num_neighbors=avg_num_neighbors,
            use_sc=use_sc,
        )
        self.irreps_out = o3.Irreps(self.nonlin.irreps_out)
        self.resnet = bool(resnet) and self.irreps_out == self.irreps_in

    def forward(
        self,
        features: Tensor,
        node_attrs: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_attr: Tensor,
        edge_embed: Tensor,
    ) -> Tensor:
        previous = features
        updated = self.interaction(
            features, node_attrs, edge_src, edge_dst, edge_attr, edge_embed
        )
        updated = self.nonlin(updated)
        if self.resnet:
            updated = previous + updated
        return updated
