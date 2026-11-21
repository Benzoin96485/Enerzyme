"""Equivariant blocks shared by the MACE energy head and MACE-POLAR-1.

The scalar MLP is Linear → SiLU, with no LayerNorm on the hidden widths.
The channel dot and the scalar modulation are e3nn ``uvu`` tensor products,
so every irrep of the node features has to carry the same multiplicity.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from e3nn import o3
from e3nn.nn import Gate
from torch import Tensor, nn
import torch.nn.functional as F


def feature_spec(built_layers, feature_irreps: Optional[str] = None, num_blocks: Optional[int] = None):
    """Irreps and layer count advertised by the nearest preceding Core."""
    if feature_irreps is None:
        for layer in reversed(list(built_layers or [])):
            advertised = getattr(layer, "feature_irreps", None)
            if not advertised:
                continue
            feature_irreps = advertised
            if num_blocks is None:
                num_blocks = getattr(layer, "num_output_blocks", None)
            break
    if not feature_irreps:
        raise ValueError(
            "feature_irreps is missing. Place this layer after a Core that sets feature_irreps."
        )
    irreps = o3.Irreps(feature_irreps)
    if num_blocks is None:
        num_blocks = 1
    return irreps, int(num_blocks)


def uniform_multiplicity(irreps: o3.Irreps) -> int:
    irreps = o3.Irreps(irreps).simplify()
    if irreps.dim == 0:
        raise ValueError("node irreps are empty")
    counts = {mul for mul, _ in irreps}
    if len(counts) != 1:
        raise ValueError(
            "The polarizable field update needs the same multiplicity on every "
            f"node irrep, got {irreps}."
        )
    return next(iter(counts))


class SiLUMLP(nn.Module):
    """``Linear → SiLU`` stack. The last linear has no activation."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(f"MLP needs at least two widths, got {list(channels)}")
        layers: List[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(channels[:-1], channels[1:])):
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
            if index < len(channels) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class GatedIrrepsMLP(nn.Module):
    """Two gated linear maps, then a linear map to ``irreps_out``.

    Even scalars use SiLU. Gate scalars use sigmoid. With a purely scalar
    hidden irrep the gate is SiLU on that hidden irrep. ``irreps_out`` is
    kept in the order it is given: do not sort a spin-resolved target, or the
    monopole and dipole channels change places.
    """

    def __init__(self, irreps_in, hidden_irreps, irreps_out) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.irreps_out = o3.Irreps(irreps_out)
        if self.hidden_irreps.dim == 0:
            raise ValueError("hidden irreps are empty")
        scalars = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l == 0])
        gated = o3.Irreps([(mul, ir) for mul, ir in self.hidden_irreps if ir.l > 0])
        if gated.dim == 0:
            self.gate = None
            nonlin = self.hidden_irreps
            activated = self.hidden_irreps
        else:
            gates = o3.Irreps([(mul, "0e") for mul, _ in gated])
            self.gate = Gate(
                irreps_scalars=scalars,
                act_scalars=[F.silu for _, _ir in scalars],
                irreps_gates=gates,
                act_gates=[F.sigmoid] * len(gates),
                irreps_gated=gated,
            )
            nonlin = self.gate.irreps_in
            activated = self.gate.irreps_out
        self.linear_1 = o3.Linear(self.irreps_in, nonlin)
        self.linear_mid = o3.Linear(activated, nonlin)
        self.linear_2 = o3.Linear(activated, self.irreps_out)

    def _activate(self, features: Tensor) -> Tensor:
        if self.gate is None:
            return F.silu(features)
        return self.gate(features)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self._activate(self.linear_1(features))
        hidden = self._activate(self.linear_mid(hidden))
        return self.linear_2(hidden)


class InvariantChannelDot(nn.Module):
    """Per-channel invariant ``l × l → 0`` of two copies of the same irreps."""

    def __init__(self, irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps).simplify()
        self.n_channels = uniform_multiplicity(self.irreps)
        instructions = [
            (index, index, 0, "uvu", True) for index in range(len(self.irreps))
        ]
        self.tp = o3.TensorProduct(
            self.irreps,
            self.irreps,
            o3.Irreps(f"{self.n_channels}x0e"),
            instructions=instructions,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.tp(left, right)


class ScalarModulation(nn.Module):
    """Per-channel ``l × 0e → l``. ``scalars`` has one channel per node irrep copy."""

    def __init__(self, irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps).simplify()
        n_channels = uniform_multiplicity(self.irreps)
        scalar_irreps = o3.Irreps(f"{n_channels}x0e")
        instructions = [
            (index, 0, index, "uvu", True) for index in range(len(self.irreps))
        ]
        self.tp = o3.TensorProduct(
            self.irreps,
            scalar_irreps,
            self.irreps,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, features: Tensor, scalars: Tensor) -> Tensor:
        return self.tp(features, scalars)


def stack_node_features(atom_feature: Tensor, n_layers: int) -> Tensor:
    """``[N, irreps, T]`` or a single ``[N, irreps]`` layer."""
    if atom_feature is None:
        raise ValueError(
            "atom_feature is missing. The Core has to run with output_mode 'layers' "
            "(or 'feature' with keep_last_layer_irreps) before this head."
        )
    if atom_feature.ndim == 2:
        if n_layers != 1:
            raise ValueError(
                f"Got a single feature tensor but this head was built for {n_layers} layers. "
                "Use output_mode: layers on the Core."
            )
        return atom_feature.unsqueeze(-1)
    if atom_feature.ndim != 3:
        raise ValueError(
            f"atom_feature must be [N, irreps] or [N, irreps, T], got {tuple(atom_feature.shape)}"
        )
    if atom_feature.shape[-1] != n_layers:
        raise ValueError(
            f"atom_feature has {atom_feature.shape[-1]} layers, expected {n_layers}."
        )
    return atom_feature
