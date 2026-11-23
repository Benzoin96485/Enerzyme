"""Per-layer BOTNet energy head.

Intermediate layers use one even-scalar linear map. The last layer uses that
linear map when the stack has a single interaction, and a scalar MLP
(linear, activation, linear) when a later interaction exists. That is the
``NonLinearBodyOrderedModel`` readout in ``gncs/botnet``. Site energies are
summed. :class:`~enerzyme.models.mace.readout.MACEEnergyReadout` is a gated
MLP on every layer and is not this head.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import Activation
from torch import Tensor, nn

from ..layers._base_layer import BaseFFLayer
from ..mace.blocks import feature_spec, stack_node_features


def resolve_gate(name: str) -> Callable[[Tensor], Tensor]:
    key = str(name).lower()
    if key in {"silu", "swish"}:
        return F.silu
    if key == "tanh":
        return torch.tanh
    if key in {"none", "identity", "linear"}:
        return lambda features: features
    raise ValueError(
        f"gate must be 'silu', 'tanh', or 'none', got {name!r}"
    )


def _core_attr(built_layers, name: str, default):
    for layer in reversed(list(built_layers or [])):
        if hasattr(layer, name):
            value = getattr(layer, name)
            if value is not None:
                return value
    return default


class _ScalarLinear(nn.Module):
    def __init__(self, irreps_in: o3.Irreps) -> None:
        super().__init__()
        self.linear = o3.Linear(irreps_in, o3.Irreps("0e"))

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features).squeeze(-1)


class _ScalarMLP(nn.Module):
    """``Linear → activation → Linear`` onto one even scalar.

    ``mlp_irreps`` has to be scalars. ``e3nn`` applies one activation to each
    irrep entry, which is one SiLU on ``16x0e``.
    """

    def __init__(self, irreps_in: o3.Irreps, mlp_irreps: o3.Irreps, gate: Callable) -> None:
        super().__init__()
        self.mlp_irreps = o3.Irreps(mlp_irreps).simplify()
        if self.mlp_irreps.dim == 0:
            raise ValueError("MLP_irreps is empty")
        if any(ir.l != 0 for _, ir in self.mlp_irreps):
            raise ValueError(
                "BOTNet's last readout is a scalar MLP. "
                f"MLP_irreps must have l = 0, got {self.mlp_irreps}."
            )
        self.linear_1 = o3.Linear(irreps_in, self.mlp_irreps)
        self.activation = Activation(
            self.mlp_irreps, acts=[gate] * len(self.mlp_irreps)
        )
        self.linear_2 = o3.Linear(self.mlp_irreps, o3.Irreps("0e"))

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.activation(self.linear_1(features))
        return self.linear_2(hidden).squeeze(-1)


class BOTNetEnergyReadout(BaseFFLayer):
    def __init__(
        self,
        built_layers=None,
        feature_irreps: Optional[str] = None,
        num_output_blocks: Optional[int] = None,
        MLP_irreps: Optional[str] = None,
        gate: Optional[str] = None,
        last_readout: Optional[str] = None,
    ) -> None:
        super().__init__(output_fields={"Ea"})
        self.feature_irreps, self.num_output_blocks = feature_spec(
            built_layers, feature_irreps, num_output_blocks
        )
        if MLP_irreps is None:
            MLP_irreps = _core_attr(built_layers, "readout_mlp_irreps", "16x0e")
        if gate is None:
            gate = _core_attr(built_layers, "readout_gate", "silu")
        if last_readout is None:
            last_readout = _core_attr(built_layers, "last_readout", "nonlinear")
        if last_readout not in {"linear", "nonlinear"}:
            raise ValueError(
                f"last_readout must be 'linear' or 'nonlinear', got {last_readout!r}"
            )
        self.last_readout = last_readout
        self.MLP_irreps = str(o3.Irreps(MLP_irreps))
        self.gate_name = str(gate)
        activation = resolve_gate(self.gate_name)
        hidden = o3.Irreps(self.MLP_irreps)
        nonlinear_at = self.num_output_blocks - 1 if last_readout == "nonlinear" else None
        heads = []
        for index in range(self.num_output_blocks):
            if index == nonlinear_at:
                heads.append(_ScalarMLP(self.feature_irreps, hidden, activation))
            else:
                heads.append(_ScalarLinear(self.feature_irreps))
        self.heads = nn.ModuleList(heads)

    def get_output(self, atom_feature: Optional[Tensor] = None) -> Dict[str, Tensor]:
        layers = stack_node_features(atom_feature, self.num_output_blocks)
        energy = layers.new_zeros(layers.shape[0])
        for index, head in enumerate(self.heads):
            energy = energy + head(layers[:, :, index])
        return {"Ea": energy}
