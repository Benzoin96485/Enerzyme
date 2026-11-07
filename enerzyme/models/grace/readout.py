"""Energy heads for GRACE densities. Both sit outside the Core.

The foundation head (Lysogorskiy et al., 2025, SI A) is the first density plus
an MLP of the remaining densities, plus a per-element shift. The
Finnis–Sinclair student applies one power to each density,

``E_i = Σ_k c_k ρ_{ik}^{m_k} + shift(Z_i)``,

with the published powers ``m = 1, 1/2, 2, 3/4``. Even integer powers use
``ρ^m``; other non-integer powers use ``sign(ρ) |ρ|^m`` so the head is defined
for negative densities. Pair repulsion such as ZBL stays a separate layer.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential, SiLU, Tanh

from ..layers._base_layer import BaseFFLayer


def _power(values: Tensor, exponent: float, epsilon: float) -> Tensor:
    if math.isclose(exponent, 1.0):
        return values
    rounded = int(round(exponent))
    if math.isclose(exponent, rounded):
        return values.pow(rounded)
    return values.sign() * values.abs().clamp_min(epsilon).pow(exponent)


class GRACEEnergyReadout(BaseFFLayer):
    """Per-atom energy from GRACE densities."""

    def __init__(
        self,
        built_layers: List[Module],
        max_Za: int,
        hidden: Optional[int] = None,
        activation: Optional[str] = None,
        fs_terms: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        super().__init__(
            input_fields={"atom_feature", "Za"},
            output_fields={"Ea"},
        )
        core = _find_grace_core(built_layers)
        self.max_Za = int(max_Za)
        self.n_densities = int(core.n_densities)
        self.head = str(core.readout_head)
        self.shift = Parameter(torch.zeros(self.max_Za + 1))
        hidden_width = int(core.readout_hidden if hidden is None else hidden)
        activation_name = (
            str(core.readout_activation) if activation is None else str(activation)
        )
        self.mlp: Optional[Module] = None
        self.coeff: Optional[Parameter] = None
        self.register_buffer("powers", torch.empty(0), persistent=False)
        self.epsilon = 1e-8
        if self.head == "linear_mlp":
            nonlinear = self.n_densities - 1
            if nonlinear > 0:
                if hidden_width < 1:
                    raise ValueError(
                        f"linear_mlp hidden width must be >= 1, got {hidden_width}"
                    )
                act: Module = SiLU() if activation_name == "silu" else Tanh()
                if activation_name not in {"silu", "tanh"}:
                    raise ValueError(
                        f"Unknown GRACE readout activation {activation_name!r}"
                    )
                self.mlp = Sequential(
                    Linear(nonlinear, hidden_width),
                    act,
                    Linear(hidden_width, 1),
                )
        elif self.head == "finnis_sinclair":
            terms = tuple(core.fs_terms if fs_terms is None else fs_terms)
            if len(terms) != self.n_densities:
                raise ValueError(
                    f"FS head has {len(terms)} terms but the Core emitted "
                    f"{self.n_densities} densities"
                )
            coefficients = [float(term[0]) for term in terms]
            powers = [float(term[1]) for term in terms]
            self.coeff = Parameter(torch.tensor(coefficients))
            self.register_buffer("powers", torch.tensor(powers), persistent=False)
        else:
            raise ValueError(f"Unknown GRACE readout head {self.head!r}")

    def get_Ea(self, atom_feature: Tensor, Za: Tensor) -> Tensor:
        if atom_feature.ndim != 2 or atom_feature.shape[-1] != self.n_densities:
            raise ValueError(
                f"GRACE densities must be [N, {self.n_densities}], "
                f"got {tuple(atom_feature.shape)}"
            )
        if self.head == "linear_mlp":
            energy = atom_feature[:, 0]
            if self.mlp is not None:
                energy = energy + self.mlp(atom_feature[:, 1:]).squeeze(-1)
        else:
            terms = [
                self.coeff[index] * _power(atom_feature[:, index], float(power), self.epsilon)
                for index, power in enumerate(self.powers.tolist())
            ]
            energy = torch.stack(terms, dim=-1).sum(dim=-1)
        return energy + self.shift[Za]


def _find_grace_core(built_layers: List[Module]) -> Module:
    for layer in reversed(built_layers):
        if getattr(layer, "is_grace_core", False):
            return layer
    raise TypeError("GRACEEnergyReadout must follow a GRACECore in the layer list")
