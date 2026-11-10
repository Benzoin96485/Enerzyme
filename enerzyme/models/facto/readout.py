"""FACTO atomic-energy head (Liu et al., arXiv:2611.03276, Eq. 12).

The head is outside the Core. It reads the concatenated post-block scalar
states, subtracts the isolated-atom state of the same element, and applies
one shared bias-free linear map, an elemental scale, and an elemental shift.
A fixed elemental reference buffer is added on top. Leave it at zero when
the dataset transform already subtracts and restores isolated-atom energies;
that transform plays the role of the paper's ``e_ref``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module, Parameter

from ..layers._base_layer import BaseFFLayer


def concat_block_states(atom_feature: Tensor) -> Tensor:
    """Pack ``(N, D, K)`` post-block states as ``[h^(1); ...; h^(K)]``."""
    if atom_feature.ndim != 3:
        raise ValueError(
            f"FACTO atom features must be (N, D, K), got {tuple(atom_feature.shape)}"
        )
    return atom_feature.transpose(1, 2).reshape(atom_feature.shape[0], -1)


class FACTOEnergyReadout(BaseFFLayer):
    """Per-atom energy from the difference against the isolated-atom state."""

    def __init__(
        self,
        built_layers: List[Module],
        max_Za: int,
        element_reference: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__(
            input_fields={"atom_feature", "isolated_atom_feature", "Za"},
            output_fields={"Ea"},
        )
        core = _find_facto_core(built_layers)
        self.scalar_dim = int(core.scalar_dim)
        self.num_blocks = int(core.num_blocks)
        self.max_Za = int(max_Za)
        self.projection = Linear(
            self.scalar_dim * self.num_blocks, 1, bias=False
        )
        self.shift = Parameter(torch.zeros(self.max_Za + 1))
        self.scale = Parameter(torch.ones(self.max_Za + 1))
        reference = torch.zeros(self.max_Za + 1)
        if element_reference is not None:
            values = torch.as_tensor(element_reference, dtype=torch.float32)
            if values.shape != (self.max_Za + 1,):
                raise ValueError(
                    "element_reference must have length max_Za + 1, "
                    f"got {tuple(values.shape)} for max_Za={self.max_Za}"
                )
            reference = values
        self.register_buffer("reference", reference)

    def get_Ea(
        self,
        atom_feature: Tensor,
        isolated_atom_feature: Tensor,
        Za: Tensor,
    ) -> Tensor:
        difference = concat_block_states(atom_feature - isolated_atom_feature)
        projected = self.projection(difference).squeeze(-1)
        return self.reference[Za] + self.shift[Za] + self.scale[Za] * projected


def _find_facto_core(built_layers: List[Module]) -> Module:
    for layer in reversed(built_layers):
        if getattr(layer, "emits_isolated_atom_feature", False):
            return layer
    raise TypeError(
        "FACTOEnergyReadout must follow a FACTOCore in the layer list"
    )
