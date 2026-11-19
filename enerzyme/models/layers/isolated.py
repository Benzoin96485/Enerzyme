"""Subtract the neighbor-free state of the same element.

Geodite Eq. 1 and the FACTO energy head both form the residual
``h - h_iso`` before a bias-free readout, so an isolated atom contributes
no network energy. The Core emits both states. This layer writes the
difference back to ``atom_feature`` and leaves the readout itself generic
(:class:`~enerzyme.models.layers.readout.HierachicalReadout` or
:class:`~enerzyme.models.layers.readout.SimpleReadout`).
"""

from __future__ import annotations

from typing import List, Optional

from torch import Tensor
from torch.nn import Module

from . import BaseFFLayer


class IsolatedAtomResidualLayer(BaseFFLayer):
    """``atom_feature = atom_feature - isolated_atom_feature``.

    When a Core is already in ``built_layers``, this layer copies
    ``dim_feature_out``, ``feature_irreps``, and ``num_output_blocks`` so a
    following :class:`~enerzyme.models.layers.readout.HierachicalReadout`
    still sees the latent width. The subtraction itself does not need them.
    """

    def __init__(self, built_layers: Optional[List[Module]] = None) -> None:
        super().__init__(
            input_fields={"atom_feature", "isolated_atom_feature"},
            output_fields={"atom_feature"},
        )
        for layer in reversed(built_layers or []):
            width = getattr(layer, "dim_feature_out", None)
            if width is None:
                continue
            self.dim_feature_out = int(width)
            irreps = getattr(layer, "feature_irreps", None)
            if irreps is not None:
                self.feature_irreps = str(irreps)
            blocks = getattr(layer, "num_output_blocks", None)
            if blocks is not None:
                self.num_output_blocks = int(blocks)
            break

    def get_atom_feature(
        self,
        atom_feature: Tensor,
        isolated_atom_feature: Tensor,
    ) -> Tensor:
        if tuple(atom_feature.shape) != tuple(isolated_atom_feature.shape):
            raise ValueError(
                "atom_feature and isolated_atom_feature must share a shape, "
                f"got {tuple(atom_feature.shape)} and {tuple(isolated_atom_feature.shape)}"
            )
        return atom_feature - isolated_atom_feature
