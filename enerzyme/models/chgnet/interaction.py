"""One CHGNet interaction: atom convolution, bond convolution, angle update.

Deng et al., Nat. Mach. Intell. 2023, eq. (2), with the reference
concatenation order. ``idx_i`` is the receiver. Messages are summed, passed
through a bias-free linear map, and added back to the state. The gated MLP
is the shared module, including the optional layer norm used by the
reference network.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module

from ..mp.nn.mlp import GatedMLP
from ..mp.ops import scatter_sum


def _feature_norm(name: Optional[str], dim: int) -> Optional[Module]:
    if name is None:
        return None
    if name == "layer":
        return torch.nn.LayerNorm(int(dim))
    if name == "batch":
        return torch.nn.BatchNorm1d(int(dim))
    raise ValueError(f"normalization must be 'layer', 'batch', or None, got {name!r}")


def _gated(
    in_features: int,
    hidden_dims: Sequence[int],
    out_features: int,
    dropout: float,
    norm: Optional[str],
) -> GatedMLP:
    return GatedMLP(
        in_features,
        list(hidden_dims),
        out_features,
        activate_last=True,
        use_bias=True,
        norm=norm,
        dropout=dropout,
    )


class AtomConv(Module):
    """Update atoms from ``(v_i, e_ij, v_j)`` weighted by the atom-graph basis."""

    def __init__(
        self,
        dim_atom: int,
        dim_bond: int,
        hidden_dims: Sequence[int],
        dropout: float,
        gmlp_norm: Optional[str],
        conv_norm: Optional[str],
        mlp_out_bias: bool,
    ) -> None:
        super().__init__()
        self.message = _gated(
            2 * int(dim_atom) + int(dim_bond),
            hidden_dims,
            int(dim_atom),
            dropout,
            gmlp_norm,
        )
        self.mlp_out = Linear(int(dim_atom), int(dim_atom), bias=mlp_out_bias)
        self.norm = _feature_norm(conv_norm, dim_atom)

    def forward(
        self,
        atom: Tensor,
        bond: Tensor,
        bond_weight: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        directed_to_undirected: Tensor,
    ) -> Tensor:
        if idx_i.numel() == 0:
            updated = self.mlp_out(atom.new_zeros(atom.shape))
        else:
            packed = torch.cat(
                (atom[idx_i], bond[directed_to_undirected], atom[idx_j]),
                dim=-1,
            )
            messages = self.message(packed) * bond_weight[directed_to_undirected]
            updated = scatter_sum(messages, idx_i, atom.shape[0])
            updated = self.mlp_out(updated)
        updated = updated + atom
        if self.norm is not None:
            updated = self.norm(updated)
        return updated


class BondConv(Module):
    """Update the left bond from ``(e_left, e_right, a, v_center)``."""

    def __init__(
        self,
        dim_atom: int,
        dim_bond: int,
        dim_angle: int,
        hidden_dims: Sequence[int],
        dropout: float,
        gmlp_norm: Optional[str],
        conv_norm: Optional[str],
        mlp_out_bias: bool,
    ) -> None:
        super().__init__()
        self.message = _gated(
            int(dim_atom) + 2 * int(dim_bond) + int(dim_angle),
            hidden_dims,
            int(dim_bond),
            dropout,
            gmlp_norm,
        )
        self.mlp_out = Linear(int(dim_bond), int(dim_bond), bias=mlp_out_bias)
        self.norm = _feature_norm(conv_norm, dim_bond)

    def forward(
        self,
        atom: Tensor,
        bond: Tensor,
        bond_weight: Tensor,
        angle: Tensor,
        center: Tensor,
        left: Tensor,
        right: Tensor,
    ) -> Tensor:
        if left.numel() == 0:
            updated = self.mlp_out(bond.new_zeros(bond.shape))
        else:
            packed = torch.cat(
                (bond[left], bond[right], angle, atom[center]),
                dim=-1,
            )
            weight = bond_weight[left] * bond_weight[right]
            messages = self.message(packed) * weight
            updated = scatter_sum(messages, left, bond.shape[0])
            updated = self.mlp_out(updated)
        updated = updated + bond
        if self.norm is not None:
            updated = self.norm(updated)
        return updated


class AngleUpdate(Module):
    """Pointwise residual on each angle. There is no cutoff weight and no sum."""

    def __init__(
        self,
        dim_atom: int,
        dim_bond: int,
        dim_angle: int,
        hidden_dims: Sequence[int],
        dropout: float,
        gmlp_norm: Optional[str],
        conv_norm: Optional[str],
    ) -> None:
        super().__init__()
        self.message = _gated(
            int(dim_atom) + 2 * int(dim_bond) + int(dim_angle),
            hidden_dims,
            int(dim_angle),
            dropout,
            gmlp_norm,
        )
        self.norm = _feature_norm(conv_norm, dim_angle)

    def forward(
        self,
        atom: Tensor,
        bond: Tensor,
        angle: Tensor,
        center: Tensor,
        left: Tensor,
        right: Tensor,
    ) -> Tensor:
        if angle.numel() == 0:
            return angle
        packed = torch.cat(
            (bond[left], bond[right], angle, atom[center]),
            dim=-1,
        )
        updated = self.message(packed) + angle
        if self.norm is not None:
            updated = self.norm(updated)
        return updated
