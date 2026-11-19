"""One M3GNet block: three-body bond update, then gated graph convolution.

The three-body term aggregates

``j_l(r_ik) Y_l^0(θ_jik) * σ(W v_k) * f_c(r_ij) f_c(r_ik)``

onto bond ``ij`` and adds a bias-free gated map. The convolution then updates
the bond from ``(v_i, v_j, e_ij)`` and scatters the new bond onto the receiver.
Both updates are multiplied by a bias-free linear map of the spherical Bessel
bond basis. ``i`` is the receiver (``idx_i``).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, Module, Sigmoid

from ..blocks.mlp import GatedMLP
from ..functional import segment_sum_coo


class M3GNetBlock(Module):
    """Three-body update followed by an edge update and a receiver update."""

    def __init__(
        self,
        degree: int,
        dim_node: int,
        dim_edge: int,
        units: int,
        n_conv_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.degree = int(degree)
        self.dim_node = int(dim_node)
        self.dim_edge = int(dim_edge)
        hidden = [int(units)] * int(n_conv_hidden)
        self.atom_to_basis = Linear(self.dim_node, self.degree)
        self.atom_act = Sigmoid()
        self.bond_update = GatedMLP(
            self.degree,
            [],
            self.dim_edge,
            activate_last=True,
            use_bias=False,
        )
        packed = 2 * self.dim_node + self.dim_edge
        self.edge_mlp = GatedMLP(
            packed, hidden, self.dim_edge, activate_last=True, use_bias=True
        )
        self.node_mlp = GatedMLP(
            packed, hidden, self.dim_node, activate_last=True, use_bias=True
        )
        self.edge_weight = Linear(self.degree, self.dim_edge, bias=False)
        self.node_weight = Linear(self.degree, self.dim_node, bias=False)
        self.dropout = Dropout(float(dropout)) if float(dropout) > 0 else None

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        idx_ij: Tensor,
        idx_ik: Tensor,
        joint: Tensor,
        envelope: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if edge.shape[0] == 0:
            return node, edge

        aggregated = edge.new_zeros(edge.shape[0], self.degree)
        if idx_ij.numel() > 0:
            end_atom = self.atom_act(self.atom_to_basis(node))
            neighbor = end_atom[idx_j[idx_ik]]
            weight = envelope[idx_ij] * envelope[idx_ik]
            basis = joint * neighbor * weight.unsqueeze(-1)
            aggregated = segment_sum_coo(basis, idx_ij, dim_size=edge.shape[0])
        edge = edge + self.bond_update(aggregated)

        center = node[idx_i]
        neighbor_state = node[idx_j]
        packed = torch.cat((center, neighbor_state, edge), dim=-1)
        radial_edge = self.edge_weight(rbf)
        edge = edge + self.edge_mlp(packed) * radial_edge

        packed = torch.cat((center, neighbor_state, edge), dim=-1)
        message = self.node_mlp(packed) * self.node_weight(rbf)
        node = node + segment_sum_coo(message, idx_i, dim_size=node.shape[0])
        if self.dropout is not None:
            edge = self.dropout(edge)
            node = self.dropout(node)
        return node, edge
