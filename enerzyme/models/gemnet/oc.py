"""GemNet-OC interaction block on top of the GemNet layers.

Gasteiger et al., TMLR 2022. The edge triplet and the dihedral quadruplet
reuse :class:`~enerzyme.models.gemnet.layers.TripletInteraction` and
:class:`~enerzyme.models.gemnet.layers.QuadrupletInteraction`. Atom–edge,
edge–atom, and atom–atom messages are the interaction hierarchy. Aggregation
is the scatter form of the reference bilinear. The padded matmul in
fairchem is the same contraction.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..mp.ops import scatter_sum
from .layers import (
    AtomUpdate,
    BilinearAggregate,
    EdgeEmbedding,
    GemNetDense,
    GemNetResidual,
    QuadrupletInteraction,
    TripletInteraction,
    VarianceScale,
)


_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


class AtomEdgeInteraction(Module):
    """Atom embeddings ride an auxiliary edge and update a main edge.

    The radial gate uses the auxiliary distance. The circular bilinear uses
    the main-edge radial factor and the angle at the shared receiver.
    Both orientations of the main edge are updated.
    """

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_trip: int,
        emb_bilinear: int,
        emb_rbf: int,
        emb_cbf: int,
    ) -> None:
        super().__init__()
        self.dense = GemNetDense(emb_atom, emb_atom, activation=True)
        self.mlp_rbf = GemNetDense(emb_rbf, emb_atom, activation=False)
        self.scale_rbf = VarianceScale()
        self.down = GemNetDense(emb_atom, emb_trip, activation=True)
        self.bilinear = BilinearAggregate(emb_trip, emb_cbf, emb_bilinear)
        self.scale_cbf = VarianceScale()
        self.up_ca = GemNetDense(emb_bilinear, emb_edge, activation=True)
        self.up_ac = GemNetDense(emb_bilinear, emb_edge, activation=True)

    def forward(
        self,
        h: Tensor,
        rbf: Tensor,
        radial_w: Tensor,
        angular: Tensor,
        idx_j: Tensor,
        idx_other: Tensor,
        idx_edge: Tensor,
        swap: Tensor,
        num_main: int,
    ) -> Tensor:
        width = self.up_ca.linear.out_features
        if num_main == 0 or idx_j.numel() == 0:
            return h.new_zeros(num_main, width)
        x = self.down(self.scale_rbf(self.dense(h)[idx_j] * self.mlp_rbf(rbf)))
        aggregated = self.scale_cbf(
            self.bilinear(x, idx_other, idx_edge, angular, radial_w, num_main)
        )
        direct = self.up_ca(aggregated)
        swapped = self.up_ac(aggregated)[swap] if swap.numel() else direct
        return (direct + swapped) * _INV_SQRT_2


class EdgeAtomInteraction(Module):
    """Main-edge messages update atoms through an auxiliary edge.

    The first aggregation lands on the auxiliary edge. The second sums those
    edges onto their receiver. There is no reverse-edge swap: the output is
    an atom residual.
    """

    def __init__(
        self,
        emb_edge: int,
        emb_atom: int,
        emb_trip: int,
        emb_bilinear: int,
        emb_rbf: int,
        emb_cbf: int,
    ) -> None:
        super().__init__()
        self.dense = GemNetDense(emb_edge, emb_edge, activation=True)
        self.mlp_rbf = GemNetDense(emb_rbf, emb_edge, activation=False)
        self.scale_rbf = VarianceScale()
        self.down = GemNetDense(emb_edge, emb_trip, activation=True)
        self.bilinear = BilinearAggregate(emb_trip, emb_cbf, emb_bilinear)
        self.scale_cbf = VarianceScale()
        self.up = GemNetDense(emb_bilinear, emb_atom, activation=True)

    def forward(
        self,
        m: Tensor,
        rbf: Tensor,
        radial_w: Tensor,
        angular: Tensor,
        idx_other: Tensor,
        idx_edge: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        width = self.bilinear.units_out
        n_aux = int(receiver.shape[0])
        if n_aux == 0 or m.shape[0] == 0:
            scattered = m.new_zeros(num_nodes, width)
        else:
            x = self.down(self.scale_rbf(self.dense(m) * self.mlp_rbf(rbf)))
            aggregated = self.scale_cbf(
                self.bilinear(x, idx_other, idx_edge, angular, radial_w, n_aux)
            )
            scattered = scatter_sum(aggregated, receiver, num_nodes)
        return self.up(scattered)


class AtomPairInteraction(Module):
    """Long-range atom–atom bilinear with a radial basis and no angle."""

    def __init__(
        self,
        emb_atom: int,
        emb_in: int,
        emb_out: int,
        emb_rbf: int,
    ) -> None:
        super().__init__()
        self.emb_in = emb_in
        self.emb_rbf = emb_rbf
        self.down = GemNetDense(emb_atom, emb_in, activation=True)
        self.bilinear = GemNetDense(emb_rbf * emb_in, emb_out, activation=False)
        self.scale = VarianceScale()
        self.up = GemNetDense(emb_out, emb_atom, activation=True)

    def forward(self, h: Tensor, rbf: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        num_nodes = int(h.shape[0])
        if idx_i.numel() == 0:
            mixed = h.new_zeros(num_nodes, self.emb_rbf * self.emb_in)
        else:
            sender = self.down(h)[idx_j]
            outer = rbf.unsqueeze(-1) * sender.unsqueeze(-2)
            mixed = scatter_sum(outer.reshape(outer.shape[0], -1), idx_i, num_nodes)
        return self.up(self.scale(self.bilinear(mixed)))


class GemNetOCOutput(Module):
    """Per-block atom latent. The energy head stays outside the Core.

    ``n_after >= 1`` residual-adds the current atom state before the extra
    residuals, which is the GemNet-OC output block. ``n_after == 0`` is the
    GemNet-T scatter.
    """

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_rbf: int,
        n_hidden: int,
        n_after: int,
    ) -> None:
        super().__init__()
        if n_after < 0:
            raise ValueError(f"n_after must be >= 0, got {n_after}")
        self.n_after = int(n_after)
        self.update = AtomUpdate(emb_atom, emb_edge, emb_rbf, n_hidden)
        self.after = ModuleList(
            [GemNetResidual(emb_atom) for _ in range(self.n_after)]
        )

    def forward(self, h: Tensor, m: Tensor, rbf: Tensor, idx_i: Tensor) -> Tensor:
        hidden = self.update(m, rbf, idx_i, h.shape[0])
        if self.n_after == 0:
            return hidden
        hidden = (hidden + h) * _INV_SQRT_2
        for layer in self.after:
            hidden = layer(hidden)
        return hidden


class GemNetOCBlock(Module):
    """One GemNet-OC block: edge hierarchy, then atom hierarchy, then residuals."""

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_trip_in: int,
        emb_trip_out: int,
        emb_quad_in: int,
        emb_quad_out: int,
        emb_aint_in: int,
        emb_aint_out: int,
        emb_rbf: int,
        emb_cbf: int,
        emb_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_atom_emb_layers: int,
        quad_interaction: bool,
        atom_edge_interaction: bool,
        edge_atom_interaction: bool,
        atom_interaction: bool,
    ) -> None:
        super().__init__()
        self.quad_interaction = bool(quad_interaction)
        self.atom_edge_interaction = bool(atom_edge_interaction)
        self.edge_atom_interaction = bool(edge_atom_interaction)
        self.atom_interaction = bool(atom_interaction)
        self.dense_ca = GemNetDense(emb_edge, emb_edge, activation=True)
        self.trip = TripletInteraction(
            emb_edge, emb_trip_in, emb_trip_out, emb_rbf, emb_cbf
        )
        self.quad: Optional[QuadrupletInteraction]
        if self.quad_interaction:
            self.quad = QuadrupletInteraction(
                emb_edge, emb_quad_in, emb_quad_out, emb_rbf, emb_cbf, emb_sbf
            )
        else:
            self.quad = None
        self.atom_edge: Optional[AtomEdgeInteraction]
        if self.atom_edge_interaction:
            self.atom_edge = AtomEdgeInteraction(
                emb_atom, emb_edge, emb_trip_in, emb_trip_out, emb_rbf, emb_cbf
            )
        else:
            self.atom_edge = None
        self.edge_atom: Optional[EdgeAtomInteraction]
        if self.edge_atom_interaction:
            self.edge_atom = EdgeAtomInteraction(
                emb_edge, emb_atom, emb_trip_in, emb_trip_out, emb_rbf, emb_cbf
            )
        else:
            self.edge_atom = None
        self.pair: Optional[AtomPairInteraction]
        if self.atom_interaction:
            self.pair = AtomPairInteraction(emb_atom, emb_aint_in, emb_aint_out, emb_rbf)
        else:
            self.pair = None

        self.layers_before_skip = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_before_skip)]
        )
        self.layers_after_skip = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_after_skip)]
        )
        self.atom_emb_layers = ModuleList(
            [GemNetResidual(emb_atom) for _ in range(num_atom_emb_layers)]
        )
        self.atom_update = AtomUpdate(emb_atom, emb_edge, emb_rbf, num_atom)
        self.concat = EdgeEmbedding(emb_atom, emb_edge, emb_edge)
        self.residual_m = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_concat)]
        )
        n_edge = 2 + int(self.quad_interaction) + int(self.atom_edge_interaction)
        n_atom = 1 + int(self.edge_atom_interaction) + int(self.atom_interaction)
        self.inv_sqrt_edge = n_edge ** -0.5
        self.inv_sqrt_atom = n_atom ** -0.5

    def forward(
        self,
        h: Tensor,
        m: Tensor,
        rbf_e2e: Tensor,
        rbf_h: Tensor,
        radial_w3: Tensor,
        angular3: Tensor,
        idx_edge3: Tensor,
        idx_other3: Tensor,
        swap: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf_q: Optional[Tensor] = None,
        cbf_q: Optional[Tensor] = None,
        radial_w4: Optional[Tensor] = None,
        angular4: Optional[Tensor] = None,
        edge_db: Optional[Tensor] = None,
        edge_ca: Optional[Tensor] = None,
        rbf_a2e: Optional[Tensor] = None,
        radial_a2e: Optional[Tensor] = None,
        angular_a2e: Optional[Tensor] = None,
        a2e_sender: Optional[Tensor] = None,
        a2e_other: Optional[Tensor] = None,
        a2e_edge: Optional[Tensor] = None,
        rbf_e2a: Optional[Tensor] = None,
        radial_e2a: Optional[Tensor] = None,
        angular_e2a: Optional[Tensor] = None,
        e2a_other: Optional[Tensor] = None,
        e2a_edge: Optional[Tensor] = None,
        e2a_receiver: Optional[Tensor] = None,
        rbf_a2a: Optional[Tensor] = None,
        a2a_i: Optional[Tensor] = None,
        a2a_j: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        num_main = int(m.shape[0])
        interacted = self.dense_ca(m) + self.trip(
            m, rbf_e2e, radial_w3, angular3, idx_edge3, idx_other3, swap
        )
        if self.quad is not None:
            assert (
                rbf_q is not None
                and cbf_q is not None
                and radial_w4 is not None
                and angular4 is not None
                and edge_db is not None
                and edge_ca is not None
            )
            interacted = interacted + self.quad(
                m, rbf_q, cbf_q, radial_w4, angular4, edge_db, edge_ca, swap
            )
        if self.atom_edge is not None:
            assert (
                rbf_a2e is not None
                and radial_a2e is not None
                and angular_a2e is not None
                and a2e_sender is not None
                and a2e_other is not None
                and a2e_edge is not None
            )
            interacted = interacted + self.atom_edge(
                h,
                rbf_a2e,
                radial_a2e,
                angular_a2e,
                a2e_sender,
                a2e_other,
                a2e_edge,
                swap,
                num_main,
            )
        interacted = interacted * self.inv_sqrt_edge

        if self.edge_atom is not None:
            assert (
                rbf_e2a is not None
                and radial_e2a is not None
                and angular_e2a is not None
                and e2a_other is not None
                and e2a_edge is not None
                and e2a_receiver is not None
            )
            h = h + self.edge_atom(
                m,
                rbf_e2a,
                radial_e2a,
                angular_e2a,
                e2a_other,
                e2a_edge,
                e2a_receiver,
                h.shape[0],
            )
        if self.pair is not None:
            assert rbf_a2a is not None and a2a_i is not None and a2a_j is not None
            h = h + self.pair(h, rbf_a2a, a2a_i, a2a_j)
        h = h * self.inv_sqrt_atom

        x = interacted
        for layer in self.layers_before_skip:
            x = layer(x)
        m = (m + x) * _INV_SQRT_2
        for layer in self.layers_after_skip:
            m = layer(m)
        for layer in self.atom_emb_layers:
            h = layer(h)
        h = (h + self.atom_update(m, rbf_h, idx_i, h.shape[0])) * _INV_SQRT_2
        m2 = self.concat(h, m, idx_i, idx_j)
        for layer in self.residual_m:
            m2 = layer(m2)
        m = (m + m2) * _INV_SQRT_2
        return h, m
