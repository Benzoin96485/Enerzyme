"""CACE density embedding and message-passing blocks."""

from __future__ import annotations

from torch import Tensor, nn

from ..mp.basis.envelope import polynomial_envelope
from ..mp.block import BlockState
from ..mp.core import Embed
from ..mp.graph import GraphContext
from ..mp.ops import scatter_sum
from .basis import CartesianAngularBasis, InvariantSymmetrizer
from .interaction import MessagePassing, RadialChannelMix


class CACEEmbed(Embed):
    """Build the edge basis and contract the initial summed atomic density."""

    input_fields = ("atom_embedding",)

    def __init__(self, dim_embedding, cutoff_sr, num_rbf, lmax, max_nu, n_radial, envelope_p, separate_receiver):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.cutoff_sr = cutoff_sr
        self.num_rbf = num_rbf
        self.n_radial = n_radial
        self.n_channels = dim_embedding ** 2
        self.envelope_p = envelope_p
        self.eps = 1e-12
        self.angular = CartesianAngularBasis(lmax)
        self.n_angular = self.angular.n_angular
        self.symmetrizer = InvariantSymmetrizer(lmax, max_nu)
        self.n_invariants = self.symmetrizer.n_invariants
        self.mix = RadialChannelMix(
            self.num_rbf,
            self.n_radial,
            self.n_channels,
            self.angular.shells,
        )
        self.receiver_proj = (
            nn.Linear(self.dim_embedding, self.dim_embedding, bias=False)
            if separate_receiver
            else None
        )
        if self.receiver_proj is not None:
            nn.init.eye_(self.receiver_proj.weight)

    def _edge_basis(
        self,
        embedding: Tensor,
        receiver: Tensor,
        rbf: Tensor,
        distance: Tensor,
        vectors: Tensor,
        receivers: Tensor,
        senders: Tensor,
        window: Tensor,
    ) -> Tensor:
        radial = rbf * window.unsqueeze(-1)
        direction = vectors / distance.clamp_min(self.eps).unsqueeze(-1)
        monomials = self.angular(direction)
        edge_type = (
            receiver.index_select(0, receivers).unsqueeze(-1)
            * embedding.index_select(0, senders).unsqueeze(-2)
        ).reshape(receivers.shape[0], self.n_channels)
        return (
            radial[:, :, None, None]
            * monomials[:, None, :, None]
            * edge_type[:, None, None, :]
        )

    def forward(self, inputs, ctx: GraphContext) -> BlockState:
        atom_embedding = inputs["atom_embedding"]
        rbf, distance, vectors = ctx.rbf, ctx.dist, ctx.vec
        receivers, senders = ctx.receiver, ctx.sender
        cutoff = ctx.cutoff_values
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must be [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must be [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        n_atoms = atom_embedding.shape[0]
        n_edges = int(receivers.shape[0])
        if (
            senders.shape[0] != n_edges
            or distance.shape[0] != n_edges
            or vectors.shape[0] != n_edges
        ):
            raise ValueError(
                "distance, vectors, receivers, and senders must share the edge count, "
                f"got {distance.shape[0]}, {vectors.shape[0]}, {n_edges}, {senders.shape[0]}"
            )
        if vectors.shape[-1] != 3 and n_edges:
            raise ValueError(f"vectors must be [E, 3], got {tuple(vectors.shape)}")
        receiver = (
            self.receiver_proj(atom_embedding)
            if self.receiver_proj is not None
            else atom_embedding
        )
        if n_edges == 0:
            mixed = atom_embedding.new_zeros(
                n_atoms, self.n_radial, self.n_angular, self.n_channels
            )
            edge = atom_embedding.new_zeros(
                0, self.num_rbf, self.n_angular, self.n_channels
            )
            window = atom_embedding.new_zeros(0)
        else:
            if cutoff is None:
                window = polynomial_envelope(distance, self.cutoff_sr, self.envelope_p)
            else:
                window = cutoff.reshape(-1).to(dtype=distance.dtype)
                if window.shape[0] != n_edges:
                    raise ValueError(
                        f"cutoff has {window.shape[0]} values for {n_edges} edges"
                    )
            edge = self._edge_basis(
                atom_embedding,
                receiver,
                rbf,
                distance,
                vectors,
                receivers,
                senders,
                window,
            )
            mixed = self.mix(scatter_sum(edge, receivers, n_atoms))
        invariant = self.symmetrizer(mixed)
        return BlockState(scalar=invariant.flatten(1), equi=mixed, edge=(edge, window))


class CACEBlock(MessagePassing):
    """Update the density and reapply the shared contraction after each message."""

    def __init__(self, *args, mix, symmetrizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.mix = mix
        self.symmetrizer = symmetrizer

    def forward(self, state: BlockState, ctx: GraphContext) -> BlockState:
        edge, window = state.edge
        mixed = state.equi
        invariant = state.scalar.view(ctx.num_nodes, mixed.shape[1], self.symmetrizer.n_invariants, mixed.shape[-1])
        mixed = super().forward(mixed, invariant, edge, ctx.dist, window, ctx.receiver, ctx.sender, ctx.num_nodes, self.mix)
        return BlockState(scalar=self.symmetrizer(mixed).flatten(1), equi=mixed, edge=state.edge)
