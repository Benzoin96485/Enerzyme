"""NewtonNet interaction blocks.

Haghighatlari et al., Digital Discovery 2022 (arXiv:2108.02913). The
reference package is https://github.com/THGLab/NewtonNet. It is not
imported, and its checkpoints do not load.

``NewtonNetPaperInteraction`` follows the paper's four modules: a symmetric
invariant message, a latent force along the unit displacement (Newton's
third law), a featurized force, a displacement updated from that force
(Newton's second law), and the contraction ``-φ_u (f · dr)`` back into the
atomic features.

``NewtonNetRepoInteraction`` follows ``InteractionNet`` on the current
``main`` branch. That rewrite keeps one Cartesian force feature, mixes in
the neighbor's force, and contracts ``f · W f``.

Both blocks receive a radial basis and a cutoff envelope. The linear map
on the basis lives here because each layer has its own weights. Receivers
are ``idx_i`` and senders are ``idx_j``. ``dir_ij`` is
``(R_j - R_i) / ||R_j - R_i||``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..mp.basis.envelope import polynomial_envelope
from ..mp.block import BlockState
from ..mp.core import FieldEmbed
from ..mp.head import Head
from ..mp.ops import scatter_sum


def xavier_linear(in_features: int, out_features: int, bias: bool, gain: float = 1.0) -> nn.Linear:
    """Linear map with the Xavier init used by NewtonNet's ``Dense``."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    nn.init.xavier_uniform_(layer.weight, gain=float(gain))
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def silu_mlp(
    in_features: int,
    out_features: int,
    bias: bool,
    first_gain: float = 1.0,
) -> nn.Sequential:
    """Two linear maps with SiLU between them and none after the second."""
    return nn.Sequential(
        xavier_linear(in_features, out_features, bias=bias, gain=first_gain),
        nn.SiLU(),
        xavier_linear(out_features, out_features, bias=bias, gain=1.0),
    )



class NewtonNetPaperInteraction(nn.Module):
    """One paper message-passing layer (Eqs. 7–17).

    ``a`` is ``[N, F]``. ``force`` and ``displacement`` are ``[N, 3, F]``.
    ``latent`` is the accumulated physical force ``[N, 3]``. ``envelope``
    multiplies the radial linear map, including its bias, so a vanished
    window removes the edge. ``φ_F``, ``φ_f``, and ``φ'_r`` are bias-free,
    which is how the paper keeps that window on the equivariant path.
    The displacement update replaces the center state with the weighted
    neighbor displacements plus ``φ_r(a) f``. It does not add the center's
    previous displacement again.
    """

    def __init__(
        self,
        n_features: int,
        n_basis: int,
        displacement_init_gain: float = 1e-3,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.phi_rbf = xavier_linear(int(n_basis), self.n_features, bias=True)
        self.phi_a = silu_mlp(self.n_features, self.n_features, bias=True)
        self.phi_force = xavier_linear(self.n_features, 1, bias=False)
        self.phi_f = silu_mlp(self.n_features, self.n_features, bias=False)
        self.phi_r = silu_mlp(
            self.n_features,
            self.n_features,
            bias=True,
            first_gain=float(displacement_init_gain),
        )
        self.phi_r_ext = silu_mlp(self.n_features, self.n_features, bias=False)
        self.phi_u = silu_mlp(self.n_features, self.n_features, bias=True)

    def forward(
        self,
        atom: Tensor,
        force: Tensor,
        displacement: Tensor,
        latent: Tensor,
        rbf: Tensor,
        envelope: Tensor,
        dir_ij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        n_atoms: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        edge = self.phi_rbf(rbf) * envelope
        projected = self.phi_a(atom)
        message = edge * projected[idx_i] * projected[idx_j]
        atom = atom + scatter_sum(message, idx_i, n_atoms)

        pair_force = self.phi_force(message) * dir_ij
        latent = latent + scatter_sum(pair_force, idx_i, n_atoms)
        force_edge = self.phi_f(message).unsqueeze(1) * pair_force.unsqueeze(-1)
        force = force + scatter_sum(force_edge, idx_i, n_atoms)

        delta_r = self.phi_r(atom).unsqueeze(1) * force
        neighbor_dr = self.phi_r_ext(message).unsqueeze(1) * displacement[idx_j]
        displacement = scatter_sum(neighbor_dr, idx_i, n_atoms) + delta_r

        contracted = torch.sum(force * displacement, dim=1)
        atom = atom - self.phi_u(atom) * contracted
        return atom, force, displacement, latent


class NewtonNetRepoInteraction(nn.Module):
    """One ``InteractionNet`` step from THGLab/NewtonNet ``main``.

    The radial map is bias-free, so multiplying the envelope after the map
    matches ``Linear(envelope * basis)``. The force feature is updated by a
    directional message plus a bias-free gate on the sender's force. The
    scalar update is the spatial contraction ``f · W f`` with bias-free ``W``.
    """

    def __init__(self, n_features: int, n_basis: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.phi_rbf = xavier_linear(int(n_basis), self.n_features, bias=False)
        self.phi_a = silu_mlp(self.n_features, self.n_features, bias=True)
        self.equiv_direction = silu_mlp(self.n_features, self.n_features, bias=False)
        self.equiv_neighbor = silu_mlp(self.n_features, self.n_features, bias=False)
        self.equiv_update = xavier_linear(self.n_features, self.n_features, bias=False)

    def forward(
        self,
        atom: Tensor,
        force: Tensor,
        rbf: Tensor,
        envelope: Tensor,
        dir_ij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        n_atoms: int,
    ) -> tuple[Tensor, Tensor]:
        edge = self.phi_rbf(rbf) * envelope
        projected = self.phi_a(atom)
        message = edge * projected[idx_i] * projected[idx_j]
        atom = atom + scatter_sum(message, idx_i, n_atoms)

        directional = self.equiv_direction(message).unsqueeze(1) * dir_ij.unsqueeze(-1)
        transported = self.equiv_neighbor(message).unsqueeze(1) * force[idx_j]
        force = force + scatter_sum(directional + transported, idx_i, n_atoms)
        atom = atom + torch.sum(force * self.equiv_update(force), dim=1)
        return atom, force


def build_newtonnet(core, built_layers):
    from ..layers.rbf import BesselRBFLayer

    for layer in built_layers:
        if isinstance(layer, BesselRBFLayer) and layer.apply_cutoff_fn:
            raise ValueError(
                "NewtonNet multiplies the polynomial envelope onto the "
                "radial basis. Set BesselRBF apply_cutoff_fn: false."
            )
    core.adopt_radius_stack(built_layers)


class NewtonEmbed(FieldEmbed):
    def __init__(self, width, num_rbf, cutoff, order, variant):
        super().__init__(width=width, vector_channels=width)
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.order = order
        self.variant = variant

    def forward(self, inputs, ctx):
        state = super().forward(inputs, ctx)
        if ctx.rbf.ndim != 2 or ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf must have shape [E, {self.num_rbf}], got {tuple(ctx.rbf.shape)}")
        if ctx.sender.shape[0] != ctx.num_edges or ctx.dist.shape[0] != ctx.num_edges:
            raise ValueError("edge indices and distances must cover the same edges")
        if ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError(f"vij_sr must have shape [E, 3], got {tuple(ctx.vec.shape)}")
        window = polynomial_envelope(ctx.dist.reshape(-1), self.cutoff, self.order).reshape(-1, 1)
        if self.variant == "repo":
            return BlockState(state.scalar, state.equi, window)
        displacement = state.equi.new_zeros(state.equi.shape)
        latent = state.scalar.new_zeros(ctx.num_nodes, 3)
        return BlockState(state.scalar, state.equi, window, {"displacement": displacement, "latent": latent})


def newton_paper_step(message, state, ctx):
    atom, force, displacement, latent = message(
        state.scalar, state.equi, state.aux["displacement"], state.aux["latent"],
        ctx.rbf, state.edge, ctx.unit(), ctx.receiver, ctx.sender, ctx.num_nodes,
    )
    return BlockState(atom, force, state.edge, {"displacement": displacement, "latent": latent})


def newton_repo_step(message, state, ctx):
    atom, force = message(state.scalar, state.equi, ctx.rbf, state.edge, ctx.unit(),
                          ctx.receiver, ctx.sender, ctx.num_nodes)
    return BlockState(atom, force, state.edge)


class NewtonHead(Head):
    output_fields = ("atom_feature", "atom_vector_feature", "latent_force")

    def __init__(self, variant):
        super().__init__()
        self.variant = variant

    def forward(self, state, ctx, taps, inputs):
        latent = state.aux["latent"] if self.variant == "paper" else state.equi.sum(dim=-1)
        return {"atom_feature": state.scalar, "atom_vector_feature": state.equi, "latent_force": latent}
