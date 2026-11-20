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

from ..functional import segment_sum_coo


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


def _scatter(src: Tensor, index: Tensor, n_atoms: int) -> Tensor:
    return segment_sum_coo(src, index, dim_size=n_atoms)


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
        atom = atom + _scatter(message, idx_i, n_atoms)

        pair_force = self.phi_force(message) * dir_ij
        latent = latent + _scatter(pair_force, idx_i, n_atoms)
        force_edge = self.phi_f(message).unsqueeze(1) * pair_force.unsqueeze(-1)
        force = force + _scatter(force_edge, idx_i, n_atoms)

        delta_r = self.phi_r(atom).unsqueeze(1) * force
        neighbor_dr = self.phi_r_ext(message).unsqueeze(1) * displacement[idx_j]
        displacement = _scatter(neighbor_dr, idx_i, n_atoms) + delta_r

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
        atom = atom + _scatter(message, idx_i, n_atoms)

        directional = self.equiv_direction(message).unsqueeze(1) * dir_ij.unsqueeze(-1)
        transported = self.equiv_neighbor(message).unsqueeze(1) * force[idx_j]
        force = force + _scatter(directional + transported, idx_i, n_atoms)
        atom = atom + torch.sum(force * self.equiv_update(force), dim=1)
        return atom, force
