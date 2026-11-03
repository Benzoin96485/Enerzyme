"""PaiNN message and update blocks (Schütt, Unke, Gastegger, ICML 2021).

Residual continuous-filter convolution on scalar features ``[N, F]`` and
Cartesian vector features ``[N, 3, F]``. Adapted from SchNetPack 2.0.4
``schnetpack.representation.painn`` (MIT). The radial filter itself is owned
by :class:`~enerzyme.models.painn.core.PaiNNCore`.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Sequential

from ..functional import segment_sum_coo
from ..layers.gated_equivariant import cartesian_dense


class PaiNNInteraction(Module):
    """Interatomic residual (paper Eqs. 7–8).

    Messages are gathered from neighbor ``idx_j`` onto center ``idx_i``.
    ``dir_ij`` is the unit vector ``(R_j - R_i) / ||R_j - R_i||``.
    ``propagate_vectors=False`` drops the ``W_vv`` term (the paper ablation).
    """

    def __init__(self, n_atom_basis: int, propagate_vectors: bool = True) -> None:
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.propagate_vectors = propagate_vectors
        self.interatomic_context_net = Sequential(
            cartesian_dense(n_atom_basis, n_atom_basis, activation=True),
            cartesian_dense(n_atom_basis, 3 * n_atom_basis, activation=False),
        )

    def forward(
        self,
        q: Tensor,
        mu: Tensor,
        wij: Tensor,
        dir_ij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        n_atoms: int,
    ) -> tuple[Tensor, Tensor]:
        x = self.interatomic_context_net(q)[idx_j]
        x = wij * x
        dq, dmu_r, dmu_mu = torch.split(x, self.n_atom_basis, dim=-1)
        dq = segment_sum_coo(dq, idx_i, dim_size=n_atoms)
        dmu = dmu_r.unsqueeze(-2) * dir_ij.unsqueeze(-1)
        if self.propagate_vectors:
            dmu = dmu + dmu_mu.unsqueeze(-2) * mu[idx_j]
        dmu = segment_sum_coo(dmu, idx_i, dim_size=n_atoms)
        return q + dq, mu + dmu


class PaiNNMixing(Module):
    """Atomwise residual that couples the two streams (paper Eqs. 9–10).

    ``use_vector_scalar_product=False`` drops ``<Uv, Vv>`` from the scalar
    update. The vector channel mix is bias-free.
    """

    def __init__(
        self,
        n_atom_basis: int,
        epsilon: float = 1e-8,
        use_vector_scalar_product: bool = True,
    ) -> None:
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.epsilon = float(epsilon)
        self.use_vector_scalar_product = use_vector_scalar_product
        self.intraatomic_context_net = Sequential(
            cartesian_dense(2 * n_atom_basis, n_atom_basis, activation=True),
            cartesian_dense(n_atom_basis, 3 * n_atom_basis, activation=False),
        )
        self.mu_channel_mix = cartesian_dense(
            n_atom_basis, 2 * n_atom_basis, bias=False, activation=False
        )

    def forward(self, q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        mixed = self.mu_channel_mix(mu)
        mu_v, mu_w = torch.split(mixed, self.n_atom_basis, dim=-1)
        mu_vn = torch.sqrt(torch.sum(mu_v * mu_v, dim=-2) + self.epsilon)
        x = self.intraatomic_context_net(torch.cat([q, mu_vn], dim=-1))
        dq, dmu_scale, dq_prod = torch.split(x, self.n_atom_basis, dim=-1)
        dmu = dmu_scale.unsqueeze(-2) * mu_w
        if self.use_vector_scalar_product:
            dq = dq + dq_prod * torch.sum(mu_v * mu_w, dim=-2)
        return q + dq, mu + dmu
