"""TensorNet embedding and interaction (Simeon & De Fabritiis, NeurIPS 2023).

The pure PyTorch update rules follow
`torchmd/torchmd-net <https://github.com/torchmd/torchmd-net>`_ (MIT),
``torchmdnet.models.tensornet``, without the optional Warp kernels.
Species tables and the radial basis stay outside this module. Receivers are
``idx_i`` and senders are ``idx_j``. ``hat`` is the unit vector
``(R_j - R_i) / ||R_j - R_i||``.

:class:`TensorNet2Interaction` is the same tensor update. Its scalar MLP
reads the radial basis concatenated with the receiver and sender charge
channels, and the residual square is not multiplied by a total-charge factor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, LayerNorm, Module, ModuleList

from ..mp.ops import scatter_sum
from ..mp.equivariant.cartesian import (
    compose_tensor,
    decompose_tensor,
    frobenius_squared,
    symmetric_traceless,
    tensor_anticommutator,
    tensor_matmul,
    vector_to_skew_tensor,
)


def aggregate_edge_tensors(
    hat: Tensor,
    processed: Tensor,
    idx_i: Tensor,
    n_atoms: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sum edge channels ``(I, r_hat, r_hat r_hat^T)`` onto receivers.

    ``processed`` has shape ``[E, 3, F]`` and already includes the cutoff and
    the pair weight. The vector channel becomes a skew matrix. The outer
    product is symmetrized and made traceless after the sum.
    """
    width = int(processed.shape[-1])
    iso = scatter_sum(processed[:, 0, :], idx_i, n_atoms)
    vector = scatter_sum(processed[:, 1, None, :] * hat.unsqueeze(-1), idx_i, n_atoms)
    outer = hat.unsqueeze(-1) * hat.unsqueeze(-2)
    symmetric = scatter_sum(processed[:, 2, None, None, :] * outer.unsqueeze(-1), idx_i, n_atoms)
    return iso, vector_to_skew_tensor(vector), symmetric_traceless(symmetric)


class TensorNetEmbedding(Module):
    """Edge tensor features → gated atomic tensor (paper Eqs. 7–10)."""

    def __init__(self, dim_embedding: int, num_rbf: int) -> None:
        super().__init__()
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.distance_proj1 = Linear(self.num_rbf, self.dim_embedding)
        self.distance_proj2 = Linear(self.num_rbf, self.dim_embedding)
        self.distance_proj3 = Linear(self.num_rbf, self.dim_embedding)
        # TorchMD-Net ``emb2``: one linear mix of the concatenated species pair.
        self.pair_mix = Linear(2 * self.dim_embedding, self.dim_embedding)
        self.linears_scalar = ModuleList(
            [
                Linear(self.dim_embedding, 2 * self.dim_embedding, bias=True),
                Linear(2 * self.dim_embedding, 3 * self.dim_embedding, bias=True),
            ]
        )
        self.linears_tensor = ModuleList(
            [
                Linear(self.dim_embedding, self.dim_embedding, bias=False)
                for _ in range(3)
            ]
        )
        self.init_norm = LayerNorm(self.dim_embedding)

    def forward(
        self,
        atom_embedding: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        hat: Tensor,
        rbf: Tensor,
        window: Tensor,
    ) -> Tensor:
        n_atoms = atom_embedding.shape[0]
        width = self.dim_embedding
        # The two halves of the affine pair map act independently on the
        # receiver and sender embeddings, which are shared by incident edges.
        weights = self.pair_mix.weight.reshape(2 * width, width)
        projected = F.linear(atom_embedding, weights).reshape(n_atoms, width, 2)
        receiver = projected[..., 0] + self.pair_mix.bias
        pair = receiver[idx_i] + projected[..., 1][idx_j]
        radial = torch.stack(
            (
                self.distance_proj1(rbf),
                self.distance_proj2(rbf),
                self.distance_proj3(rbf),
            ),
            dim=1,
        )
        processed = window.unsqueeze(-1).unsqueeze(-1) * pair.unsqueeze(1) * radial
        iso, skew, traceless = aggregate_edge_tensors(
            hat, processed, idx_i, n_atoms
        )
        features = compose_tensor(iso, skew, traceless)

        gates = self.init_norm(frobenius_squared(features))
        for linear in self.linears_scalar:
            gates = F.silu(linear(gates))
        gates = gates.reshape(n_atoms, 3, width)
        iso = self.linears_tensor[0](iso) * gates[:, 0, :]
        skew = self.linears_tensor[1](skew) * gates[:, 1, None, None, :]
        traceless = self.linears_tensor[2](traceless) * gates[:, 2, None, None, :]
        return compose_tensor(iso, skew, traceless)


def scalar_edge_gates(edge_input: Tensor, linears_scalar: ModuleList, window: Tensor, width: int) -> Tensor:
    """SiLU MLP, cutoff window, then ``[E, 3, F]`` gates."""
    edge = edge_input
    for linear in linears_scalar:
        edge = F.silu(linear(edge))
    return (edge * window.unsqueeze(-1)).reshape(-1, 3, width)


def tensor_message_update(
    features: Tensor,
    edge: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    linears_tensor: ModuleList,
    equivariance_group: str,
    charge_scale: Tensor,
) -> Tensor:
    """Normalized tensor residual shared by TensorNet and TensorNet2.

    ``edge`` is ``[E, 3, F]`` and already includes the cutoff window. ``O(3)``
    uses the anticommutator ``Y M + M Y``. ``SO(3)`` uses ``2 Y M``. The
    residual square is the ordinary matrix product. ``charge_scale`` multiplies
    the O(3) product and the square; TensorNet2 passes ones.
    """
    n_atoms = features.shape[0]
    width = int(features.shape[-1])
    normalized = features / (frobenius_squared(features) + 1.0)[:, None, None, :]
    iso, skew, traceless = decompose_tensor(normalized)
    iso = linears_tensor[0](iso)
    skew = linears_tensor[1](skew)
    traceless = linears_tensor[2](traceless)
    node = compose_tensor(iso, skew, traceless)
    # The channel map preserves the antisymmetry produced by decomposition.
    vector = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), dim=1)

    iso_m = scatter_sum(edge[:, 0, :] * iso[idx_j], idx_i, n_atoms)
    vector_m = scatter_sum(edge[:, 1, None, :] * vector[idx_j], idx_i, n_atoms)
    traceless_m = scatter_sum(edge[:, 2, None, None, :] * traceless[idx_j], idx_i, n_atoms)
    message = compose_tensor(iso_m, vector_to_skew_tensor(vector_m), traceless_m)
    scale = charge_scale[:, None, None, None]
    if equivariance_group == "O(3)":
        # Charge multiplies the anticommutator. The SO(3) product does not.
        mixed = scale * tensor_anticommutator(node, message)
    else:
        mixed = 2.0 * tensor_matmul(node, message)

    iso_c, skew_c, traceless_c = decompose_tensor(mixed)
    norm = frobenius_squared(mixed) + 1.0
    iso_c = linears_tensor[3](iso_c / norm)
    skew_c = linears_tensor[4](skew_c / norm[:, None, None, :])
    traceless_c = linears_tensor[5](traceless_c / norm[:, None, None, :])
    delta = compose_tensor(iso_c, skew_c, traceless_c)
    # Residual uses the normalized state. The square is charge-scaled for both groups.
    return normalized + delta + scale * tensor_matmul(delta, delta)


class TensorNetInteraction(Module):
    """One normalized tensor update (paper Eqs. 11 and the residual polynomial)."""

    def __init__(self, dim_embedding: int, num_rbf: int, equivariance_group: str) -> None:
        super().__init__()
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.equivariance_group = equivariance_group
        hidden = self.dim_embedding
        self.linears_scalar = ModuleList(
            [
                Linear(self.num_rbf, hidden, bias=True),
                Linear(hidden, 2 * hidden, bias=True),
                Linear(2 * hidden, 3 * hidden, bias=True),
            ]
        )
        self.linears_tensor = ModuleList(
            [Linear(hidden, hidden, bias=False) for _ in range(6)]
        )

    def forward(
        self,
        features: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        window: Tensor,
        charge_scale: Tensor,
    ) -> Tensor:
        edge = scalar_edge_gates(rbf, self.linears_scalar, window, self.dim_embedding)
        return tensor_message_update(
            features,
            edge,
            idx_i,
            idx_j,
            self.linears_tensor,
            self.equivariance_group,
            charge_scale,
        )


class TensorNet2Interaction(Module):
    """TensorNet update whose radial MLP also reads ``q_i`` and ``q_j``.

    Farr et al. (arXiv:2601.00581) Eq. 2. The total-charge scale used by
    TensorNet is absent: both the O(3) product and the residual square use a
    factor of one.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        q_dim: int,
        equivariance_group: str,
    ) -> None:
        super().__init__()
        if int(q_dim) < 1:
            raise ValueError(f"q_dim must be >= 1, got {q_dim}")
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.q_dim = int(q_dim)
        self.equivariance_group = equivariance_group
        hidden = self.dim_embedding
        self.linears_scalar = ModuleList(
            [
                Linear(self.num_rbf + 2 * self.q_dim, hidden, bias=True),
                Linear(hidden, 2 * hidden, bias=True),
                Linear(2 * hidden, 3 * hidden, bias=True),
            ]
        )
        self.linears_tensor = ModuleList(
            [Linear(hidden, hidden, bias=False) for _ in range(6)]
        )

    def forward(
        self,
        features: Tensor,
        charges: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf: Tensor,
        window: Tensor,
    ) -> Tensor:
        if charges.ndim != 2 or charges.shape[-1] != self.q_dim:
            raise ValueError(
                f"charges must have shape [N, {self.q_dim}], got {tuple(charges.shape)}"
            )
        if idx_i.numel() == 0:
            edge_input = rbf.new_zeros((0, self.num_rbf + 2 * self.q_dim))
        else:
            edge_input = torch.cat((rbf, charges[idx_i], charges[idx_j]), dim=-1)
        edge = scalar_edge_gates(
            edge_input, self.linears_scalar, window, self.dim_embedding
        )
        ones = features.new_ones(features.shape[0])
        return tensor_message_update(
            features,
            edge,
            idx_i,
            idx_j,
            self.linears_tensor,
            self.equivariance_group,
            ones,
        )
