"""Derivative polarizability for a scalar atomic latent.

Zhang, Xia, and Jiang (J. Chem. Phys. 2022, Eqs. 9–11) build a symmetric
polarizability from derivatives of scalar outputs of an invariant network.
The layer is not REANN-specific: any Core that emits ``atom_feature`` ``[N, F]``
can use it. ``Ra`` must require gradients, including at inference, because the
tensor is a coordinate derivative of the latent scalars.

``variant="paper"`` uses one latent energy for both the rank-1 term and the
position term, plus a second scalar on the diagonal:

.. math::

    \\alpha = g g^T + r g^T + g r^T + s I

``variant="repo"`` is ``Property_POL`` in https://github.com/zhangylch/REANN:
two latent energies and a third scalar,

.. math::

    \\alpha = \\mathrm{sym}\\big((g_1 + r) g_2^T\\big) + s I

with ``sym(M) = M + M^T``. Coordinates are used as stored. A neutral origin
has to be applied to ``Ra`` before this layer when the reference frame is the
center of mass. ``EquivariantPolarizability`` remains the head for Cores that
already emit Cartesian vector features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.autograd import grad
from torch.nn import Module, Sequential

from ..blocks.mlp import DenseLayer
from ..functional import segment_sum_coo
from ._base_layer import BaseFFLayer


def _scalar_width(built_layers: Optional[List[Module]], fallback: Optional[int]) -> int:
    for layer in reversed(list(built_layers or [])):
        fields = getattr(layer, "_output_fields", None) or set()
        width = getattr(layer, "dim_feature_out", None)
        if "atom_feature" not in fields or not width:
            continue
        if "Ea" in fields or "Qa" in fields:
            continue
        return int(width)
    if fallback is None:
        raise ValueError(
            "dim_embedding is required when no preceding layer emits atom_feature"
        )
    return int(fallback)


def _latent_head(
    n_in: int,
    n_out: int,
    hidden_dims: Sequence[int],
    activation_fn: str,
) -> Sequential:
    modules = []
    previous = int(n_in)
    for width in hidden_dims:
        width = int(width)
        if width < 1:
            raise ValueError(f"hidden width must be >= 1, got {width}")
        modules.append(
            DenseLayer(
                previous,
                width,
                activation_fn=activation_fn,
                activation_params={
                    "dim_feature": width,
                    "initial_alpha": 1.0,
                    "initial_beta": 1.0,
                    "learnable": False,
                },
                initial_weight="xavier_uniform",
            )
        )
        previous = width
    modules.append(
        DenseLayer(
            previous,
            int(n_out),
            initial_weight="zero",
            initial_bias="zero",
        )
    )
    return Sequential(*modules)


def _batch_index(batch_seg: Optional[Tensor], n_atoms: int, device: torch.device) -> Tensor:
    if batch_seg is None:
        return torch.zeros(n_atoms, dtype=torch.long, device=device)
    return batch_seg.reshape(-1).long()


def _coordinate_gradient(molecular: Tensor, positions: Tensor, create_graph: bool) -> Tensor:
    coordinate_grad = grad(
        molecular.sum(),
        positions,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if coordinate_grad is None:
        return torch.zeros_like(positions)
    return coordinate_grad


def assemble_polarizability(
    positions: Tensor,
    grad_a: Tensor,
    grad_b: Tensor,
    isotropic: Tensor,
    batch_seg: Tensor,
    variant: str,
) -> Tensor:
    """Sum per-atom contributions into ``[n_graph, 3, 3]``.

    ``isotropic`` is an atomic scalar. Its sum is the molecular diagonal shift.
    """
    if positions.shape != grad_a.shape or positions.shape != grad_b.shape:
        raise ValueError(
            "positions, grad_a, and grad_b must share shape [N, 3], "
            f"got {tuple(positions.shape)}, {tuple(grad_a.shape)}, {tuple(grad_b.shape)}"
        )
    n_graph = int(batch_seg.max()) + 1 if batch_seg.numel() else 1
    if variant == "paper":
        rank1 = grad_a.unsqueeze(-1) * grad_a.unsqueeze(-2)
        cross = positions.unsqueeze(-1) * grad_a.unsqueeze(-2)
        atomic = rank1 + cross + cross.transpose(-1, -2)
    elif variant == "repo":
        left = grad_a + positions
        mixed = left.unsqueeze(-1) * grad_b.unsqueeze(-2)
        atomic = mixed + mixed.transpose(-1, -2)
    else:
        raise ValueError(f"variant must be 'paper' or 'repo', got {variant!r}")
    alpha = segment_sum_coo(atomic, batch_seg, dim_size=n_graph)
    scalar = segment_sum_coo(isotropic.reshape(-1), batch_seg, dim_size=n_graph)
    eye = torch.eye(3, device=alpha.device, dtype=alpha.dtype)
    return alpha + scalar.reshape(-1, 1, 1) * eye


class DerivativePolarizabilityLayer(BaseFFLayer):
    """Molecular polarizability from derivatives of scalar latents."""

    def __init__(
        self,
        built_layers: Optional[List[Module]] = None,
        dim_embedding: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
        activation_fn: str = "swish",
        variant: str = "paper",
    ) -> None:
        variant_key = str(variant).lower()
        if variant_key not in {"paper", "repo"}:
            raise ValueError(f"variant must be 'paper' or 'repo', got {variant!r}")
        super().__init__(
            input_fields={"atom_feature", "Ra", "batch_seg"},
            output_fields={"polarizability"},
        )
        self.variant = variant_key
        self.n_in = _scalar_width(built_layers, dim_embedding)
        self.n_channels = 2 if self.variant == "paper" else 3
        self.head = _latent_head(
            self.n_in,
            self.n_channels,
            list(hidden_dims or []),
            activation_fn,
        )

    def get_output(
        self,
        atom_feature: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if atom_feature.ndim != 2 or atom_feature.shape[-1] != self.n_in:
            raise ValueError(
                f"atom_feature must have shape [N, {self.n_in}], got {tuple(atom_feature.shape)}"
            )
        if Ra.ndim != 2 or Ra.shape[-1] != 3 or Ra.shape[0] != atom_feature.shape[0]:
            raise ValueError(
                f"Ra must have shape [{atom_feature.shape[0]}, 3], got {tuple(Ra.shape)}"
            )
        if not Ra.requires_grad:
            raise ValueError(
                "DerivativePolarizability differentiates latent scalars with respect "
                "to Ra. Set Ra.requires_grad_(True) before the forward, including at inference."
            )
        latent = self.head(atom_feature)
        batch = _batch_index(batch_seg, latent.shape[0], latent.device)
        if batch.shape[0] != latent.shape[0]:
            raise ValueError(
                f"batch_seg has length {batch.shape[0]} but there are {latent.shape[0]} atoms"
            )
        create_graph = bool(self.training)
        if self.variant == "paper":
            energy = segment_sum_coo(latent[:, 0], batch)
            grad_a = _coordinate_gradient(energy, Ra, create_graph)
            grad_b = grad_a
            isotropic = latent[:, 1]
        else:
            energy_a = segment_sum_coo(latent[:, 0], batch)
            energy_b = segment_sum_coo(latent[:, 1], batch)
            grad_a = _coordinate_gradient(energy_a, Ra, create_graph)
            grad_b = _coordinate_gradient(energy_b, Ra, create_graph)
            isotropic = latent[:, 2]
        alpha = assemble_polarizability(Ra, grad_a, grad_b, isotropic, batch, self.variant)
        return {"polarizability": alpha}
