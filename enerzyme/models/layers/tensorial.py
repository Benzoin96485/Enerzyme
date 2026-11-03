"""Architecture-independent Cartesian tensor heads (PaiNN Sec. IV.2).

Any Core that emits ``atom_feature`` ``[N, F]`` and ``atom_vector_feature``
``[N, 3, F]`` can use these layers. They are not PaiNN-specific.

* :class:`EquivariantDipoleLayer` — Eq. 13, latent charges plus atomic dipoles.
  Charge-only dipoles stay on ``SimpleReadout(Qa)`` + ``AtomicCharge2Dipole``.
* :class:`EquivariantPolarizabilityLayer` — Eq. 14, isotropic scalar plus
  ``ν ⊗ r + r ⊗ ν``.

Both follow SchNetPack 2.0.4 ``DipoleMoment`` / ``Polarizability`` (MIT),
including the gated MLP width schedule. Coordinates are used as stored.
SchNetPack's spectra configs subtract the center of mass in the data
pipeline; do that before these layers when the reference frame should be
the molecular center of mass. A neutral corrected charge makes ``M2``
independent of origin.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..functional import segment_sum_coo
from . import BaseFFLayer
from .gated_equivariant import build_gated_equivariant_mlp


def resolve_cartesian_feature_dim(
    built_layers: Optional[List[Module]],
    dim_embedding: Optional[int],
) -> int:
    """Width F of a preceding Cartesian scalar/vector Core, else ``dim_embedding``."""
    for layer in reversed(list(built_layers or [])):
        fields = getattr(layer, "_output_fields", None) or set()
        width = getattr(layer, "dim_feature_out", None)
        if "atom_vector_feature" in fields and width:
            width = int(width)
            if dim_embedding is not None and int(dim_embedding) != width:
                raise ValueError(
                    f"dim_embedding ({dim_embedding}) != Cartesian feature width ({width})"
                )
            return width
    if dim_embedding is None:
        raise ValueError(
            "dim_embedding is required when no layer emits atom_vector_feature"
        )
    return int(dim_embedding)


def _as_batch_index(batch_seg: Optional[Tensor], n_atoms: int, device: torch.device) -> Tensor:
    if batch_seg is None:
        return torch.zeros(n_atoms, dtype=torch.long, device=device)
    return batch_seg.long()


def correct_partial_charges(
    charges: Tensor,
    batch_seg: Tensor,
    total_charge: Optional[Tensor],
) -> Tensor:
    """Shift latent charges so each molecule sums to ``total_charge`` (or 0)."""
    ones = torch.ones_like(charges)
    n_per = segment_sum_coo(ones, batch_seg).clamp_min(1)
    sum_q = segment_sum_coo(charges, batch_seg)
    if total_charge is None:
        target = torch.zeros_like(sum_q)
    else:
        target = total_charge.reshape(sum_q.shape).to(dtype=charges.dtype)
    delta = (target - sum_q) / n_per
    return charges + delta[batch_seg]


def _latent_pair(
    atom_feature: Tensor,
    atom_vector_feature: Tensor,
    n_in: int,
) -> tuple[Tensor, Tensor]:
    if atom_feature.ndim != 2:
        raise ValueError(
            f"atom_feature must have shape [N, F], got {tuple(atom_feature.shape)}"
        )
    if atom_vector_feature.ndim != 3 or atom_vector_feature.shape[-2] != 3:
        raise ValueError(
            "atom_vector_feature must have shape [N, 3, F] "
            f"(Cartesian, channel-last), got {tuple(atom_vector_feature.shape)}"
        )
    if atom_feature.shape[0] != atom_vector_feature.shape[0]:
        raise ValueError("scalar and vector features cover different atoms")
    if atom_feature.shape[-1] != n_in or atom_vector_feature.shape[-1] != n_in:
        raise ValueError(
            f"expected feature width {n_in}, got scalar {atom_feature.shape[-1]} "
            f"and vector {atom_vector_feature.shape[-1]}"
        )
    return atom_feature, atom_vector_feature


class EquivariantDipoleLayer(BaseFFLayer):
    """Molecular dipole from latent charges and atomic dipoles.

    ``return_charges`` writes the (optionally corrected) latent charges to
    ``Qa``. That overwrites a charge head earlier in the stack. Leave it
    false when electrostatics should keep a separate ``Qa``.
    ``predict_magnitude`` also writes the scalar ``M2mag = ||M2||``.
    """

    def __init__(
        self,
        built_layers: Optional[List[Module]] = None,
        dim_embedding: Optional[int] = None,
        n_layers: int = 2,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        correct_charges: bool = True,
        return_charges: bool = False,
        predict_magnitude: bool = False,
    ) -> None:
        if built_layers is None:
            built_layers = []
        output_fields = {"M2"}
        if return_charges:
            output_fields.add("Qa")
        if predict_magnitude:
            output_fields.add("M2mag")
        super().__init__(
            input_fields={"atom_feature", "atom_vector_feature", "Ra", "batch_seg", "Q"},
            output_fields=output_fields,
        )
        self.n_in = resolve_cartesian_feature_dim(built_layers, dim_embedding)
        self.n_layers = int(n_layers)
        self.correct_charges = bool(correct_charges)
        self.return_charges = bool(return_charges)
        self.predict_magnitude = bool(predict_magnitude)
        self.outnet = build_gated_equivariant_mlp(
            n_in=self.n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=self.n_layers,
        )

    def get_output(
        self,
        atom_feature: Tensor,
        atom_vector_feature: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        q, v = _latent_pair(atom_feature, atom_vector_feature, self.n_in)
        scalars, vectors = self.outnet((q, v))
        charges = scalars.squeeze(-1)
        atomic_dipoles = vectors.squeeze(-1)
        batch_seg = _as_batch_index(batch_seg, charges.shape[0], charges.device)
        if self.correct_charges:
            charges = correct_partial_charges(charges, batch_seg, Q)
        moment = Ra.to(dtype=charges.dtype) * charges.unsqueeze(-1) + atomic_dipoles
        dipole = segment_sum_coo(moment, batch_seg)
        out: Dict[str, Tensor] = {"M2": dipole}
        if self.return_charges:
            out["Qa"] = charges
        if self.predict_magnitude:
            out["M2mag"] = torch.linalg.vector_norm(dipole, dim=-1)
        return out


class EquivariantPolarizabilityLayer(BaseFFLayer):
    """Molecular polarizability ``[batch, 3, 3]`` from scalar and vector features.

    The isotropic part is ``α0 I``. The anisotropic part is
    ``ν ⊗ R + R ⊗ ν`` summed over atoms. The tensor uses the coordinate
    origin of ``Ra``.
    """

    def __init__(
        self,
        built_layers: Optional[List[Module]] = None,
        dim_embedding: Optional[int] = None,
        n_layers: int = 2,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        if built_layers is None:
            built_layers = []
        super().__init__(
            input_fields={"atom_feature", "atom_vector_feature", "Ra", "batch_seg"},
            output_fields={"polarizability"},
        )
        self.n_in = resolve_cartesian_feature_dim(built_layers, dim_embedding)
        self.n_layers = int(n_layers)
        self.outnet = build_gated_equivariant_mlp(
            n_in=self.n_in,
            n_out=1,
            n_hidden=n_hidden,
            n_layers=self.n_layers,
        )

    def get_output(
        self,
        atom_feature: Tensor,
        atom_vector_feature: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        q, v = _latent_pair(atom_feature, atom_vector_feature, self.n_in)
        scalars, vectors = self.outnet((q, v))
        alpha0 = scalars.expand(-1, 3)
        alpha = torch.diag_embed(alpha0)
        nu = vectors.squeeze(-1)
        positions = Ra.to(dtype=nu.dtype)
        outer = nu.unsqueeze(-1) * positions.unsqueeze(-2)
        alpha = alpha + outer + outer.transpose(-1, -2)
        batch_seg = _as_batch_index(batch_seg, alpha.shape[0], alpha.device)
        return {"polarizability": segment_sum_coo(alpha, batch_seg)}
