"""Direction loss on NewtonNet's latent force.

The paper adds ``1 - cos(F_latent, F_ref)`` to the energy and force losses.
Conservative MD forces stay ``Fa = -∇E`` from the shared ``Force`` layer.
This term only trains the latent vector the Core already emits.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from ..loss import LossComponent


def latent_force_cosine(pred: Tensor, ref: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean ``1 - cos`` between ``[N, 3]`` latent forces and reference forces.

    An atom whose latent force and reference force are both near zero
    contributes 0. The mean runs over atoms in the batch.
    """
    if pred.shape != ref.shape or pred.ndim != 2 or pred.shape[-1] != 3:
        raise ValueError(
            "latent force and Fa must both have shape [N, 3], "
            f"got {tuple(pred.shape)} and {tuple(ref.shape)}"
        )
    pred = pred.to(dtype=ref.dtype)
    dot = torch.sum(pred * ref, dim=-1)
    pred_norm = torch.linalg.vector_norm(pred, dim=-1)
    ref_norm = torch.linalg.vector_norm(ref, dim=-1)
    cosine = dot / (pred_norm.clamp_min(eps) * ref_norm.clamp_min(eps))
    cosine = cosine.clamp(-1.0, 1.0)
    both_zero = (pred_norm < eps) & (ref_norm < eps)
    term = torch.where(both_zero, torch.zeros_like(cosine), 1.0 - cosine)
    return term.mean()


class LatentForceCosineLoss:
    """Paper direction penalty. YAML key ``latent_force_cosine``, weight ``λ_D``."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-8) -> None:
        self.weight = float(weight)
        self.eps = float(eps)

    def components(self, output, target):
        if "latent_force" not in output:
            raise KeyError("latent_force_cosine expects output['latent_force'] from the NewtonNet preset")
        if "Fa" not in target:
            raise KeyError("latent_force_cosine expects reference forces target['Fa']")
        pred, ref = output["latent_force"], target["Fa"]
        mean = latent_force_cosine(pred, ref, self.eps) if len(ref) else pred.sum()
        yield LossComponent(mean, len(ref), self.weight)

    def __call__(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        component, = self.components(output, target)
        return self.weight * component.mean


LOSS_REGISTER = {
    "latent_force_cosine": LatentForceCosineLoss,
}
