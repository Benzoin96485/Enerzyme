from abc import ABC, abstractmethod
import math
from typing import Dict, Optional
import torch
from torch import Tensor
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss
from torch.nn import functional as F


class WeightedLoss(ABC):
    def __init__(self, **weights: Dict[str, float]) -> None:
        self.weights = weights

    @abstractmethod
    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        ...

    def __call__(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        for k, v in self.weights.items():
            if output[k].dim() == target[k].dim() + 1:
                target[k] = target[k].unsqueeze(-1).expand_as(output[k])
                loss_term = self.loss_fn(output, target, k)
                target[k] = target[k].narrow(-1, 0, 1).squeeze(-1)
            else:
                loss_term = self.loss_fn(output, target, k)
            loss = loss + v * loss_term
        return loss


class MSELoss(WeightedLoss):
    def __init__(self, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.mseloss = MSELoss_()

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return self.mseloss(output[k], target[k])


class RMSELoss(WeightedLoss):
    def __init__(self, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.mseloss = MSELoss_()
        self.eps = 1e-10

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return torch.sqrt(self.mseloss(output[k], target[k]) + self.eps)


class MAELoss(WeightedLoss):
    def __init__(self, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.maeloss = L1Loss()

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return self.maeloss(output[k], target[k])


class NLLLoss(WeightedLoss):
    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.eps = eps

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return 0.5 * torch.mean(torch.log(torch.clamp(output[k + "_var"], self.eps, 1)) + (output[k] - target[k]) ** 2 / torch.clamp(output[k + "_var"], self.eps, 1))


class NLLLossVarOnly(WeightedLoss):
    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.eps = eps

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        with torch.no_grad():
            mse = (output[k] - target[k]) ** 2
        return 0.5 * torch.mean(torch.log(output[k + "_var"] + self.eps) + mse / (output[k + "_var"] + self.eps))


class CRPSLoss(WeightedLoss):
    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        from torch.distributions.normal import Normal
        super().__init__(**weights)
        self.normal = Normal(0, 1)
        self.eps = eps
        self.phi = lambda x: self.normal.log_prob(x).exp()
        self.Phi = self.normal.cdf
        self.sqrt_pi = math.sqrt(math.pi)

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        if k + "_std" in output:
            std = output[k + "_std"] + self.eps
        elif k + "_var" in output:
            std = torch.sqrt(output[k + "_var"] + self.eps) 
        dev = (output[k] - target[k]) / std
        return torch.mean(std * (dev * (2 * self.Phi(dev) - 1) + 2 * self.phi(dev) - 1 / self.sqrt_pi))


class L2Penalty:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        return output.get("l2_penalty", 0) * self.weight


class CFMLoss:
    """Conditional flow matching loss: match per-atom velocities to label minus init.

    Minimizes ``weight_q * mean((v_q - (Qa - Q_init))^2) + weight_s * mean((v_s - (Sa - S_init))^2)``.
    With Generator batch decoration, ``Q_init_a`` / ``S_init_a`` are also present on ``target``.
    """

    def __init__(
        self,
        weight_q: float = 1.0,
        weight_s: float = 1.0,
        vel_q_key: str = "Q_vel_a",
        vel_s_key: str = "S_vel_a",
        target_q_key: str = "Qa",
        target_s_key: str = "Sa",
        init_q_key: str = "Q_init_a",
        init_s_key: str = "S_init_a",
    ) -> None:
        self.weight_q = float(weight_q)
        self.weight_s = float(weight_s)
        self.vel_q_key = vel_q_key
        self.vel_s_key = vel_s_key
        self.target_q_key = target_q_key
        self.target_s_key = target_s_key
        self.init_q_key = init_q_key
        self.init_s_key = init_s_key

    def __call__(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.vel_q_key in output and self.vel_s_key in output:
            loss = output[self.vel_q_key].new_zeros(())
            if self.weight_q != 0.0:
                dq = target[self.target_q_key] - target[self.init_q_key]
                loss = loss + self.weight_q * (output[self.vel_q_key] - dq).pow(2).mean()
            if self.weight_s != 0.0:
                ds = target[self.target_s_key] - target[self.init_s_key]
                loss = loss + self.weight_s * (output[self.vel_s_key] - ds).pow(2).mean()
            return loss
        if self.target_q_key in output and self.target_s_key in output:
            loss = output[self.target_q_key].new_zeros(())
            if self.weight_q != 0.0:
                loss = loss + self.weight_q * (
                    output[self.target_q_key] - target[self.target_q_key]
                ).pow(2).mean()
            if self.weight_s != 0.0:
                loss = loss + self.weight_s * (
                    output[self.target_s_key] - target[self.target_s_key]
                ).pow(2).mean()
            return loss
        raise KeyError(
            f"CFMLoss expected ({self.vel_q_key}, {self.vel_s_key}) or "
            f"({self.target_q_key}, {self.target_s_key}) in output, got keys {set(output.keys())}"
        )


class HuberLoss(WeightedLoss):
    """Element-wise Huber loss with per-target weights. ``delta`` is not a target key."""

    def __init__(self, delta: float = 1.0, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.delta = float(delta)

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return F.huber_loss(output[k], target[k], reduction="mean", delta=self.delta)


def _atom_counts(
    output: Dict[str, Tensor],
    target: Dict[str, Tensor],
    n_graphs: int,
    energy: Tensor,
) -> Tensor:
    """Per-graph atom counts for per-atom energy standardization in the loss only."""
    batch_seg = None
    if "batch_seg" in output and output["batch_seg"] is not None:
        batch_seg = output["batch_seg"]
    elif hasattr(output, "batch") and getattr(output, "batch", None) is not None:
        batch_seg = output.batch
    if batch_seg is not None:
        counts = torch.bincount(batch_seg, minlength=n_graphs)
        return counts.to(device=energy.device, dtype=energy.dtype).clamp_min(1)
    n_raw = output.get("N", None)
    if n_raw is None:
        n_raw = target.get("N", None)
    if n_raw is None:
        raise KeyError(
            "ResidualHuberLoss needs batch_seg or N to standardize energy per atom"
        )
    if not torch.is_tensor(n_raw):
        n_raw = torch.as_tensor(n_raw, device=energy.device)
    counts = n_raw.reshape(-1).to(device=energy.device, dtype=energy.dtype)
    if counts.numel() == 1 and n_graphs > 1:
        counts = counts.expand(n_graphs)
    if counts.numel() != n_graphs:
        raise ValueError(
            f"N length {counts.numel()} does not match energy graphs {n_graphs}"
        )
    return counts.clamp_min(1)


class ResidualHuberLoss:
    """Huber residual loss with per-atom energy and force-component RMS scales.

    Model energies must stay in physical units; this class is the only place
    that divides graph energy error by ``N`` and force error by the RMS of
    target force components. Prefer ``dE`` / ``dFa`` so eval totals on ``E`` /
    ``Fa`` do not leak into the training objective.
    """

    _ENERGY_KEYS = ("dE", "E")
    _FORCE_KEYS = ("dFa", "Fa")

    def __init__(
        self,
        delta: float = 0.01,
        per_atom_energy: bool = True,
        force_rms: Optional[float] = None,
        eps: float = 1e-8,
        **weights: Dict[str, float],
    ) -> None:
        self.delta = float(delta)
        self.per_atom_energy = bool(per_atom_energy)
        self.force_rms = None if force_rms is None else float(force_rms)
        self.eps = float(eps)
        self.weights = dict(weights)
        if not self.weights:
            self.weights = {"dE": 1.0, "dFa": 1.0}

    def _huber(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(pred, target, reduction="mean", delta=self.delta)

    def _energy_term(self, output: Dict[str, Tensor], target: Dict[str, Tensor], k: str) -> Tensor:
        pred = output[k]
        truth = target[k]
        if pred.dim() > 1:
            pred = pred.reshape(pred.shape[0], -1).sum(dim=-1)
        if truth.dim() > 1:
            truth = truth.reshape(truth.shape[0], -1).sum(dim=-1)
        if self.per_atom_energy:
            counts = _atom_counts(output, target, pred.shape[0], pred)
            pred = pred / counts
            truth = truth / counts
        return self._huber(pred, truth)

    def _force_term(self, output: Dict[str, Tensor], target: Dict[str, Tensor], k: str) -> Tensor:
        pred = output[k]
        truth = target[k]
        if self.force_rms is None:
            rms = truth.detach().pow(2).mean().sqrt().clamp_min(self.eps)
        else:
            rms = pred.new_tensor(self.force_rms).clamp_min(self.eps)
        return self._huber(pred / rms, truth / rms)

    def __call__(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        loss = None
        for k, weight in self.weights.items():
            if weight == 0:
                continue
            if k not in output or k not in target:
                continue
            if k in self._FORCE_KEYS or k.endswith("Fa"):
                term = self._force_term(output, target, k)
            else:
                term = self._energy_term(output, target, k)
            loss = term * weight if loss is None else loss + term * weight
        if loss is None:
            raise KeyError(
                "ResidualHuberLoss found none of "
                f"{list(self.weights)} in both output and target "
                f"(output={set(output.keys())}, target={set(target.keys())})"
            )
        return loss


LOSS_REGISTER = {
    "mae": MAELoss,
    "mse": MSELoss,
    "rmse": RMSELoss,
    "nll": NLLLoss,
    "nll_var_only": NLLLossVarOnly,
    "l2_penalty": L2Penalty,
    "crps": CRPSLoss,
    "cfm": CFMLoss,
    "huber": HuberLoss,
    "residual_huber": ResidualHuberLoss,
}
