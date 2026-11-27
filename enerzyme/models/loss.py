from abc import ABC, abstractmethod
import math
from typing import Dict, NamedTuple, Optional
import torch
import torch.distributed as dist
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss


class LossComponent(NamedTuple):
    """One target-wise reduction before its weight and optional RMSE transform."""
    mean: torch.Tensor
    count: int
    weight: float
    root_eps: Optional[float] = None

    def weighted(self, mean):
        if self.root_eps is not None:
            mean = (mean + self.root_eps).sqrt()
        return mean * self.weight


class WeightedLoss(ABC):
    root_eps = None
    uncertainty_suffixes = ()

    def __init__(self, **weights: Dict[str, float]) -> None:
        self.weights = weights

    @abstractmethod
    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        ...

    def _mean(self, output, target, key):
        return self.loss_fn(output, target, key)

    def _targets(self, output, target, key):
        if output[key].ndim == target[key].ndim + 1:
            # Train each ensemble member against the same reference.
            return {**target, key: target[key].unsqueeze(-1).expand_as(output[key])}
        return target

    def __call__(self, output, target):
        return sum(weight * self.loss_fn(output, self._targets(output, target, key), key)
                   for key, weight in self.weights.items())

    def components(self, output, target):
        for key, weight in self.weights.items():
            labels = self._targets(output, target, key)
            shapes = [output[key].shape, labels[key].shape]
            for suffix in self.uncertainty_suffixes:
                if key + suffix in output:
                    shapes.append(output[key + suffix].shape)
                    break
            count = (output[key].numel() if all(s == shapes[0] for s in shapes)
                     else math.prod(torch.broadcast_shapes(*shapes)))
            mean = self._mean(output, labels, key) if count else output[key].sum() * 0
            yield LossComponent(mean, count, weight, self.root_eps)


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
        self.root_eps = self.eps

    def _mean(self, output, target, key):
        return self.mseloss(output[key], target[key])

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return torch.sqrt(self.mseloss(output[k], target[k]) + self.eps)


class MAELoss(WeightedLoss):
    def __init__(self, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.maeloss = L1Loss()

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return self.maeloss(output[k], target[k])


class NLLLoss(WeightedLoss):
    uncertainty_suffixes = ("_var",)

    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.eps = eps

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return 0.5 * torch.mean(torch.log(torch.clamp(output[k + "_var"], self.eps, 1)) + (output[k] - target[k]) ** 2 / torch.clamp(output[k + "_var"], self.eps, 1))


class NLLLossVarOnly(WeightedLoss):
    uncertainty_suffixes = ("_var",)

    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        super().__init__(**weights)
        self.eps = eps

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        with torch.no_grad():
            mse = (output[k] - target[k]) ** 2
        return 0.5 * torch.mean(torch.log(output[k + "_var"] + self.eps) + mse / (output[k + "_var"] + self.eps))


class CRPSLoss(WeightedLoss):
    uncertainty_suffixes = ("_std", "_var")

    def __init__(self, eps: float = 1e-6, **weights: Dict[str, float]) -> None:
        from torch.distributions.normal import Normal
        super().__init__(**weights)
        self.normal = Normal(0, 1)
        self.eps = eps
        self.sqrt_pi = math.sqrt(math.pi)

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        if k + "_std" in output:
            std = output[k + "_std"] + self.eps
        elif k + "_var" in output:
            std = torch.sqrt(output[k + "_var"] + self.eps)
        dev = (output[k] - target[k]) / std
        return torch.mean(std * (dev * (2 * self.normal.cdf(dev) - 1) + 2 * self.normal.log_prob(dev).exp() - 1 / self.sqrt_pi))


class L2Penalty:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        return output.get("l2_penalty", 0) * self.weight

    def components(self, output, target):
        # A model parameter regularizer is replicated, even on unequal atom counts.
        value = output.get("l2_penalty")
        if value is None:
            value = next(t for t in output.values() if isinstance(t, torch.Tensor)).new_zeros(())
        yield LossComponent(value, 1, self.weight)


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

    def components(self, output, target):
        use_velocity = self.vel_q_key in output and self.vel_s_key in output
        if not use_velocity and not (self.target_q_key in output and self.target_s_key in output):
            raise KeyError(
                f"CFMLoss expected ({self.vel_q_key}, {self.vel_s_key}) or "
                f"({self.target_q_key}, {self.target_s_key}) in output, got keys {set(output.keys())}"
            )
        for weight, vel_key, target_key, init_key in (
            (self.weight_q, self.vel_q_key, self.target_q_key, self.init_q_key),
            (self.weight_s, self.vel_s_key, self.target_s_key, self.init_s_key),
        ):
            if weight == 0:
                continue
            label = target[target_key]
            if use_velocity:
                label = label - target[init_key]
            diff = output[vel_key if use_velocity else target_key] - label
            yield LossComponent(diff.square().mean() if diff.numel() else diff.sum(), diff.numel(), weight)

    def __call__(self, output, target):
        reference = output[self.vel_q_key] if self.vel_q_key in output else output[self.target_q_key]
        return sum((part.weighted(part.mean) for part in self.components(output, target)),
                   reference.new_zeros(()))


def _loss_components(loss_terms, output, target):
    for term in loss_terms.values():
        if hasattr(term, "components"):
            yield from term.components(output, target)
        else:
            # Existing custom callables can still run on one process. DDP needs
            # their actual denominator to reproduce the full-batch objective.
            if dist.is_initialized() and dist.get_world_size() > 1:
                raise TypeError(f"{type(term).__name__} must define components() for distributed training")
            yield LossComponent(term(output, target), 1, 1.0)


def _statistics(parts):
    means = torch.stack([part.mean for part in parts])
    counts = means.new_tensor([part.count for part in parts], dtype=torch.float64)
    totals = means.detach().to(torch.float64) * counts
    return means, counts, torch.stack((totals, counts))


def training_loss(loss_terms, output, target):
    """Full-batch objective, with rank gradients scaled for DDP's averaging."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return sum(term(output, target) for term in loss_terms.values())
    parts = list(_loss_components(loss_terms, output, target))
    if not parts:
        return next(t for t in output.values() if isinstance(t, torch.Tensor)).sum() * 0
    local_means, local_counts, stats = _statistics(parts)
    dist.all_reduce(stats)
    totals, counts = stats
    counts = counts.clamp_min(1)
    global_means = (totals / counts).to(local_means.dtype)
    scales = (dist.get_world_size() * local_counts / counts).to(local_means.dtype)
    # The collective carries values only. Local autograd contributes the
    # appropriate fraction of the global derivative, including sqrt(MSE).
    means = global_means + (local_means - local_means.detach()) * scales
    return sum(part.weighted(mean) for part, mean in zip(parts, means.unbind()))


class EvaluationLoss:
    """Pool target-wise sufficient statistics once after all evaluation batches."""

    def __init__(self, loss_terms):
        self.loss_terms = loss_terms
        self.stats = None
        self.transforms = []

    @torch.no_grad()
    def update(self, output, target):
        parts = list(_loss_components(self.loss_terms, output, target))
        if not parts:
            return 0.0
        _, _, stats = _statistics(parts)
        if self.stats is None:
            self.stats = stats
            self.transforms = [(p.weight, p.root_eps) for p in parts]
        else:
            self.stats += stats
        return sum(p.weighted(p.mean) for p in parts).item()

    def compute(self, device=None):
        packet = (self.stats.cpu(), self.transforms) if self.stats is not None else None
        packets = [packet]
        if dist.is_initialized() and dist.get_world_size() > 1:
            packets = [None] * dist.get_world_size()
            dist.all_gather_object(packets, packet)
        nonempty = [p for p in packets if p is not None]
        if not nonempty:
            return 0.0
        totals, counts = sum(p[0] for p in nonempty)
        result = 0.0
        for mean, (weight, eps) in zip((totals / counts.clamp_min(1)).tolist(), nonempty[0][1]):
            result += weight * (math.sqrt(mean + eps) if eps is not None else mean)
        return result


LOSS_REGISTER = {
    "mae": MAELoss,
    "mse": MSELoss,
    "rmse": RMSELoss,
    "nll": NLLLoss,
    "nll_var_only": NLLLossVarOnly,
    "l2_penalty": L2Penalty,
    "crps": CRPSLoss,
    "cfm": CFMLoss,
}
