from abc import ABC, abstractmethod
import math
from typing import Dict
import torch
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss


class WeightedLoss(ABC):
    def __init__(self, adaptive: bool=False, eps: float=1e-6, beta: float=1, **weights: Dict[str, float]) -> None:
        self.initial_weights = weights
        self.weights = weights.copy()
        self.prefactor = sum(self.initial_weights.values())
        self.adaptive = adaptive
        self.last_loss = {k: 0. for k in weights.keys()}
        self.current_loss = {k: 0. for k in weights.keys()}
        self.beta = 1
        self.eps = eps
        self.recorded = False

    @abstractmethod
    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        ...

    def record(self) -> None:
        self.recorded = True
        self.current_loss = {k: 0. for k in self.weights.keys()}

    def __call__(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        for k, v in self.weights.items():
            if output[k].dim() == target[k].dim() + 1:
                target[k] = target[k].unsqueeze(-1).expand_as(output[k])
                loss_term = self.loss_fn(output, target, k)
                target[k] = target[k].narrow(-1, 0, 1).squeeze(-1)
            else:
                loss_term = self.loss_fn(output, target, k)
            if self.adaptive and self.recorded:
                self.current_loss[k] += loss_term.item()
            loss = loss + v * loss_term
        
        return loss
        
    def update(self) -> None:
        if self.adaptive and self.recorded:
            loss_change = {k: self.current_loss[k] - self.last_loss[k] for k in self.weights.keys()}
            sum_abs_loss_change = sum(abs(v) for v in loss_change.values())
            loss_change_ratio = {k: loss_change[k] / (sum_abs_loss_change + self.eps) for k in self.weights.keys()}
            raw_weights = {k: math.exp(self.beta * loss_change_ratio[k]) for k in self.weights.keys()}
            sum_raw_weights = sum(raw_weights.values()) / len(self.weights.keys())
            normalized_weights = {k: raw_weights[k] / sum_raw_weights for k in self.weights.keys()}
            self.weights = {k: self.initial_weights[k] * normalized_weights[k] for k in self.weights.keys()}
            self.last_loss = self.current_loss.copy()
            self.recorded = False

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

    def loss_fn(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
        return torch.sqrt(self.mseloss(output[k], target[k]))


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


LOSS_REGISTER = {
    "mae": MAELoss,
    "mse": MSELoss,
    "rmse": RMSELoss,
    "nll": NLLLoss,
    "nll_var_only": NLLLossVarOnly,
    "l2_penalty": L2Penalty,
    "crps": CRPSLoss
}
