from abc import ABC, abstractmethod
from typing import Dict
import torch
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss


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
    "l2_penalty": L2Penalty
}
