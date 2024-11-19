from abc import ABC, abstractmethod
import torch
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss


class WeightedLoss(ABC):
    def __init__(self, **weights):
        self.weights = weights

    @abstractmethod
    def loss_fn(self, output, target, k):
        ...

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss = loss + v * self.loss_fn(output, target, k)
        return loss


class MSELoss(WeightedLoss):
    def __init__(self, **weights):
        super().__init__(**weights)
        self.mseloss = MSELoss_()

    def loss_fn(self, output, target, k):
        return self.mseloss(output[k], target[k])


class RMSELoss(WeightedLoss):
    def __init__(self, **weights):
        super().__init__(**weights)
        self.mseloss = MSELoss_()

    def loss_fn(self, output, target, k):
        return torch.sqrt(self.mseloss(output[k], target[k]))


class MAELoss(WeightedLoss):
    def __init__(self, **weights):
        super().__init__(**weights)
        self.maeloss = L1Loss()

    def loss_fn(self, output, target, k):
        return self.maeloss(output[k], target[k])


class NLLLoss(WeightedLoss):
    def __init__(self, eps=1e-6, **weights):
        super().__init__(**weights)
        self.eps = eps

    def loss_fn(self, output, target, k):
        return 0.5 * torch.mean(torch.log(output[k + "_var"] + self.eps) + (output[k] - target[k]) ** 2 / (output[k + "_var"] + self.eps))


LOSS_REGISTER = {
    "mae": MAELoss,
    "mse": MSELoss,
    "rmse": RMSELoss,
    "nll": NLLLoss
}
