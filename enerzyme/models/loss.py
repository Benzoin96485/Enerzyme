import torch
from torch.nn import MSELoss as MSELoss_
from torch.nn import L1Loss


class MSELoss:
    def __init__(self, **weights):
        self.weights = weights
        self.mseloss = MSELoss_()

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss = loss + self.mseloss(output[k], target[k]) * v
        return loss
    

class RMSELoss:
    def __init__(self, **weights):
        self.weights = weights
        self.mseloss = MSELoss_()

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss = loss + torch.sqrt(self.mseloss(output[k], target[k])) * v
        return loss


class MAELoss:
    def __init__(self, **weights):
        self.weights = weights
        self.maeloss = L1Loss()

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss = loss + self.maeloss(output[k], target[k]) * v
        return loss


LOSS_REGISTER = {
    "mae": MAELoss,
    "mse": MSELoss,
    "rmse": RMSELoss
}