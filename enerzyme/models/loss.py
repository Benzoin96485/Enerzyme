from torch.nn import MSELoss, L1Loss


class MSELoss:
    def __init__(self, **weights):
        self.weights = weights
        self.mseloss = MSELoss()

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss += self.mseloss(output[k], target[k]) * v
        return loss


class MAELoss:
    def __init__(self, **weights):
        self.weights = weights
        self.maeloss = L1Loss()

    def __call__(self, output, target):
        loss = 0
        for k, v in self.weights.items():
            loss += self.maeloss(output[k], target[k]) * v
        return loss