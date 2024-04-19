from torch.nn import MSELoss, L1Loss

class MSE_nh_Loss:
    def __init__(self, weights, **params):
        self.weights = weights
        self.mseloss = MSELoss()

    def __call__(self, output, target):
        loss = output["nh_loss"]
        for k, v in self.weights.items():
            if k != "nh_loss":
                loss += self.mseloss(output[k], target[k].type(output[k].dtype)) * v
        return loss


class MAE_nh_Loss:
    def __init__(self, weights, **params):
        self.weights = weights
        self.maeloss = L1Loss()

    def __call__(self, output, target):
        loss = output["nh_loss"] * self.weights.get("nh_loss", 0)
        for k, v in self.weights.items():
            if k != "nh_loss":
                loss += self.maeloss(output[k], target[k].type(output[k].dtype)) * v
        return loss
        