from torch.nn import MSELoss

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
        