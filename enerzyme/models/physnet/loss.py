class NHLoss:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, output, target):
        return output.get("nh_loss", 0) * self.weight
        