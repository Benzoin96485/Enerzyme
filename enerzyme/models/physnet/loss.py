from typing import Dict
from torch.nn import Tensor


class NHLoss:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        return output.get("nh_loss", 0) * self.weight

LOSS_REGISTER = {
    "nh_penalty": NHLoss,
}