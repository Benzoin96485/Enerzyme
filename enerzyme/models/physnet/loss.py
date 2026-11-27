from typing import Dict
from torch import Tensor
from ..loss import LossComponent


class NHLoss:
    def __init__(self, weight: float) -> None:
        self.weight = weight

    def __call__(self, output: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        return output.get("nh_loss", 0) * self.weight

    def components(self, output, target):
        value = output.get("nh_loss")
        if value is None:
            zero = next(t for t in output.values() if isinstance(t, Tensor)).new_zeros(())
            yield LossComponent(zero, 1, self.weight)
            return
        # Each readout block averages over atoms, fields and ensemble members.
        # Only the atom count varies between replicas of the same architecture.
        for fields, key in ((output, "Za"), (output, "Ea"), (output, "Qa"), (target, "Fa")):
            if key in fields:
                yield LossComponent(value, len(fields[key]), self.weight)
                return
        raise ValueError("nh_penalty requires atomic output Za, Ea, Qa or force labels to count atoms")


LOSS_REGISTER = {
    "nh_penalty": NHLoss,
}
