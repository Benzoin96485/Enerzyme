import math
import torch.nn.functional as F
from torch import Tensor


LOG2 = math.log(2.0)


def shifted_softplus(x: Tensor) -> Tensor:
    return F.softplus(x) - LOG2


ACTIVATION_REGISTER = {
    "shifted_softplus": shifted_softplus
}