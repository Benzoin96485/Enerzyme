import torch
from torch import Tensor
from torch.autograd import grad
from torch.autograd.functional import jacobian
from ..layers import BaseFFLayer


class ForceLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"E", "Ra"}, output_fields={"Fa"})

    def get_Fa(self, E: Tensor, Ra: Tensor) -> Tensor:
        if E.dim() > 1:
            return torch.stack([-grad(torch.sum(E[:,i]), Ra, create_graph=True)[0] for i in range(E.shape[-1])]).permute(1, 2, 0)
        else:
            return -grad(torch.sum(E), Ra, create_graph=True)[0]
