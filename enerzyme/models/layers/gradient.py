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
            return torch.stack([-grad(torch.sum(E[:,i]), Ra, retain_graph=True, create_graph=self.training)[0] for i in range(E.shape[-1])], dim=-1)
        else:
            return -grad(torch.sum(E), Ra, retain_graph=True, create_graph=self.training)[0]

class EnergyVarianceGradientLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"E_var", "Ra"}, output_fields={"E_var_grad"})

    def get_E_var_grad(self, E_var: Tensor, Ra: Tensor) -> Tensor:
        return grad(torch.sum(E_var), Ra, retain_graph=True, create_graph=self.training)[0]
