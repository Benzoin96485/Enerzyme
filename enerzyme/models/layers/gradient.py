import torch
from torch import Tensor
from torch.autograd import grad
from torch.autograd.functional import jacobian
from ..layers import BaseFFLayer


class ForceLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"E", "Ra"}, output_fields={"Fa"})

    def get_Fa(self, E: Tensor, Ra: Tensor) -> Tensor:
        # A position-independent energy (an isolated atom whose head is only a
        # species baseline) has no ``Ra`` in the graph. The force is zero.
        if E.dim() > 1:
            columns = [
                self._neg_coordinate_grad(torch.sum(E[:, i]), Ra)
                for i in range(E.shape[-1])
            ]
            return torch.stack(columns, dim=-1)
        return self._neg_coordinate_grad(torch.sum(E), Ra)

    def _neg_coordinate_grad(self, energy: Tensor, Ra: Tensor) -> Tensor:
        coordinate_grad = grad(
            energy,
            Ra,
            retain_graph=True,
            create_graph=self.training,
            allow_unused=True,
        )[0]
        if coordinate_grad is None:
            return torch.zeros_like(Ra)
        return -coordinate_grad

class EnergyVarianceGradientLayer(BaseFFLayer):
    def __init__(self) -> None:
        super().__init__(input_fields={"E_var", "Ra"}, output_fields={"E_var_grad"})

    def get_E_var_grad(self, E_var: Tensor, Ra: Tensor) -> Tensor:
        return grad(torch.sum(E_var), Ra, retain_graph=True, create_graph=self.training)[0]
