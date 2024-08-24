from typing import Dict
import torch
from torch import Tensor
from torch.nn import Module


class ForceLayer(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output["Fa"] = -torch.autograd.grad(torch.sum(net_input["E"]), net_input["Ra"], create_graph=True)[0]
        return output