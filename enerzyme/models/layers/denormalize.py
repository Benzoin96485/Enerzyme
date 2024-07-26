from typing import Dict
from torch import nn, Tensor

class AtomicAffine(nn.Module):
    def __init__(self, max_Za) -> None:
        affine_params = dict()
        self.scales = nn.ParameterDict()
        self.shifts = nn.ParameterDict()
        

    def forward(**net_input) -> Dict[str, Tensor]:
        output = net_input.copy()
        output[]