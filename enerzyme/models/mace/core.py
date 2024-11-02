from typing import Dict
from torch import Tensor
from ..layers import BaseFFCore

try:
    from mace.modules.models import MACE
except ImportError:
    raise ImportError("External FF: MACE is not installed. Please install it with `pip install mace-torch`.")


class MACEWrapper(BaseFFCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = MACE(**kwargs)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.model(data)
