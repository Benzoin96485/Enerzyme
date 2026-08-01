from addict import Dict
from ase.units import Bohr
import torch
from torch.nn import Module
from ..data.transform import Transform
from ..data.neighbor_list import full_neighbor_list
from .trainer import DTYPE_MAPPING, _load_state_dict
from .batch import _decorate_batch_input, _to_device, _decorate_batch_output


class Server:
    def __init__(self, config: Dict, model: Module, model_path: str, out_dir: str, transform: Transform):
        self.neighbor_list_type = config.Server.get("neighbor_list", "full")
        self.cuda = config.Server.get('cuda', False)
        self.dtype = DTYPE_MAPPING[config.Server.get("dtype", "float64")]
        self.Hartree_in_E = config.Server.get("Hartree_in_E", 1)
        self.Bohr_in_R = config.Server.get("Bohr_in_R", Bohr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        # single ff simulation
        self.model = model.to(self.device).type(self.dtype)
        _load_state_dict(model, self.device, model_path, inference=True)
        self.model.eval()
        self.calculator = None
        self.out_dir = out_dir
        self.transform = transform
        
    def calculate(self, info):
        features = info.get("features", None)
        if features is None:
            return {}
        if features["N"] is None:
            features["N"] = len(features["Ra"])
        if self.neighbor_list_type == "full":
            idx_i, idx_j = full_neighbor_list(features["N"])
            features["idx_i"] = idx_i
            features["idx_j"] = idx_j
            features["N_pair"] = len(idx_i)
        net_input, _ = _decorate_batch_input(
            batch=[(features, None)],
            device=self.device,
            dtype=self.dtype
        )
        net_input, _ = _to_device((net_input, {}), self.device)
        net_output = self.model(net_input)
        output, _ = _decorate_batch_output(
            output=net_output,
            features=net_input,
            targets=None
        )
        self.transform.inverse_transform(output)
        return output
