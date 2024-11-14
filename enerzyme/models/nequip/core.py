from typing import Dict, List
import torch
from torch import Tensor
from ..layers import BaseFFCore
try:
    from nequip.model import model_from_config
    from nequip.data.AtomicData import neighbor_list_and_relative_vec
    from nequip.data.transforms import TypeMapper
except ImportError:
    raise ImportError("External FF: NequIP is not installed. Please install it with `pip install nequip`.")


DEFAULT_BUILD_PARAMS = {
    "default_dtype": "float32",
    "model_dtype": "float32",
    "r_max": 6.0,

}
DEFAULT_LAYER_PARAMS = [{"name": "Core", "params": {
    "model_builders": [
        "SimpleIrrepsConfig",
        "EnergyModel",
        "PerSpeciesRescale",
        "ForceOutput",
        "RescaleEnergyEtc"
    ],
    "num_layers": 4,
    "l_max": 1,
    "parity": True,
    "num_features": 32,
    "nonlinearity_type": "gate",
    "resnet": False,
    "activation": "silu",
    "nonlinearity_scalars": {
        "e": "silu",
        "o": "tanh"
    },
    "nonlinearity_gates": {
        "e": "silu",
        "o": "tanh"
    },
    "num_basis": 8,
    "BesselBasis_trainable": True,
    "PolynomialCutoff_p": 6,
    "invariant_layers": 2,
    "invariant_neurons": 64,
    "avg_num_neighbors": 8,
    "use_sc": True,
    "chemical_symbols": ["H", "C", "N", "O", "P", "S"]
}}]


class NequIPWrapper(BaseFFCore):
    def __init__(
        self, 
        default_dtype: str,
        model_dtype: str,
        r_max: float,
        model_builders: List[str],
        num_layers: int,
        l_max: int,
        parity: bool,
        num_features: int,
        nonlinearity_type: str,
        resnet: bool,
        activation: str,
        nonlinearity_scalars: Dict[str, str],
        nonlinearity_gates: Dict[str, str],
        num_basis: int,
        BesselBasis_trainable: bool,
        PolynomialCutoff_p: int,
        invariant_layers: int,
        invariant_neurons: int,
        avg_num_neighbors: float,
        use_sc: bool,
        chemical_symbols: List[str]
    ):
        super().__init__(input_fields={"Ra", "Za", "batch_seg"}, output_fields={"E", "Fa"})
        self.r_max = r_max
        self.model = model_from_config({
            "default_dtype": default_dtype,
            "model_dtype": model_dtype,
            "r_max": r_max,
            "model_builders": model_builders,
            "num_layers": num_layers,
            "l_max": l_max,
            "parity": parity,
            "num_features": num_features,
            "nonlinearity_type": nonlinearity_type,
            "resnet": resnet,
            "activation": activation,
            "nonlinearity_scalars": nonlinearity_scalars,
            "nonlinearity_gates": nonlinearity_gates,
            "num_basis": num_basis,
            "BesselBasis_trainable": BesselBasis_trainable,
            "PolynomialCutoff_p": PolynomialCutoff_p,
            "invariant_layers": invariant_layers,
            "invariant_neurons": invariant_neurons,
            "avg_num_neighbors": avg_num_neighbors,
            "use_sc": use_sc,
            "chemical_symbols": chemical_symbols
        })
        self.type_mapper = TypeMapper(chemical_symbols=chemical_symbols)

    def build(self, built_layers) -> None:
        pass

    def get_output(self, Ra: Tensor, Za: Tensor, batch_seg: Tensor) -> Dict[str, Tensor]:
        edge_index = None
        ptr = [0]
        count = 0
        for i in range(batch_seg[-1] + 1):
            mask = batch_seg == i
            
            edge_index_batch, _, _ = neighbor_list_and_relative_vec(
                pos=Ra[mask], r_max=self.r_max, strict_self_interaction=False
            )
            if i == 0:
                edge_index = edge_index_batch
            else:
                edge_index = torch.cat([edge_index, edge_index_batch + count], dim=1)
            N = mask.sum().item()
            count += N
            ptr.append(count)
        ptr = torch.tensor(ptr, dtype=torch.long, device=Ra.device)
        raw_output = self.model({
            "pos": Ra,
            "edge_index": edge_index,
            "atom_types": self.type_mapper.transform(Za),
            "batch": batch_seg,
            "ptr": ptr
        })
        return {"E": raw_output["total_energy"].squeeze(), "Fa": raw_output["forces"]}
