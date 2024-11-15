from typing import List, Optional, Union
import torch
from torch.nn import Module
from ..layers import BaseFFCore, DistanceLayer, RangeSeparationLayer
try:
    from xequinet.nn.model import XPaiNN
    from xequinet.utils.config import NetConfig
    from torch_geometric.data import Data
except ImportError as e:
    raise ImportError("External FF: XPaiNN is not installed. Please install it following https://github.com/X1X1010/XequiNet.", e)


DEFAULT_BUILD_PARAMS = {
    'cutoff_sr': 5.0,
}
DEFAULT_LAYER_PARAMS = [
    {'name': 'RangeSeparation'},
    {"name": "Core", "params": {
        'embed_basis': "gfn2-xtb",                  # embedding basis type
        'aux_basis': "aux56",                       # auxiliary basis type
        'node_dim': 128,                            # node irreps for the input
        'edge_irreps': "128x0e + 64x1o + 32x2e",    # edge irreps for the input
        'hidden_dim': 64,                           # hidden dimension for the output
        'hidden_irreps': "64x0e + 32x1o + 16x2e",   # hidden irreps for the output
        'rbf_kernel': "bessel",                     # radial basis function type
        'num_basis': 20,                            # number of the radial basis functions
        'cutoff_fn': "cosine",                      # cutoff function type
        'max_edges': 100,                           # maximum number of the edges
        'action_blocks': 3,                         # number of the action blocks
        'activation': "silu",                       # activation function type
        'norm_type': "nonorm",                      # normalization layer type
        'output_mode': "grad",                    # task type (`scalar` is for energy like, `grad` is for force like, etc.)
        'output_dim': 1,                            # output dimension of multi-task (only for `scalar` mode)
        'node_average': False,                      # whether to add the node average to the output (only for `scalar` mode)
        'default_dtype': "float32"                 # default data type      
    }}
]


class XPaiNNWrapper(BaseFFCore):
    def __init__(self, 
        cutoff_sr: float, 
        embed_basis: str,
        aux_basis: str,
        node_dim: int,
        edge_irreps: str,
        hidden_dim: int,
        hidden_irreps: str,
        rbf_kernel: str,
        num_basis: int,
        cutoff_fn: str,
        max_edges: int,
        action_blocks: int,
        activation: str,
        norm_type: str,
        output_mode: str,
        output_dim: int,
        node_average: Union[bool, float],
        default_dtype: str,
    ):
        super().__init__(input_fields={"Ra", "Za", "batch_seg", "idx_i", "idx_j"}, output_fields={"E", "Fa"})
        config = NetConfig(**{
            "cutoff": cutoff_sr,
            "embed_basis": embed_basis,
            "aux_basis": aux_basis,
            "node_dim": node_dim,
            "edge_irreps": edge_irreps,
            "hidden_dim": hidden_dim,
            "hidden_irreps": hidden_irreps,
            "rbf_kernel": rbf_kernel,
            "num_basis": num_basis,
            "cutoff_fn": cutoff_fn,
            "max_edges": max_edges,
            "action_blocks": action_blocks,
            "activation": activation,
            "norm_type": norm_type,
            "output_mode": output_mode,
            "output_dim": output_dim,
            "node_average": node_average,
            "default_dtype": default_dtype,
        })
        self.model = XPaiNN(config)

    def __str__(self) -> str:
        return """
####################################################################
# Wrapped XPaiNN (J. Chem. Theory Comput. 2024, 20, 21, 9500–9511) #
####################################################################
"""

    def build(self, built_layers: List[Module]):
        calculate_distance = DistanceLayer()
        calculate_distance.reset_field_name(Dij="Dij_lr")
        self.pre_sequence.append(calculate_distance)
        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(self, Ra, Za, batch_seg, idx_i, idx_j):
        data = Data(edge_index = torch.stack([idx_i, idx_j]), pos=Ra, at_no=Za, batch=batch_seg)
        E, Fa = self.model(data)
        return {"E": E.squeeze(-1), "Fa": Fa}
