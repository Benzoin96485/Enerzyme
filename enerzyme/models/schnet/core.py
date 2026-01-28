"""
Description:    Code is adapted from torch geometric implementation https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py.
All rights reserved to original authors.
"""

from typing import List
import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from .interaction import InteractionBlock
from ..blocks.mlp import DenseLayer
from ..layers import BaseFFCore, DistanceLayer, RangeSeparationLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, get_activation_fn


DEFAULT_BUILD_PARAMS = {
    'dim_embedding': 128,
    'num_rbf': 128,
    'max_Za': 94,
    'cutoff_sr': 5.0,
    'Hartree_in_E': 1,
    'Bohr_in_R': 0.5291772108
}
DEFAULT_LAYER_PARAMS = [{'name': 'RangeSeparation'},
 {'name': 'GaussianSmearing',
  'params': {'no_basis_at_infinity': False,
   'init_alpha': 1,
   'exp_weighting': False,
   'learnable_shape': True,
   'init_width_flavor': 'PhysNet'}},
 {'name': 'RandomAtomEmbedding'},
 {'name': 'Core',
  'params': {'hidden_channels': 128,
   'num_interactions': 4,
   'activation_fn': 'shifted_softplus',
   'activation_params': {'dim_feature': 1,
    'initial_alpha': 1,
    'initial_beta': 1,
    'learnable': False},
}},
 {'name': 'AtomicAffine',
  'params': {'shifts': {'Ea': {'values': 0, 'learnable': True},
    'Qa': {'values': 0, 'learnable': True}},
   'scales': {'Ea': {'values': 1, 'learnable': True},
    'Qa': {'values': 1, 'learnable': True}}}},
 {'name': 'ChargeConservation'},
 {'name': 'AtomicCharge2Dipole'},
 {'name': 'ElectrostaticEnergy',
  'params': {'cutoff_lr': None, 'flavor': 'PhysNet'}},
 {'name': 'GrimmeD3Energy', 'params': {'learnable': True}},
 {'name': 'EnergyReduce'},
 {'name': 'Force'}]

class SchNetCore(BaseFFCore):
    def __init__(
            self,
            hidden_channels: int = 128,
            dim_embedding: int = 128,
            num_interactions: int = 4,
            num_rbf: int = 128,
            cutoff_sr: float = 5.0,
            activation_fn: ACTIVATION_KEY_TYPE="shifted_softplus",   # activation function
            activation_params: ACTIVATION_PARAM_TYPE=dict(),
            shallow_ensemble_size: int=1
    ):
        super().__init__(input_fields={"idx_i_sr", "idx_j_sr", "Dij_sr", "rbf", "atom_embedding"}, output_fields={"Ea", "Qa"})
        
        self.hidden_channels = hidden_channels
        self.num_filters = dim_embedding
        self.num_interactions = num_interactions
        self.num_gaussians = num_rbf
        self.cutoff = cutoff_sr

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_rbf,
                                     dim_embedding, cutoff_sr)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = get_activation_fn(activation_fn, activation_params)
        self.lin2 = DenseLayer(hidden_channels // 2, 2, initial_weight="xavier_uniform", initial_bias="zero", shallow_ensemble_size=shallow_ensemble_size)
        self.shallow_ensemble_size = shallow_ensemble_size
        self.reset_parameters()

    def __str__(self) -> str:
        return """
######################################################
# Augmented SchNet (NeurIPS 2017, arXiv: 1706.08566) #
######################################################
"""

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)

    def build(self, built_layers: List[Module]) -> None:
        # build necessary fixed pre-core layers
        # TODO: make this more flexible
        pre_core = True
        for i, layer in enumerate(built_layers):
            if i == 0:
                if isinstance(layer, DistanceLayer):
                    layer.reset_field_name(Dij="Dij_lr")
                    self.pre_sequence.append(layer)
                else:
                    calculate_distance = DistanceLayer()
                    calculate_distance.reset_field_name(Dij="Dij_lr")
                    self.pre_sequence.append(calculate_distance)
            
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                # build pre-core sequence
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                self.pre_sequence.append(layer)
            else: 
                # build post-core sequence
                self.post_sequence.append(layer)

    def get_output(self, idx_i_sr: Tensor, idx_j_sr: Tensor, Dij_sr: Tensor, rbf: Tensor, atom_embedding: Tensor):
        edge_index = torch.stack([idx_i_sr, idx_j_sr])

        for interaction in self.interactions:
            atom_embedding = atom_embedding + interaction(atom_embedding, edge_index, Dij_sr, rbf)

        atom_embedding = self.lin1(atom_embedding)
        atom_embedding = self.act(atom_embedding)
        output = self.lin2(atom_embedding)
        
        return {"Ea": output[:,0], "Qa": output[:,1]}



