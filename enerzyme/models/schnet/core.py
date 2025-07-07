"""
===============================================================================
Description:    Code is adapted from torch geometric implementation https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/schnet.py.
All rights reserved to original authors.
===============================================================================
"""

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from .interaction import InteractionBlock
from .. import BaseFFCore
from ..blocks.mlp import DenseLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, get_activation_fn


class SchNetCore(BaseFFCore):
    def __init__(
            self,
            hidden_channels: int = 128,
            dim_embedding: int = 128,
            num_interactions: int = 4,
            num_rbf: int = 128,
            cutoff: float = 5.0,
            activation_fn: ACTIVATION_KEY_TYPE="shifted_softplus",   # activation function
            activation_params: ACTIVATION_PARAM_TYPE=dict(),
            shallow_ensemble_size: int=1
    ):
        super().__init__(input_fields={"idx_i_sr", "idx_j_sr", "Dij_sr", "rbf", "atom_embedding"}, output_fields={"Ea", "Qa"})
        
        self.hidden_channels = hidden_channels
        self.num_filters = dim_embedding
        self.num_interactions = num_interactions
        self.num_gaussians = num_rbf
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_rbf,
                                     dim_embedding, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = get_activation_fn(activation_fn, activation_params)
        self.lin2 = DenseLayer(hidden_channels // 2, 2, initial_weight="xavier_uniform", initial_bias="zero", shallow_ensemble_size=shallow_ensemble_size)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def get_output(self, idx_i_sr: Tensor, idx_j_sr: Tensor, Dij_sr: Tensor, rbf: Tensor, atom_embedding: Tensor):
        edge_index = torch.stack([idx_i_sr, idx_j_sr])

        for interaction in self.interactions:
            atom_embedding = atom_embedding + interaction(atom_embedding, edge_index, Dij_sr, rbf)

        atom_embedding = self.lin1(atom_embedding)
        atom_embedding = self.act(atom_embedding)
        output = self.lin2(atom_embedding)
        
        return {"Ea": output[:,0], "Qa": output[:,1]}



