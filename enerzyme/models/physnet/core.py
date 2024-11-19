from typing import Dict, List
import torch
from torch import Tensor
from torch.nn import Module, Sequential
from .interaction import InteractionBlock, OutputBlock
from ..layers import DistanceLayer, RangeSeparationLayer, BaseFFCore
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE


DEFAULT_BUILD_PARAMS = {
    'dim_embedding': 128,
    'num_rbf': 64,
    'max_Za': 94,
    'cutoff_sr': 10.0,
    'Hartree_in_E': 1,
    'Bohr_in_R': 0.5291772108,
    'cutoff_fn': 'polynomial'
}
DEFAULT_LAYER_PARAMS = [{'name': 'RangeSeparation'},
 {'name': 'ExponentialGaussianRBF',
  'params': {'no_basis_at_infinity': False,
   'init_alpha': 1,
   'exp_weighting': False,
   'learnable_shape': True,
   'init_width_flavor': 'PhysNet'}},
 {'name': 'RandomAtomEmbedding'},
 {'name': 'Core',
  'params': {'num_blocks': 5,
   'num_residual_atomic': 2,
   'num_residual_interaction': 3,
   'num_residual_output': 1,
   'activation_fn': 'shifted_softplus',
   'activation_params': {'dim_feature': 1,
    'initial_alpha': 1,
    'initial_beta': 1,
    'learnable': False},
   'dropout_rate': 0.0}},
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


class PhysNetCore(BaseFFCore):
    def __str__(self) -> str:
        return """
###################################################################################
# Pytorch Implementation of PhysNet (J. Chem. Theory Comput. 2019, 15, 3678−3693) #
###################################################################################
"""
    
    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        num_blocks: int=3,                       # number of building blocks to be stacked
        num_residual_atomic: int=2,              # number of residual layers for atomic refinements of feature vector
        num_residual_interaction: int=2,         # number of residual layers for refinement of message vector
        num_residual_output: int=1,              # number of residual layers for the output blocks
        activation_fn: ACTIVATION_KEY_TYPE="shifted_softplus",   # activation function
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        dropout_rate: float=0.0,
        shallow_ensemble_size: int=1
    ) -> None:
        super().__init__(input_fields={"rbf", "atom_embedding", "idx_i_sr", "idx_j_sr"}, output_fields={"Ea", "Qa", "nh_loss"})
        self.num_blocks = num_blocks
        self.drop_out = dropout_rate
        self.interaction_block = Sequential(*[
            InteractionBlock(
                num_rbf, dim_embedding, num_residual_atomic, num_residual_interaction, 
                activation_fn=activation_fn, activation_params=activation_params, dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])
        self.output_block = Sequential(*[
            OutputBlock(
                dim_embedding, num_residual_output, 
                activation_fn=activation_fn, activation_params=activation_params, dropout_rate=dropout_rate, shallow_ensemble_size=shallow_ensemble_size
            ) for _ in range(num_blocks)
        ])

    def build(self, built_layers: List[Module]) -> None:
        # build necessary fixed pre-core layers
        calculate_distance = DistanceLayer()
        calculate_distance.reset_field_name(Dij="Dij_lr")
        self.pre_sequence.append(calculate_distance)

        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                # reset pre-core layers
                if isinstance(layer, RangeSeparationLayer):
                    layer.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                # build pre-core sequence
                self.pre_sequence.append(layer)
            else: 
                # build post-core sequence
                self.post_sequence.append(layer)

    def get_output(self, rbf: Tensor, atom_embedding: Tensor, idx_i_sr: Tensor, idx_j_sr: Tensor) -> Dict[str, Tensor]:
        '''
        Compute raw atomic properties
        '''
        Ea = 0 # atomic energy 
        Qa = 0 # atomic charge
        nhloss = 0 #non-hierarchicality loss
        for i in range(self.num_blocks):
            atom_embedding = self.interaction_block[i](atom_embedding, rbf, idx_i_sr, idx_j_sr)
            out = self.output_block[i](atom_embedding)
            Ea += out[:,0]
            Qa += out[:,1]
            # compute non-hierarchicality loss
            out2 = out ** 2
            if i > 0:
                nhloss += torch.mean(out2 / (out2 + lastout2 + 1e-7))
            lastout2 = out2
        return {"Ea": Ea, "Qa": Qa, "nh_loss": nhloss}
