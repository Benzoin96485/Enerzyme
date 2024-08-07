import torch
from torch import Tensor
from torch.nn import Module, Sequential
from typing import Dict, Any
from .interaction import InteractionBlock, OutputBlock
from ..layers import DistanceLayer, BaseRBF, BaseAtomEmbedding
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE


LAYERS = [
    "Distance",
    "RandomEmbedding",
    "ExponentialGaussianRBFLayer",
    "Core",
    "AtomicAffine",
    "ChargeConservation",
    "AtomicCharge2Dipole",
    "ElectrostaticEnergy",
    "GrimmeD3Energy",
    "EnergyReduce",
    "Force"
]


class PhysNetCore(Module):
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
        dropout_rate: float=0.0
    ) -> None:
        super().__init__()
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
                activation_fn=activation_fn, activation_params=activation_params, dropout_rate=dropout_rate
            ) for _ in range(num_blocks)
        ])
        
        self.distance_layer: DistanceLayer = None
        self.rbf_layer: BaseRBF = None
        self.embeddings: BaseAtomEmbedding = None

    @classmethod
    def build(cls, built_layers: Dict[str, Module], **build_params: Dict[str, Any]) -> Module:
        instance = cls(**build_params)

        # build necessary flexiable pre-core layers
        for layer_name, layer in built_layers.items():
            if isinstance(layer, BaseRBF):
                instance.rbf_layer = layer
            elif isinstance(layer, BaseAtomEmbedding):
                instance.embeddings = layer

        # check if necessary flexible pre-core layers has been built
        for layer_name in ["rbf_layer", "embeddings"]:
            if getattr(instance, layer_name) is None:
                raise AttributeError(f"{layer_name} is not built")
            
        # build necessary fixed pre-core layers
        instance.distance_layer = DistanceLayer()

        # reset pre-core layers
        
        return instance

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''
        Compute raw atomic properties
        '''
        # long-short range separation
        net_output = self.distance_layer(
            net_input,
            Dij_name = "Dij",
        )
        if "sr_idx_i" in net_input and "sr_idx_j" in net_input:
            net_output = self.distance_layer(
                net_output,
                idx_i_name = "sr_idx_i",
                idx_j_name = "sr_idx_j",
                Dij_name = "Dij_sr"
            )
            sr_idx_i = net_output["sr_idx_i"]
            sr_idx_j = net_output["sr_idx_j"]
        else:
            sr_idx_i = net_output["idx_i"]
            sr_idx_j = net_output["idx_j"]
            net_output["Dij_sr"] = net_output["Dij"]
        
        # rbf based on short range interations
        net_output = self.rbf_layer(
            net_output,
            Dij_name = "Dij_sr"
        )
        net_output.pop("Dij_sr")
        net_output = self.embeddings(net_output)

        rbf = net_output["rbf"]
        x = net_output["atom_embedding"]

        Ea = 0 #atomic energy 
        Qa = 0 #atomic charge
        nhloss = 0 #non-hierarchicality loss
        for i in range(self.num_blocks):
            x = self.interaction_block[i](x, rbf, sr_idx_i, sr_idx_j)
            out = self.output_block[i](x)
            Ea += out[:,0]
            Qa += out[:,1]
            # compute non-hierarchicality loss
            out2 = out ** 2
            if i > 0:
                nhloss += torch.mean(out2 / (out2 + lastout2 + 1e-7))
            lastout2 = out2
        net_output["Ea"] = Ea
        net_output["Qa"] = Qa
        net_output["nh_loss"] = nhloss
        return net_output
