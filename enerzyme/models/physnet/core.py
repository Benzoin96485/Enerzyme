import torch
from torch import nn, Tensor
from typing import Dict, Any, Literal
from .block import InteractionBlock, OutputBlock
from ..functional import segment_sum
from ..layers import DistanceLayer, BaseRBF, BaseAtomEmbedding
from ..activation import ACTIVATION_REGISTER


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


class PhysNetCore(nn.Module):
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
        cutoff_sr: float,                             # cutoff distance for short range interactions
        cutoff_lr: float=None,                        # cutoff distance for long range interactions (default: no cutoff)
        num_blocks: int=3,                       # number of building blocks to be stacked
        num_residual_atomic: int=2,              # number of residual layers for atomic refinements of feature vector
        num_residual_interaction: int=2,         # number of residual layers for refinement of message vector
        num_residual_output: int=1,              # number of residual layers for the output blocks
        activation_fn: Literal["shifted_softplus"]="shifted_softplus",   # activation function
        drop_out: float=0.0
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.sr_cut = cutoff_sr
        self.lr_cut = cutoff_lr
        self.activation_fn = ACTIVATION_REGISTER[activation_fn]
        self.drop_out = drop_out

        self._interaction_block = nn.Sequential(*[
            InteractionBlock(
                num_rbf, dim_embedding, num_residual_atomic, num_residual_interaction, activation_fn=self.activation_fn, drop_out=drop_out
            ) for i in range(num_blocks)
        ])
        self._output_block = nn.Sequential(*[
            OutputBlock(dim_embedding, num_residual_output, activation_fn=self.activation_fn, drop_out=drop_out
            ) for i in range(num_blocks)
        ])
        
        self.distance_layer: DistanceLayer = None
        self.rbf_layer: BaseRBF = None
        self.embeddings: BaseAtomEmbedding = None

    @classmethod
    def build(cls, built_layers: Dict[str, nn.Module], **build_params: Dict[str, Any]) -> nn.Module:
        instance = cls(**build_params)
        for layer_name, layer in built_layers.items():
            if layer_name.endswith("Distance"):
                instance.distance_layer = layer
            elif layer_name.endswith("RBF"):
                instance.rbf_layer = layer
            elif layer_name.endswith("AtomEmbedding"):
                instance.embeddings = layer
        for layer_name in ["distance_layer", "rbf_layer", "embeddings"]:
            if getattr(instance, layer_name) is None:
                raise AttributeError(f"{layer_name} is not built")
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

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
    