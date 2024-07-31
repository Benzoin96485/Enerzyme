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
    def __str__(self):
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
        # Eshift=0.0,                     #initial value for output energy shift (makes convergence faster)
        # Escale=1.0,                     #initial value for output energy scale (makes convergence faster)
        # Qshift=0.0,                     #initial value for output charge shift 
        # Qscale=1.0,                     #initial value for output charge scale 
        # kehalf=7.199822675975274,       #half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
        # d3_autoev=d3_autoev,
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
            Dij_name = "Dij_lr",
            **net_input
        )
        if "sr_idx_i" in net_input and "sr_idx_j" in net_input:
            net_output = self.distance_layer(
                idx_i_name = "sr_idx_i",
                idx_j_name = "sr_idx_j",
                Dij_name = "Dij_sr",
                **net_output
            )
        else:
            net_output["sr_idx_i"] = net_output["idx_i"]
            net_output["sr_idx_j"] = net_output["idx_j"]
            net_output["Dij_sr"] = net_output["Dij_lr"]
        
        # rbf based on short range interations
        net_output = self.rbf_layer(
            Dij_name = "Dij_sr",
            **net_output
        )

        net_output = self.embeddings(**net_output)

        rbf = net_output["rbf"]
        x = net_output["atom_embedding"]
        sr_idx_i = net_output["sr_idx_i"]
        sr_idx_j = net_output["sr_idx_j"]
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

        #apply scaling/shifting
        # Ea = self.Escale[Za] * Ea + self.Eshift[Za] + 0 * torch.sum(Ra, -1) #last term necessary to guarantee no "None" in force evaluation
        # Qa = self.Qscale[Za] * Qa + self.Qshift[Za]
    
    def energy_from_scaled_atomic_properties(self, Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        #add electrostatic and dispersion contribution to atomic energy
        if self.use_electrostatic:
            Ea += self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j)
        if self.use_dispersion:
            if self.lr_cut is not None:   
                Ea += self.d3_autoev * self.disp_layer(
                    Z, Dij / d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2, cutoff=self.lr_cut/d3_autoang
                )
            else:
                Ea += self.d3_autoev * self.disp_layer(
                    Z, Dij / d3_autoang, idx_i, idx_j, s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2
                )
        return segment_sum(Ea, batch_seg)

    def energy_and_forces_from_scaled_atomic_properties(self, Ea, Qa, Dij, Za, Ra, idx_i, idx_j, batch_seg=None, **params):
        energy = self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Za, idx_i, idx_j, batch_seg)
        forces = -torch.autograd.grad(torch.sum(energy), Ra, create_graph=True, retain_graph=True)[0]
        return energy, forces
    
    def energy_from_atomic_properties(self, Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot=None, batch_seg=None):
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        #scale charges such that they have the desired total charge
        Qa = self.scaled_charges(Z, Qa, Q_tot, batch_seg)
        return self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)

    #calculates the energy and force given the atomic properties (in order to prevent recomputation if atomic properties are calculated)
    def energy_and_forces_from_atomic_properties(self, Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None):
        energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)
        forces = -torch.autograd.grad(torch.sum(energy), R, create_graph=True)[0]
        return energy, forces

    #calculates the total energy (including electrostatic interactions)
    def energy(self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
        energy = self.energy_from_atomic_properties(Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)
        return energy

    #calculates the total energy and forces (including electrostatic interactions)
    def energy_and_forces(self, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None):
        Ea, Qa, Dij, _ = self.atomic_properties(Z, R, idx_i, idx_j, offsets, sr_idx_i, sr_idx_j, sr_offsets)
        energy, forces = self.energy_and_forces_from_atomic_properties(Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot, batch_seg)
        return energy, forces

    #returns scaled charges such that the sum of the partial atomic charges equals Q_tot (defaults to 0)
    def scaled_charges(self, Za, Qa, Q=None, batch_seg=None, **kwargs):
        return self._charge_conservation_layer.get_corrected_Qa(Za, Qa, Q, batch_seg)["Qa"]

    #calculates the electrostatic energy per atom 
    #for very small distances, the 1/r law is shielded to avoid singularities
    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        return self._electrostatic_layer.get_E_ele_a(Dij, Qa, idx_i, idx_j)

    # def forward(self, **net_input):
    #     Ea, raw_Qa, Dij, nh_loss = self.atomic_properties(**net_input)
    #     Qa = self.scaled_charges(Qa=raw_Qa, **net_input)
    #     output = {"nh_loss": nh_loss}
    #     output["Qa"] = Qa
    #     output["Q"] = segment_sum(raw_Qa, net_input["batch_seg"])
    #     energy, forces = self.energy_and_forces_from_scaled_atomic_properties(Ea, Qa, Dij, **net_input)
    #     output["E"] = energy
    #     output["Fa"] = forces
    #     output["M2"] = segment_sum(Qa.unsqueeze(1) * net_input["Ra"], net_input["batch_seg"])
    #     return output

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
