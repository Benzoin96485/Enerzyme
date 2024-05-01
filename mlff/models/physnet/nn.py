import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F_
from .func import softplus_inverse, segment_sum, shifted_softplus
from .d3 import d3_s6, d3_s8, d3_a1, d3_a2, d3_autoang, d3_autoev, DispLayer
from .layer import RBFLayer
from .block import InteractionBlock, OutputBlock


class PhysNet(nn.Module):
    def __str__(self):
        return """
###################################################################################
# Pytorch Implementation of PhysNet (J. Chem. Theory Comput. 2019, 15, 3678−3693) #
###################################################################################
"""
    
    def __init__(self,
        F,                                  # dimensionality of feature vector
        K,                                  # number of radial basis functions
        sr_cut,                             # cutoff distance for short range interactions
        lr_cut=None,                        # cutoff distance for long range interactions (default: no cutoff)
        num_blocks=3,                       # number of building blocks to be stacked
        num_residual_atomic=2,              # number of residual layers for atomic refinements of feature vector
        num_residual_interaction=2,         # number of residual layers for refinement of message vector
        num_residual_output=1,              # number of residual layers for the output blocks
        use_electrostatic=True,             # adds electrostatic contributions to atomic energy
        use_dispersion=True,                # adds dispersion contributions to atomic energy
        Eshift=0.0,                     #initial value for output energy shift (makes convergence faster)
        Escale=1.0,                     #initial value for output energy scale (makes convergence faster)
        Qshift=0.0,                     #initial value for output charge shift 
        Qscale=1.0,                     #initial value for output charge scale 
        kehalf=7.199822675975274,       #half (else double counting) of the Coulomb constant (default is in units e=1, eV=1, A=1)
        d3_autoev=d3_autoev,
        activation_fn="shifted_softplus",   # activation function
        drop_out=0.0,
        dtype=torch.double,
        fix_scale=False,
        **params
    ):
        super().__init__()
        self._num_blocks = num_blocks
        self._F = F
        self._K = K
        self._dtype = torch.float32 if dtype == "float32" else torch.float64
        self._kehalf = kehalf
        self._sr_cut = sr_cut
        self._lr_cut = lr_cut
        self._use_electrostatic = use_electrostatic
        self._use_dispersion = use_dispersion
        if callable(activation_fn):
            self._activation_fn = activation_fn
        else:
            self._activation_fn = {"shifted_softplus": shifted_softplus}.get(activation_fn, shifted_softplus)
        self._embeddings = nn.Embedding(95, self.F, dtype=self.dtype)
        self._drop_out = drop_out
        self._rbf_layer = RBFLayer(K, sr_cut, dtype=self.dtype)
        self._s6 = nn.Parameter(F_.softplus(torch.tensor(softplus_inverse(d3_s6), dtype=self.dtype)))
        self._s8 = nn.Parameter(F_.softplus(torch.tensor(softplus_inverse(d3_s8), dtype=self.dtype)))
        self._a1 = nn.Parameter(F_.softplus(torch.tensor(softplus_inverse(d3_a1), dtype=self.dtype)))
        self._a2 = nn.Parameter(F_.softplus(torch.tensor(softplus_inverse(d3_a2), dtype=self.dtype)))
        self._Eshift = nn.Parameter(Eshift * torch.ones(95, dtype=self.dtype), requires_grad=not fix_scale)
        self._Escale = nn.Parameter(Escale * torch.ones(95, dtype=self.dtype), requires_grad=not fix_scale)
        self._Qshift = nn.Parameter(Qshift * torch.ones(95, dtype=self.dtype), requires_grad=not fix_scale)
        self._Qscale = nn.Parameter(Qscale * torch.ones(95, dtype=self.dtype), requires_grad=not fix_scale)
        self._d3_autoev = d3_autoev

        self._interaction_block = nn.Sequential(*[
            InteractionBlock(
                K, F, num_residual_atomic, num_residual_interaction, activation_fn=self.activation_fn, drop_out=drop_out,
                dtype=self.dtype
            ) for i in range(num_blocks)
        ])
        self._output_block = nn.Sequential(*[
            OutputBlock(F, num_residual_output, activation_fn=self.activation_fn, drop_out=drop_out, dtype=self.dtype
            ) for i in range(num_blocks)
        ])
        self._disp_layer = DispLayer(dtype=self.dtype)

    def calculate_interatomic_distances(self, R, idx_i, idx_j, offsets=None):
        #calculate interatomic distances
        Ri = R[idx_i]
        Rj = R[idx_j]
        if offsets is not None:
            Rj += offsets
        Dij = torch.sqrt(torch.relu(torch.sum((Ri-Rj)**2, -1))) #relu prevents negative numbers in sqrt
        return Dij
    
    def atomic_properties(self, Za, Ra, idx_i, idx_j, offsets=None, sr_idx_i=None, sr_idx_j=None, sr_offsets=None, **kwargs):
        #calculate distances (for long range interaction)
        Dij_lr = self.calculate_interatomic_distances(Ra, idx_i, idx_j, offsets=offsets)
        #optionally, it is possible to calculate separate distances for short range interactions (computational efficiency)
        if sr_idx_i is not None and sr_idx_j is not None:
            Dij_sr = self.calculate_interatomic_distances(Ra, sr_idx_i, sr_idx_j, offsets=sr_offsets)
        else:
            sr_idx_i = idx_i
            sr_idx_j = idx_j
            Dij_sr = Dij_lr

        #calculate radial basis function expansion
        rbf = self.rbf_layer(Dij_sr)

        #initialize feature vectors according to embeddings for nuclear charges
        x = self.embeddings(Za)

        #apply blocks
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

        #apply scaling/shifting
        Ea = self.Escale[Za] * Ea + self.Eshift[Za] + 0 * torch.sum(Ra, -1) #last term necessary to guarantee no "None" in force evaluation
        Qa = self.Qscale[Za] * Qa + self.Qshift[Za]
        return Ea, Qa, Dij_lr, nhloss
    
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
        return torch.squeeze(segment_sum(Ea, batch_seg))

    def energy_and_forces_from_scaled_atomic_properties(self, Ea, Qa, Dij, Za, Ra, idx_i, idx_j, batch_seg=None, **params):
        energy = self.energy_from_scaled_atomic_properties(Ea, Qa, Dij, Za, idx_i, idx_j, batch_seg)
        forces = -torch.autograd.grad(torch.sum(energy), Ra, create_graph=True)[0]
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
        if batch_seg is None:
            batch_seg = torch.zeros_like(Za)
        #number of atoms per batch (needed for charge scaling)
        Na_per_batch = segment_sum(torch.ones_like(batch_seg, dtype=self.dtype), batch_seg)
        if Q is None: #assume desired total charge zero if not given
            Q = torch.zeros_like(Na_per_batch, dtype=self.dtype)
        #return scaled charges (such that they have the desired total charge)
        return Qa + ((Q - segment_sum(Qa, batch_seg)) / Na_per_batch)[batch_seg]

    #switch function for electrostatic interaction (switches between shielded and unshielded electrostatic interaction)
    def _switch(self, Dij):
        cut = self.sr_cut / 2
        x  = Dij / cut
        x3 = x ** 3
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(Dij < cut, 6 * x5 - 15 * x4 + 10 * x3, torch.ones_like(Dij))

    #calculates the electrostatic energy per atom 
    #for very small distances, the 1/r law is shielded to avoid singularities
    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        #gather charges
        Qi = Qa[idx_i]
        Qj = Qa[idx_j]
        #calculate variants of Dij which we need to calculate
        #the various shileded/non-shielded potentials
        DijS = torch.sqrt(Dij * Dij + 1.0) #shielded distance
        #calculate value of switching function
        switch = self._switch(Dij) #normal switch
        cswitch = 1.0 - switch #complementary switch
        #calculate shielded/non-shielded potentials
        if self.lr_cut is None: #no non-bonded cutoff
            Eele_ordinary = 1.0 / Dij   #ordinary electrostatic energy
            Eele_shielded = 1.0 / DijS  #shielded electrostatic energy
            #combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf * Qi * Qj * (cswitch * Eele_shielded + switch * Eele_ordinary)
        else: #with non-bonded cutoff
            cut = self.lr_cut
            cut2 = self.lr_cut * self.lr_cut
            Eele_ordinary = 1.0 / Dij + Dij / cut2 - 2.0 / cut
            Eele_shielded = 1.0 / DijS + DijS / cut2 - 2.0 / cut
            #combine shielded and ordinary interactions and apply prefactors 
            Eele = self.kehalf * Qi * Qj * (cswitch * Eele_shielded + switch * Eele_ordinary)
            Eele = torch.where(Dij <= cut, Eele, torch.zeros_like(Eele))
        return segment_sum(Eele, idx_i)

    def batch_collate_fn(self, sample):
        feature, label = zip(*sample)
        feature, label = pd.concat(feature, axis=1).T, pd.concat(label, axis=1).T
        batch_input, batch_target = dict(), dict()
        for k, v in feature.items():
            if k == "Q":
                batch_input[k] = torch.tensor(v.to_list(), dtype=self.dtype).reshape(-1)
            if k == "Za":
                batch_input[k] = torch.tensor(np.concatenate(v.to_list()), dtype=torch.long)
            if k == "Ra":
                batch_input[k] = torch.tensor(np.concatenate(v.to_list()), dtype=self.dtype, requires_grad=True)
        for k, v in label.items():
            if k in ["Qa", "F"]:
                batch_target[k] = torch.tensor(np.concatenate(v.to_list()))
            elif k in ["E", "P"]:
                batch_target[k] = torch.tensor(np.array(v.to_list()))
        batch_seg = []
        idx_i = []
        idx_j = []
        split_sections = []
        count = 0
        for i, Za in enumerate(feature["Za"]):
            N = len(Za)
            batch_seg.append(np.ones(N, dtype=int) * i)
            indices = np.indices((N, N-1))
            idx_i.append(indices[0].reshape(-1) + count)
            idx_j.append(
                (indices[1] + np.triu(np.ones((N, N-1)))).reshape(-1) + count
            )
            count += N
            split_sections.append(N)
        batch_input["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long)
        batch_input["idx_i"] = torch.tensor(np.concatenate(idx_i), dtype=torch.long)
        batch_input["idx_j"] = torch.tensor(np.concatenate(idx_j), dtype=torch.long)
        batch_target["split_sections"] = split_sections
        batch_target["atom_type"] = label["atom_type"]
        return batch_input, batch_target
    
    def batch_output_collate_fn(self, output, target):
        net_output, net_target = dict(), dict()
        for k, v in target.items():
            if k in ["Qa", "F"]:
                net_output[k] = torch.split(output[k], target["split_sections"])
                net_target[k] = torch.split(v, target["split_sections"])
            elif k in ["atom_type"]:
                net_output[k] = v
                net_target[k] = v
            elif k in ["E", "P"]:
                net_output[k] = output[k]
                net_target[k] = v
        return net_output, net_target

    def forward(self, task, **net_input):
        Ea, Qa, Dij, nh_loss = self.atomic_properties(**net_input)
        Qa = self.scaled_charges(Qa=Qa, **net_input)
        output = {"nh_loss": nh_loss}
        if "q" in task:
            output["Qa"] = Qa
        if "e" in task:
            energy, forces = self.energy_and_forces_from_scaled_atomic_properties(Ea, Qa, Dij, **net_input)
            output["E"] = energy
            output["F"] = forces
        if "p" in task:
            output["P"] = segment_sum(Qa.unsqueeze(1) * net_input["Ra"], net_input["batch_seg"]) / (self.kehalf * 2)
        return output
   
    @property
    def drop_out(self):
        return self._drop_out
    
    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def saver(self):
        return self._saver
    
    @property
    def embeddings(self):
        return self._embeddings

    @property
    def Eshift(self):
        return self._Eshift

    @property
    def Escale(self):
        return self._Escale
  
    @property
    def Qshift(self):
        return self._Qshift

    @property
    def Qscale(self):
        return self._Qscale

    @property
    def s6(self):
        return self._s6

    @property
    def s8(self):
        return self._s8
    
    @property
    def a1(self):
        return self._a1

    @property
    def a2(self):
        return self._a2

    @property
    def use_electrostatic(self):
        return self._use_electrostatic

    @property
    def use_dispersion(self):
        return self._use_dispersion
    
    @property
    def kehalf(self):
        return self._kehalf
    
    @property
    def d3_autoev(self):
        return self._d3_autoev
    
    @property
    def dtype(self):
        return self._dtype

    @property
    def F(self):
        return self._F

    @property
    def K(self):
        return self._K

    @property
    def sr_cut(self):
        return self._sr_cut

    @property
    def lr_cut(self):
        return self._lr_cut
    
    @property
    def activation_fn(self):
        return self._activation_fn
    
    @property
    def rbf_layer(self):
        return self._rbf_layer

    @property
    def interaction_block(self):
        return self._interaction_block

    @property
    def output_block(self):
        return self._output_block
    
    @property
    def disp_layer(self):
        return self._disp_layer
