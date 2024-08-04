import os
from typing import Dict
import numpy as np
import torch
from torch import Tensor, _masked_softmax
from torch.masked import masked_tensor
from torch.nn import Module, Parameter
from torch.nn.functional import softmax
from ...cutoff import polynomial_cutoff
from ...functional import segment_sum, gather_nd

# parameters
# grimme_d3_tables from https://github.com/MMunibas/PhysNet/commit/e243e2c383b4ac0a9d7062e654d9b4feb76ca859
package_directory = os.path.dirname(os.path.abspath(__file__))
d3_s6 = 1.0000 
d3_s8 = 0.9171 
d3_a1 = 0.3385 
d3_a2 = 2.8830
d3_k1 = 16.000    
d3_k3 = -4.000

# https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/pars.f
d3_c6ab = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "c6ab.npy")))

# d3_k2 has been multiplied on rcov.npy from https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3/dftd3.tgz (dftd3.f)
# adapted from Chemistry–A European Journal, 2009, 15(1), 186–197.
d3_k2 = 4/3
d3_rcov = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "rcov.npy")))

d3_r2r4 = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "r2r4.npy")))
d3_maxc = 5 #maximum number of coordination complexes
d3_maxc2 = d3_maxc * d3_maxc


def _ncoord(Zi: Tensor, Zj: Tensor, Dij: Tensor, idx_i: Tensor, cutoff: float=None, k1: float=d3_k1, rcov: Tensor=d3_rcov) -> Tensor:
    r'''
    Compute coordination numbers (CN) [1] with an inverse damping function [2]

    CN^A = \sum_{B!=A}^{N_{atom}} 1 / (1 + exp(-k_1 * (k_2 * (R_{A,cov} + R_{B,cov}) / r_{AB} - 1)))
    
    where the summation walks through all atom pairs, k_1 was chosen to be 16 for two reasons:
    - for many chemical reactions involving carbon, C-C bond breaks at 1.5-3 Å, for which CN should be 0-1.
    - k_1 must be large enough that very distant atoms are not counted, so that CN doesn't significantly depend on the system size.

    k_2 is a scale factor 4/3, R_{cov} is the scaled covalent (single-bond) radii of atoms,
    r_AB is the distance between the atom A and B.

    Params:
    -----
    Zi: Long tensor of the first atomic number in the pair, shape [N_pair * batch_size]

    Zj: Long tensor of the second atomic number in the pair, shape [N_pair * batch_size]

    Dij: Float tensor of distances in Bohr, shape [N_pair * batch_size]

    idx_i: Long tensor of the first pair indices, shape [N_pair * batch_size]

    cutoff: Cutoff in Bohr for the damping function.

    k1: k_1.

    rcov: Float tensor of k_2 * R_{cov} in Bohr, shape [max_Za, max_Za].

    Returns:
    -----
    ncoord: Float tensor of coordination numbers, shape [N * batch_size]

    References:
    -----
    [1]: J. Chem. Phys. 2010, 132, 154104.

    [2]: J. Chem. Theory Comput. 2019, 15, 3678−3693.
    '''
    rco = rcov[Zi] + rcov[Zj]
    rr = rco.type_as(Dij) / Dij
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None:
        cuton = cutoff - 1
        Dij_ = Dij - cuton
        damp *= torch.where(
            Dij_ > 0,
            polynomial_cutoff(Dij_, cutoff),
            torch.ones_like(Dij)
        )
    return segment_sum(damp, idx_i)


def _getc6(Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor, c6ab: Tensor=d3_c6ab, k3: float=d3_k3) -> Tensor:
    r'''
    Interpolate dispersion coefficients in specific coordination environments [1]

    C_6 coefficient between atom A and B as a function of CN of them 
    is a Gaussian distance weighted average:

    C_6^{AB}(CN^A, CN^B) = Z / W

    where

    Z = \sum_i^{N_A}\sum_j^{N_B} C_{6,ref}^{AB}(CN_i^A, CN_j^B) * L_{ij}

    W = \sum_i^{N_A}\sum_j^{N_B} L_{ij}

    L_{ij} = exp(-k_3 * ((CN^A - CN_i^A) ** 2 + (CN^B - CN_j^B) ** 2))

    C_{6,ref}^{AB}(CN_i^A, CN_j^B) are precomputed values of C_6^{AB} at supporting points CN_i^A and CN_j^B,
    N_A, N_B are the number of supporting points in reference molecules for atoms A and B,
    The choice of k_3 is guided by the basic requirement to get smooth curves that avoid “sharp” parts in the resulting potential 
    and concomitantly display clear plateaus close to integer CN values. 
    Reasonable choices for this value (between 3 and 5) do not significantly affect the results 
    and we take k_3=4 as default.

    Params:
    -----
    Zi: Long tensor of the first atomic number in the pair, shape [N_pair * batch_size]

    Zj: Long tensor of the second atomic number in the pair, shape [N_pair * batch_size]

    nci: Float tensor of the first CN in the pair, shape [N_pair * batch_size]

    ncj: Float tensor of the second CN in the pair, shape [N_pair * batch_size]

    c6ab: Precomputed values of C_6^{AB} at supporting points, shape [max_Za, max_Za, d3_maxc, d3_maxc, 3], where
        [A,B,i,j,0] -> C_{6,ref}^{AB}(CN_i^A, CN_j^B), [A,B,i,j,1] -> CN_i^A, [A,B,i,j,2] -> CN_j^B
        -1 in [A,B,i,j,0] means the reference point doesn't exist

    References:
    -----
    [1]: J. Chem. Phys. 2010, 132, 154104.
    '''
    ZiZj = torch.stack([Zi,Zj],axis=1) # necessary for gatherin
    c6ab_ = gather_nd(c6ab, ZiZj).type_as(nci) #gather the relevant entries from the table

    #calculate c6 coefficients
    nci_ = nci.unsqueeze(-1).repeat(1, d3_maxc2)
    ncj_ = ncj.unsqueeze(-1).repeat(1, d3_maxc2)
    c6ab_ = c6ab_.reshape(-1, d3_maxc2, 3)
    r = (c6ab_[:, :, 1] - nci_) ** 2 + (c6ab_[:, :, 2] - ncj_) ** 2
    weight = softmax(torch.where(c6ab_[:, :, 0] > 0, k3 * r, torch.full_like(c6ab_[:, :, 0], -1e10)), dim=1)
    c6 = torch.sum(weight * c6ab_[:, :, 0], dim=1)
    return c6


def edisp(Za: Tensor, Dij: Tensor, idx_i: Tensor, idx_j: Tensor, cutoff: float=None, s6: float=d3_s6, s8: float=d3_s8, a1: float=d3_a1, a2: float=d3_a2, 
    k3: float=d3_k3, c6ab: Tensor=d3_c6ab, rcov: Tensor=d3_rcov, r2r4: Tensor=d3_r2r4) -> Tensor:
    '''
    compute d3 dispersion energy in Hartree
    r: distance in bohr!
    '''
    Zi = Za[idx_i]
    Zj = Za[idx_j]
    
    nc = _ncoord(Zi, Zj, Dij, idx_i, cutoff=cutoff, rcov=rcov) #coordination numbers
    nci = nc[idx_i]
    ncj = nc[idx_j]
    c6 = _getc6(Zi, Zj, nci, ncj, c6ab=c6ab, k3=k3) #c6 coefficients
    
    c8 = 3 * c6 * r2r4[Zi].type_as(c6) * r2r4[Zj].type_as(c6) #c8 coefficient
    
    #compute all necessary powers of the distance
    r2 = Dij ** 2 #square of distances
    r6 = r2 ** 3
    r8 = r6 * r2
    
    #Becke-Johnson damping, zero-damping introduces spurious repulsion
    #and is therefore not supported/implemented
    tmp = a1 * torch.sqrt(c8 / (c6 + 1e-10) + 1e-10) + a2
    tmp2 = tmp ** 2
    tmp6 = tmp2 ** 3
    tmp8 = tmp6 * tmp2
    e6 = 1 / (r6 + tmp6)
    e8 = 1 / (r8 + tmp8)
    if cutoff is not None:
        cut2 = cutoff ** 2
        cut6 = cut2 ** 3
        cut8 = cut6 * cut2
        cut6tmp6 = cut6 + tmp6
        cut8tmp8 = cut8 + tmp8
        e6 += - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * (Dij / cutoff - 1)
        e8 += - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * (Dij / cutoff - 1)
        e6 = torch.where(Dij < cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(Dij < cutoff, e8, torch.zeros_like(e8))
    e6 = -0.5 * s6 * c6 * e6
    e8 = -0.5 * s8 * c8 * e8
    return segment_sum(e6 + e8, idx_i)


class GrimmeD3EnergyLayer(Module):
    def __init__(self, learnable: bool=True, Hartree_in_E: float=1, Bohr_in_R: float=0.5291772108) -> None:
        '''
        DFT-D3 dispersion correction [1].

        Params:
        -----
        learnable: If `True`, covalent radii,  are learnable.

        Hartree_in_E: The numerical value of one Hartree in the unit of energy targets

        Bohr_in_R: The numerical value of one Bohr in the unit of position features

        References:
        -----
        [1]: J. Chem. Phys. 2010, 132, 154104

        '''
        super().__init__()
        self.d3_c6ab = Parameter(d3_c6ab, requires_grad=learnable)
        self.d3_rcov = Parameter(d3_rcov, requires_grad=learnable)
        self.d3_r2r4 = Parameter(d3_r2r4, requires_grad=learnable)
        self.Hartree_in_E = Hartree_in_E
        self.Bohr_in_R = Bohr_in_R

    def get_e_disp(self, Za: Tensor, Dij: Tensor, idx_i: Tensor, idx_j: Tensor, cutoff: float=None, **kwargs) -> Tensor:
        return self.Hartree_in_E * edisp(
            Za, Dij / self.Bohr_in_R, idx_i, idx_j, cutoff, 
            s6=d3_s6, s8=d3_s8, a1=d3_a1, a2=d3_a2, k3=d3_k3, 
            c6ab=self.d3_c6ab, rcov=self.d3_rcov, r2r4=self.d3_r2r4
        )

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output["E_disp_a"] = self.get_e_disp(**net_input)
        return output