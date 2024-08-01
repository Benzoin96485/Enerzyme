'''
Grimme, Stefan, et al. 
"A consistent and accurate ab initio parametrization of density functional dispersion correction (DFT-D) for the 94 elements H-Pu." 
The Journal of chemical physics 132.15 (2010).
'''
import os
from typing import Dict
import numpy as np
import torch
from torch import nn, Tensor
from ...functional import segment_sum, gather_nd


package_directory = os.path.dirname(os.path.abspath(__file__))

d3_s6 = 1.0000 
d3_s8 = 0.9171 
d3_a1 = 0.3385 
d3_a2 = 2.8830
d3_k1 = 16.000
d3_k2 = 4/3
d3_k3 = -4.000

d3_c6ab = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "c6ab.npy")))
d3_rcov = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "rcov.npy")))
d3_r2r4 = torch.tensor(np.load(os.path.join(package_directory, "grimme_d3_tables", "r2r4.npy")))
d3_maxc = 5 #maximum number of coordination complexes


def _smootherstep(r, cutoff):
    '''
    computes a smooth step from 1 to 0 starting at 1 bohr
    before the cutoff
    '''
    cuton = cutoff - 1
    x = cutoff - r
    x2 = x ** 2
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return torch.where(
        r <= cuton, 
        torch.ones_like(x), 
        torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3)
    )


def _ncoord(Zi, Zj, r, idx_i, cutoff=None, k1=d3_k1, rcov=d3_rcov):
    '''
    compute coordination numbers by adding an inverse damping function
    '''
    rco = rcov[Zi] + rcov[Zj]
    rr = rco.type_as(r) / r
    damp = 1.0 / (1.0 + torch.exp(-k1 * (rr - 1.0)))
    if cutoff is not None:
        damp *= _smootherstep(r, cutoff)
    return segment_sum(damp, idx_i)


def _getc6_v2(ZiZj, nci, ncj, c6ab=d3_c6ab, k3=d3_k3):
    '''
    interpolate c6
    '''
    #gather the relevant entries from the table
    c6ab_ = gather_nd(c6ab, ZiZj).type_as(nci)
    #calculate c6 coefficients
    d3_maxc2 = d3_maxc * d3_maxc
    nci_ = nci.unsqueeze(-1).repeat(1, d3_maxc2)
    ncj_ = ncj.unsqueeze(-1).repeat(1, d3_maxc2)
    c6ab_ = c6ab_.reshape(-1, d3_maxc2, 3)
    r = (c6ab_[:, :, 1] - nci_) ** 2 + (c6ab_[:, :, 2] - ncj_) ** 2
    weight = torch.nn.functional.softmax(k3 * r, dim=1)
    c6 = torch.sum(weight * c6ab_[:, :, 0], dim=1)
    return c6


def edisp(Z, r, idx_i, idx_j, cutoff=None, r2=None, 
    r6=None, r8=None, s6=d3_s6, s8=d3_s8, a1=d3_a1, a2=d3_a2, 
    k3=d3_k3, c6ab=d3_c6ab, rcov=d3_rcov, r2r4=d3_r2r4, eps=1e-10):
    '''
    compute d3 dispersion energy in Hartree
    r: distance in bohr!
    '''
    #compute all necessary quantities
    Zi = Z[idx_i]
    Zj = Z[idx_j]
    ZiZj = torch.stack([Zi,Zj],axis=1) #necessary for gatherin
    nc = _ncoord(Zi, Zj, r, idx_i, cutoff=cutoff, rcov=rcov) #coordination numbers
    nci = nc[idx_i]
    ncj = nc[idx_j]
    c6 = _getc6_v2(ZiZj, nci, ncj, c6ab=c6ab, k3=k3) #c6 coefficients
    
    c8 = 3 * c6 * r2r4[Zi].type_as(c6) * r2r4[Zj].type_as(c6) #c8 coefficient
    
    #compute all necessary powers of the distance
    if r2 is None:
        r2 = r ** 2 #square of distances
    if r6 is None:
        r6 = r2 ** 3
    if r8 is None:
        r8 = r6 * r2
    
    #Becke-Johnson damping, zero-damping introduces spurious repulsion
    #and is therefore not supported/implemented
    tmp = a1 * torch.sqrt(c8 / (c6 + eps) + eps) + a2
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
        e6 += - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * (r / cutoff - 1)
        e8 += - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * (r / cutoff - 1)
        e6 = torch.where(r < cutoff, e6, torch.zeros_like(e6))
        e8 = torch.where(r < cutoff, e8, torch.zeros_like(e8))
    e6 = -0.5 * s6 * c6 * e6
    e8 = -0.5 * s8 * c8 * e8
    return segment_sum(e6 + e8, idx_i)


class GrimmeD3EnergyLayer(torch.nn.Module):
    def __init__(self, learnable: bool=True, Hartree_in_E: float=1, Bohr_in_R: float=0.5291772108) -> None:
        super().__init__()
        self.d3_c6ab = nn.Parameter(d3_c6ab, requires_grad=learnable)
        self.d3_rcov = nn.Parameter(d3_rcov, requires_grad=learnable)
        self.d3_r2r4 = nn.Parameter(d3_r2r4, requires_grad=learnable)
        self.Hartree_in_E = Hartree_in_E
        self.Bohr_in_R = Bohr_in_R

    def get_e_disp(self, Za: Tensor, Dij: Tensor, idx_i: Tensor, idx_j: Tensor, cutoff: float=None, **kwargs) -> Tensor:
        return self.Hartree_in_E * edisp(
            Za, Dij / self.Bohr_in_R, idx_i, idx_j, cutoff, 
            r2=None, r6=None, r8=None, 
            s6=d3_s6, s8=d3_s8, a1=d3_a1, a2=d3_a2, k3=d3_k3, 
            c6ab=self.d3_c6ab, rcov=self.d3_rcov, r2r4=self.d3_r2r4, 
            eps=1e-10
        )

    def forward(self, net_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = net_input.copy()
        output["E_disp_a"] = self.get_e_disp(**net_input)
        return output