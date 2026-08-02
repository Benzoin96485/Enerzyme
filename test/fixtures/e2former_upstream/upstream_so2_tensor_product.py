# Vendored from IQuestLab/UBio-MolFM (MIT) for offline parity tests.
# Source: src/molfm/models/e2former/wigner6j/so2_tensor_product.py
# Do NOT use as a production dependency.

# -*- coding: utf-8 -*-

from e3nn import o3
from itertools import groupby
import torch
from torch import nn
import random
import e3nn
import numpy as np

def drop_duplicates_random(arr,key):
    result = []
    seen = set()
    indices = list(range(len(arr)))
    random.shuffle(indices)  
    
    for i in indices:
        val = key(arr[i])
        if val not in seen:
            result.append(arr[i])
            seen.add(val)
    return result

from dataclasses import dataclass

from e3nn.o3._wigner import _su2_clebsch_gordan,change_basis_real_to_complex

def wigner_3j_wonorm(l1: int, l2: int, l3: int,norm = False) -> torch.Tensor:
    assert abs(l2 - l3) <= l1 <= l2 + l3
    assert isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int)

    Q1 = change_basis_real_to_complex(l1, dtype=torch.float64)
    Q2 = change_basis_real_to_complex(l2, dtype=torch.float64)
    Q3 = change_basis_real_to_complex(l3, dtype=torch.float64)
    C = _su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, torch.conj(Q3.T), C)

    # make it real
    assert torch.all(torch.abs(torch.imag(C)) < 1e-5)
    C = torch.real(C)
    if norm:
        # normalization
        C = C / torch.norm(C)
    return C

@dataclass
class Instruction():
    l1: int = None
    l2: int = None
    l3: int = None
    start_idx_l1: int = None
    start_idx_l3: int = None

class Linear_s3_l3sequential(torch.nn.Module):
    """
    features: [..., n_rows, C]  where n_rows = sum_{(l,m)} m*(2l+1)
    lmax: list of tuples [(l, mul), ...]
      - Each l gate produces a single [C] vector and broadcasts it across
        all m*(2l+1) rows for that l.
      - l=0 applies SiLU(features_l0) directly.
    """

    def __init__(self, sphere_channels, lmax,out_c, act_scalars="silu", act_vector="sigmoid"):
        super().__init__()

        self.mul0 = 1
        if isinstance(lmax, int):
            self.lmax = [(l, 1) for l in range(lmax + 1)]
        else:
            self.lmax = lmax
            for i in range(len(lmax)-1):
                if lmax[i+1][0]<lmax[i][0]:
                    raise ValueError
            if lmax[0][0]!=0:
                raise ValueError
            self.mul0 = lmax[0][1]
        self.l3_sequential = lmax
        self.out_linear = nn.ModuleList([])
        for tmp_l3,tmp_l3_cnt in self.l3_sequential:
            self.out_linear.append(nn.Linear(tmp_l3_cnt*sphere_channels,out_c,bias = False))

    def forward(self,so2_irreps):
        bs = so2_irreps.shape[0]

        start_idx = 0
        end_idx = None
        out = []
        for idx,(tmp_l3,tmp_l3_cnt) in enumerate(self.l3_sequential):
            end_idx = start_idx+(tmp_l3*2+1)*tmp_l3_cnt
            out.append(self.out_linear[idx](
                    so2_irreps[:,start_idx:end_idx].reshape(
                        bs,tmp_l3_cnt,tmp_l3*2+1,-1).permute(0,2,1,3).reshape(bs,tmp_l3*2+1,-1)
                ))
            
            start_idx = end_idx
        out = torch.cat(out,dim = 1)
        return out
    
class NonLinear_s3_l3sequential(torch.nn.Module):
    """
    features: [..., n_rows, C]  where n_rows = sum_{(l,m)} m*(2l+1)
    lmax: list of tuples [(l, mul), ...]
      - Each l gate produces a single [C] vector and broadcasts it across
        all m*(2l+1) rows for that l.
      - l=0 applies SiLU(features_l0) directly.
    """

    def __init__(self, sphere_channels, lmax,out_c, act_scalars="silu", act_vector="sigmoid"):
        super().__init__()
        self.sphere_channels = int(sphere_channels)

        self.mul0 = 1
        if isinstance(lmax, int):
            self.lmax = [(l, 1) for l in range(lmax + 1)]
        else:
            self.lmax = lmax
            for i in range(len(lmax)-1):
                if lmax[i+1][0]<lmax[i][0]:
                    raise ValueError
            if lmax[0][0]!=0:
                raise ValueError
            self.mul0 = lmax[0][1]
        self.l3_sequential = lmax
        self.out_linear = nn.ModuleList([])
        for tmp_l3,tmp_l3_cnt in self.l3_sequential:
            self.out_linear.append(nn.Linear(tmp_l3_cnt*sphere_channels,out_c,bias = False))
        self._gated_ls = [m for (l, m) in self.lmax]

        self.gates = torch.nn.Linear(self.mul0*self.sphere_channels, self.sphere_channels * sum(self._gated_ls))

        bound = 1 / math.sqrt(self.sphere_channels)
        torch.nn.init.uniform_(self.gates.weight, -bound, bound)
        if self.gates.bias is not None:
            torch.nn.init.zeros_(self.gates.bias)

        if act_scalars == "silu":
            self.act_scalars = nn.Sequential(nn.Linear(self.mul0*self.sphere_channels,out_c),
                                    torch.nn.SiLU())
                                            
        else:
            raise ValueError("in Gate_s3, only support silu for scalars")

        if act_vector == "sigmoid":
            self.act_vector = torch.nn.Sigmoid()
        else:
            raise ValueError("in Gate_s3, only support sigmoid for vector")

    def __repr__(self):
        return f"{self.__class__.__name__}(C={self.sphere_channels}, lmax={self.lmax})"

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        bs = features.shape[0]
        x0 = features[:, :self.mul0, :]

        g_all = self.gates(x0.reshape(bs,-1)).view(bs, -1, self.sphere_channels)

        gate_start = 0
        
        start_idx = 0
        
        out = []
        for idx,(tmp_l3,tmp_l3_cnt) in enumerate(self.l3_sequential):
            end_idx = start_idx+(tmp_l3*2+1)*tmp_l3_cnt
            if idx==0:
                out.append(self.act_scalars(g_all[:,gate_start:gate_start+tmp_l3_cnt].reshape(bs,1,-1)))
            else:
                out.append(self.out_linear[idx](
                        (features[:,start_idx:end_idx].reshape(
                            bs,tmp_l3_cnt,tmp_l3*2+1,-1)*self.act_vector(
                                g_all[:,gate_start:gate_start+tmp_l3_cnt].unsqueeze(dim = 2)
                                )).permute(0,2,1,3).reshape(bs,tmp_l3*2+1,-1)
                    ))
            gate_start += tmp_l3_cnt
            start_idx = end_idx

        y = torch.cat(out, dim=1)  # [B, n_rows, C]
        return y


class SO2_TP_givenl2(torch.nn.Module):
    def __init__(self,l1_list,l2,l3_legal = None,in_c=None,out_c=None,drop_path=False,with_linear = False):
        '''
        e.g. 128x0e+128x1e+128x2e+128x2e+128x3e
        l1_list: [0,1,2,2,3]
        l2: int that is order of ri or rj or rij
        '''

        super().__init__()

        legal_l1l2l3 = []
        input_start_idx = 0
        for l1 in l1_list:
            for l3 in range(abs(l1-l2),l1+l2+1):
                if not (l3_legal is None):
                    if l3 not in l3_legal:
                        continue
                # ############# ############# #############
                # if you want to drop path, please drop here.
                legal_l1l2l3.append(Instruction(
                    l1 = l1,
                    l2 = l2,
                    l3 = l3,
                    start_idx_l1 = input_start_idx))

            input_start_idx += 2*l1+1

        if drop_path:legal_l1l2l3 = drop_duplicates_random(legal_l1l2l3,key=lambda x:x.l3)
        
        self.legal_l1l2l3 = sorted(legal_l1l2l3,key=lambda x: x.l3)
        # ############# ############# ############# ########
        self.l3_sequential = [[k, len(list(g))] for k, g in groupby([tmp.l3 for tmp in self.legal_l1l2l3])]
        self.out_linear = nn.ModuleList([])
        self.with_linear = with_linear
        self.in_c = in_c
        self.out_c = out_c

        # self.out_func = NonLinear_s3_l3sequential(in_c, self.l3_sequential, self.out_c)
        # self.out_func = Linear_s3_l3sequential(in_c, self.l3_sequential, self.out_c)
        if with_linear:
            if in_c is None or out_c is None:
                raise ValueError("SO2_TP_givenl2 must set in_c and out_c")
            for tmp_l3,tmp_l3_cnt in self.l3_sequential:
                self.out_linear.append(nn.Linear(tmp_l3_cnt*in_c,out_c,bias = False))

        self.path_norm = []
        for tmp_l3,tmp_l3_cnt in self.l3_sequential:
            self.path_norm.append(torch.ones(2 * tmp_l3 + 1) * tmp_l3_cnt)
        self.path_norm = nn.Parameter(1/ torch.sqrt(
                                    torch.cat(self.path_norm,dim = 0).reshape(1,-1,1)),
                                    requires_grad=False
                                    )


        selected_idx = []
        CG_weight = []

        output_start_idx = 0
        for ins in self.legal_l1l2l3:
            l1,l2,l3,start_idx = ins.l1,ins.l2,ins.l3,ins.start_idx_l1
            ins.start_idx_l3 = output_start_idx
            C = wigner_3j_wonorm(l1, l2, l3,norm=False)[:,l2].numpy()
            # C = (C/torch.norm(C)).numpy()
            
            # rows,cols = np.nonzero(C[:,l2])
            for m3 in range(2*l3+1):
                m1 = np.nonzero(C[:,m3])[0]
                if len(m1) == 0:
                    selected_idx.append(0)
                    CG_weight.append(0)
                elif len(m1) == 1:
                    selected_idx.append(m1[0]+start_idx)
                    CG_weight.append(C[m1[0],m3])
                else:
                    raise ValueError("CG coeff is wrong")
            output_start_idx += 2*l3+1
            # else:
            #     selected_idx
            #     CG_weight.append()
        self.selected_idx = torch.nn.Parameter(torch.Tensor(selected_idx).long(),requires_grad=False)
        self.CG_weight = torch.nn.Parameter(torch.Tensor(CG_weight).float(),requires_grad=False)

    def forward(self,in_irreps,with_linear=True):
        """
        in_irreps: bs * (lmax**2) * in_channel
        """
        bs = in_irreps.shape[0] # 
        
        so2_irreps = in_irreps[:,self.selected_idx]*self.CG_weight.reshape(1,-1,1)
        if with_linear and self.with_linear:
            return self.forward_linear(so2_irreps)
        else:
            return so2_irreps
        
    def forward_linear(self,so2_irreps):
        bs = so2_irreps.shape[0] # 

        start_idx = 0
        end_idx = None
        out = []
        for idx,(tmp_l3,tmp_l3_cnt) in enumerate(self.l3_sequential):
            end_idx = start_idx+(tmp_l3*2+1)*tmp_l3_cnt
            out.append(self.out_linear[idx](
                    so2_irreps[:,start_idx:end_idx].reshape(
                        bs,tmp_l3_cnt,tmp_l3*2+1,-1).permute(0,2,1,3).reshape(bs,tmp_l3*2+1,-1)
                ))

            start_idx = end_idx
        return torch.cat(out,dim = 1) * self.path_norm
        # out = self.out_func(so2_irreps)
        # return out * self.path_norm




class E2TensorProductSO2_FirstOrder(torch.nn.Module):
    """Equivariant tensor product layer that can handle arbitrary order spherical harmonics.

    This module implements a generalized version of the E2TensorProduct that can work with
    any order of spherical harmonics. It combines multiple tensor products and Wigner 6-j
    symbols to maintain equivariance.

    Args:
        irreps_in (str): Input irreducible representations specification
        irreps_out (str): Output irreducible representations specification
        head (int): Number of attention heads
        order (int): Order of spherical harmonics to use
        learnable_weight (bool, optional): Whether weights should be learnable. Defaults to True.
        connection_mode (str, optional): Connection mode - 'uvw' or 'uvu'. Defaults to 'uvw'.
        path_normalization (str, optional): Type of path normalization. Defaults to 'element'.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order,
        # learnable_weight=True,
        # connection_mode="uvw",
        # path_normalization="element",
        **kwargs
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]
        self.head = head

        self.so2_tp = SO2_TP_givenl2(l1_list=range(self.lmax+1),
                                    l2=1,
                                    l3_legal=range(self.lmax+1),
                                    in_c=self.in_c,
                                    out_c=self.out_c,
                                    drop_path=False,
                                    with_linear = True)
    def forward(
            self,
            pos,
            exp_pos,
            h,
            exp_h,
            alpha_ij_divr,
            triton_kernel= None,
            f_sparse_idx_expnode=None,
            batched_data={},
        ):
        def _bmm_with_wigner_double(wigner, feat):
            out = torch.bmm(wigner.double(), feat.double())
            return out.to(feat.dtype)

        f_N1 = pos.shape[0]
        f_N2 = exp_pos.shape[0]
        # alpha_ij_divr  f_N1*topK*heads
        # h_e2_b = exp_h
        # wigner = batched_data["wigner"]
        # wigner_inv = batched_data["wigner_inv"]
        # TODO: wigner and wigner_inv set to exp_wigner and exp_wigner_inv
        h_e2_b = torch.bmm(batched_data["wigner_exp"], exp_h)
        h_e2_b = self.so2_tp(h_e2_b,with_linear=False)*torch.norm(exp_pos,dim=-1).reshape(-1,1,1)
        h_e2_b = torch.bmm(batched_data["wigner_inv_exp"], h_e2_b)


        dim_b = h_e2_b.shape[1]
        merged_ab = torch.cat([h_e2_b,exp_h],dim = 1).reshape(f_N2,-1,self.head)


        if triton_kernel is not None:
            merged_ab = triton_kernel(value=merged_ab, alpha=alpha_ij_divr)
            merged_ab = merged_ab.reshape(f_N1,-1,self.in_c)
            h_e2_b,h_e2_a = merged_ab[:,:dim_b], merged_ab[:,dim_b:]
        else:
            f_sparse_idx_expnode = batched_data["f_sparse_idx_expnode"]
            merged_ab = merged_ab[f_sparse_idx_expnode]*alpha_ij_divr.unsqueeze(dim=2)
            merged_ab = torch.sum(merged_ab,dim = 1)
            merged_ab = merged_ab.reshape(f_N1,-1,self.in_c)
            h_e2_b,h_e2_a = merged_ab[:,:dim_b], merged_ab[:,dim_b:]

        h_e2_a = torch.bmm(batched_data["wigner"], h_e2_a)
        h_e2_a = self.so2_tp(h_e2_a,with_linear=False)*torch.norm(pos,dim=-1).reshape(-1,1,1)
        h_e2_a = torch.bmm(batched_data["wigner_inv"], h_e2_a)

        h_e2 = self.so2_tp.forward_linear(h_e2_a-h_e2_b)

        return h_e2


