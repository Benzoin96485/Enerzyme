# Adapted from IQuestLab/UBio-MolFM (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""EAAS / SO(2) first-order tensor product for E2Former-V2.

Implements align → sparse CG re-index (P) → inverse align for
``h ⊗ R(r)`` with geometric degree ``ℓ_f = 1`` (Huang et al., 2026).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import groupby
from typing import Optional

import e3nn
import numpy as np
import torch
from e3nn import o3
from e3nn.o3._wigner import _su2_clebsch_gordan, change_basis_real_to_complex
from torch import nn


def drop_duplicates_random(arr, key):
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


def wigner_3j_wonorm(l1: int, l2: int, l3: int, norm: bool = False) -> torch.Tensor:
    assert abs(l2 - l3) <= l1 <= l2 + l3
    Q1 = change_basis_real_to_complex(l1, dtype=torch.float64)
    Q2 = change_basis_real_to_complex(l2, dtype=torch.float64)
    Q3 = change_basis_real_to_complex(l3, dtype=torch.float64)
    C = _su2_clebsch_gordan(l1, l2, l3).to(dtype=torch.complex128)
    C = torch.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, torch.conj(Q3.T), C)
    assert torch.all(torch.abs(torch.imag(C)) < 1e-5)
    C = torch.real(C)
    if norm:
        C = C / torch.norm(C)
    return C


@dataclass
class Instruction:
    l1: int = None
    l2: int = None
    l3: int = None
    start_idx_l1: int = None
    start_idx_l3: int = None


class SO2_TP_givenl2(torch.nn.Module):
    """Sparse CG re-index (EAAS operator P) for fixed geometric degree ``l2``."""

    def __init__(
        self,
        l1_list,
        l2,
        l3_legal=None,
        in_c=None,
        out_c=None,
        drop_path=False,
        with_linear=False,
    ):
        super().__init__()
        legal_l1l2l3 = []
        input_start_idx = 0
        for l1 in l1_list:
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if l3_legal is not None and l3 not in l3_legal:
                    continue
                legal_l1l2l3.append(
                    Instruction(l1=l1, l2=l2, l3=l3, start_idx_l1=input_start_idx)
                )
            input_start_idx += 2 * l1 + 1

        if drop_path:
            legal_l1l2l3 = drop_duplicates_random(legal_l1l2l3, key=lambda x: x.l3)

        self.legal_l1l2l3 = sorted(legal_l1l2l3, key=lambda x: x.l3)
        self.l3_sequential = [
            [k, len(list(g))] for k, g in groupby([tmp.l3 for tmp in self.legal_l1l2l3])
        ]
        self.out_linear = nn.ModuleList([])
        self.with_linear = with_linear
        self.in_c = in_c
        self.out_c = out_c

        if with_linear:
            if in_c is None or out_c is None:
                raise ValueError("SO2_TP_givenl2 must set in_c and out_c")
            for tmp_l3, tmp_l3_cnt in self.l3_sequential:
                self.out_linear.append(
                    nn.Linear(tmp_l3_cnt * in_c, out_c, bias=False)
                )

        path_norm = []
        for tmp_l3, tmp_l3_cnt in self.l3_sequential:
            path_norm.append(torch.ones(2 * tmp_l3 + 1) * tmp_l3_cnt)
        self.path_norm = nn.Parameter(
            1
            / torch.sqrt(torch.cat(path_norm, dim=0).reshape(1, -1, 1)),
            requires_grad=False,
        )

        selected_idx = []
        CG_weight = []
        output_start_idx = 0
        for ins in self.legal_l1l2l3:
            l1, l2_i, l3, start_idx = ins.l1, ins.l2, ins.l3, ins.start_idx_l1
            ins.start_idx_l3 = output_start_idx
            C = wigner_3j_wonorm(l1, l2_i, l3, norm=False)[:, l2_i].numpy()
            for m3 in range(2 * l3 + 1):
                m1 = np.nonzero(C[:, m3])[0]
                if len(m1) == 0:
                    selected_idx.append(0)
                    CG_weight.append(0.0)
                elif len(m1) == 1:
                    selected_idx.append(int(m1[0] + start_idx))
                    CG_weight.append(float(C[m1[0], m3]))
                else:
                    raise ValueError("CG coeff is wrong")
            output_start_idx += 2 * l3 + 1

        self.selected_idx = torch.nn.Parameter(
            torch.tensor(selected_idx, dtype=torch.long), requires_grad=False
        )
        self.CG_weight = torch.nn.Parameter(
            torch.tensor(CG_weight, dtype=torch.float32), requires_grad=False
        )

    def forward(self, in_irreps, with_linear=True):
        so2_irreps = in_irreps[:, self.selected_idx] * self.CG_weight.reshape(1, -1, 1)
        if with_linear and self.with_linear:
            return self.forward_linear(so2_irreps)
        return so2_irreps

    def forward_linear(self, so2_irreps):
        bs = so2_irreps.shape[0]
        start_idx = 0
        out = []
        for idx, (tmp_l3, tmp_l3_cnt) in enumerate(self.l3_sequential):
            end_idx = start_idx + (tmp_l3 * 2 + 1) * tmp_l3_cnt
            out.append(
                self.out_linear[idx](
                    so2_irreps[:, start_idx:end_idx]
                    .reshape(bs, tmp_l3_cnt, tmp_l3 * 2 + 1, -1)
                    .permute(0, 2, 1, 3)
                    .reshape(bs, tmp_l3 * 2 + 1, -1)
                )
            )
            start_idx = end_idx
        return torch.cat(out, dim=1) * self.path_norm


class E2TensorProductSO2_FirstOrder(torch.nn.Module):
    """First-order equivariant value path via EAAS (SO2 TP + on-the-fly Wigner)."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        head,
        order=1,
        **kwargs,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.order = order
        self.in_c = o3.Irreps(self.irreps_in)[0][0]
        self.out_c = o3.Irreps(self.irreps_out)[0][0]
        self.lmax = e3nn.o3.Irreps(irreps_in)[-1][1][0]
        self.head = head
        self.so2_tp = SO2_TP_givenl2(
            l1_list=range(self.lmax + 1),
            l2=1,
            l3_legal=range(self.lmax + 1),
            in_c=self.in_c,
            out_c=self.out_c,
            drop_path=False,
            with_linear=True,
        )

    def forward(
        self,
        pos,
        exp_pos,
        h,
        exp_h,
        alpha_ij_divr,
        triton_kernel=None,
        f_sparse_idx_expnode=None,
        batched_data=None,
    ):
        if batched_data is None:
            batched_data = {}
        f_n1 = pos.shape[0]
        f_n2 = exp_pos.shape[0]

        h_e2_b = torch.bmm(batched_data["wigner_exp"], exp_h)
        h_e2_b = self.so2_tp(h_e2_b, with_linear=False) * torch.norm(
            exp_pos, dim=-1
        ).reshape(-1, 1, 1)
        h_e2_b = torch.bmm(batched_data["wigner_inv_exp"], h_e2_b)

        dim_b = h_e2_b.shape[1]
        merged_ab = torch.cat([h_e2_b, exp_h], dim=1).reshape(f_n2, -1, self.head)

        if triton_kernel is not None:
            merged_ab = triton_kernel(value=merged_ab, alpha=alpha_ij_divr)
            merged_ab = merged_ab.reshape(f_n1, -1, self.in_c)
            h_e2_b, h_e2_a = merged_ab[:, :dim_b], merged_ab[:, dim_b:]
        else:
            idx = (
                f_sparse_idx_expnode
                if f_sparse_idx_expnode is not None
                else batched_data["f_sparse_idx_expnode"]
            )
            merged_ab = merged_ab[idx] * alpha_ij_divr.unsqueeze(dim=2)
            merged_ab = torch.sum(merged_ab, dim=1)
            merged_ab = merged_ab.reshape(f_n1, -1, self.in_c)
            h_e2_b, h_e2_a = merged_ab[:, :dim_b], merged_ab[:, dim_b:]

        h_e2_a = torch.bmm(batched_data["wigner"], h_e2_a)
        h_e2_a = self.so2_tp(h_e2_a, with_linear=False) * torch.norm(
            pos, dim=-1
        ).reshape(-1, 1, 1)
        h_e2_a = torch.bmm(batched_data["wigner_inv"], h_e2_a)
        return self.so2_tp.forward_linear(h_e2_a - h_e2_b)
