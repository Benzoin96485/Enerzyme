'''
Fused SpookyNet local message passing for inference on CUDA devices.

For pairs e = (i, j) sorted by i (CSR offsets), with radial features
R = [Rs | Rp | Rd] of shape [P, 3F] and node features xs, xp, xd of shape [N, F]:

    s[i]    = sum_e Rs[e] * xs[j]
    p[i, c] = sum_e pij[e, c] * Rp[e] * xp[j]      (c = 0..2)
    d[i, c] = sum_e dij[e, c] * Rd[e] * xd[j]      (c = 0..4)

which is what LocalInteraction computes with [P, 3, F] / [P, 5, F] intermediates.
'''
from typing import Tuple
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _local_message_fwd_kernel(
        R_ptr, pij_ptr, dij_ptr, xs_ptr, xp_ptr, xd_ptr, idx_j_ptr, offsets_ptr,
        s_ptr, p_ptr, d_ptr,
        F: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr,
    ):
        i = tl.program_id(0).to(tl.int64)
        f = tl.program_id(1) * BLOCK_F + tl.arange(0, BLOCK_F)
        fmask = f < F
        start = tl.load(offsets_ptr + i)
        end = tl.load(offsets_ptr + i + 1)
        dtype = R_ptr.dtype.element_ty
        acc_s = tl.zeros([BLOCK_F], dtype=dtype)
        acc_p0 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_p1 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_p2 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_d0 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_d1 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_d2 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_d3 = tl.zeros([BLOCK_F], dtype=dtype)
        acc_d4 = tl.zeros([BLOCK_F], dtype=dtype)
        for e0 in range(start, end, BLOCK_E):
            e = e0 + tl.arange(0, BLOCK_E)
            emask = e < end
            m2 = emask[:, None] & fmask[None, :]
            j = tl.load(idx_j_ptr + e, mask=emask, other=0)
            R_row = R_ptr + e[:, None] * (3 * F) + f[None, :]
            x_row = j[:, None] * F + f[None, :]
            ms = tl.load(R_row, mask=m2, other=0.0) * tl.load(xs_ptr + x_row, mask=m2, other=0.0)
            mp = tl.load(R_row + F, mask=m2, other=0.0) * tl.load(xp_ptr + x_row, mask=m2, other=0.0)
            md = tl.load(R_row + 2 * F, mask=m2, other=0.0) * tl.load(xd_ptr + x_row, mask=m2, other=0.0)
            acc_s += tl.sum(ms, axis=0)
            acc_p0 += tl.sum(tl.load(pij_ptr + e * 3, mask=emask, other=0.0)[:, None] * mp, axis=0)
            acc_p1 += tl.sum(tl.load(pij_ptr + e * 3 + 1, mask=emask, other=0.0)[:, None] * mp, axis=0)
            acc_p2 += tl.sum(tl.load(pij_ptr + e * 3 + 2, mask=emask, other=0.0)[:, None] * mp, axis=0)
            acc_d0 += tl.sum(tl.load(dij_ptr + e * 5, mask=emask, other=0.0)[:, None] * md, axis=0)
            acc_d1 += tl.sum(tl.load(dij_ptr + e * 5 + 1, mask=emask, other=0.0)[:, None] * md, axis=0)
            acc_d2 += tl.sum(tl.load(dij_ptr + e * 5 + 2, mask=emask, other=0.0)[:, None] * md, axis=0)
            acc_d3 += tl.sum(tl.load(dij_ptr + e * 5 + 3, mask=emask, other=0.0)[:, None] * md, axis=0)
            acc_d4 += tl.sum(tl.load(dij_ptr + e * 5 + 4, mask=emask, other=0.0)[:, None] * md, axis=0)
        tl.store(s_ptr + i * F + f, acc_s, mask=fmask)
        p_row = p_ptr + i * (3 * F) + f
        tl.store(p_row, acc_p0, mask=fmask)
        tl.store(p_row + F, acc_p1, mask=fmask)
        tl.store(p_row + 2 * F, acc_p2, mask=fmask)
        d_row = d_ptr + i * (5 * F) + f
        tl.store(d_row, acc_d0, mask=fmask)
        tl.store(d_row + F, acc_d1, mask=fmask)
        tl.store(d_row + 2 * F, acc_d2, mask=fmask)
        tl.store(d_row + 3 * F, acc_d3, mask=fmask)
        tl.store(d_row + 4 * F, acc_d4, mask=fmask)

    @triton.jit
    def _local_message_bwd_kernel(
        R_ptr, pij_ptr, dij_ptr, xs_ptr, xp_ptr, xd_ptr, idx_j_ptr, offsets_ptr,
        gs_ptr, gp_ptr, gd_ptr,
        gR_ptr, gpij_ptr, gdij_ptr, gxs_ptr, gxp_ptr, gxd_ptr,
        F: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_F: tl.constexpr,
    ):
        i = tl.program_id(0).to(tl.int64)
        f = tl.arange(0, BLOCK_F)
        fmask = f < F
        start = tl.load(offsets_ptr + i)
        end = tl.load(offsets_ptr + i + 1)
        gs = tl.load(gs_ptr + i * F + f, mask=fmask, other=0.0)[None, :]
        gp_row = gp_ptr + i * (3 * F) + f
        gp0 = tl.load(gp_row, mask=fmask, other=0.0)[None, :]
        gp1 = tl.load(gp_row + F, mask=fmask, other=0.0)[None, :]
        gp2 = tl.load(gp_row + 2 * F, mask=fmask, other=0.0)[None, :]
        gd_row = gd_ptr + i * (5 * F) + f
        gd0 = tl.load(gd_row, mask=fmask, other=0.0)[None, :]
        gd1 = tl.load(gd_row + F, mask=fmask, other=0.0)[None, :]
        gd2 = tl.load(gd_row + 2 * F, mask=fmask, other=0.0)[None, :]
        gd3 = tl.load(gd_row + 3 * F, mask=fmask, other=0.0)[None, :]
        gd4 = tl.load(gd_row + 4 * F, mask=fmask, other=0.0)[None, :]
        for e0 in range(start, end, BLOCK_E):
            e = e0 + tl.arange(0, BLOCK_E)
            emask = e < end
            m2 = emask[:, None] & fmask[None, :]
            j = tl.load(idx_j_ptr + e, mask=emask, other=0)
            R_row = R_ptr + e[:, None] * (3 * F) + f[None, :]
            gR_row = gR_ptr + e[:, None] * (3 * F) + f[None, :]
            x_row = j[:, None] * F + f[None, :]
            p0 = tl.load(pij_ptr + e * 3, mask=emask, other=0.0)[:, None]
            p1 = tl.load(pij_ptr + e * 3 + 1, mask=emask, other=0.0)[:, None]
            p2 = tl.load(pij_ptr + e * 3 + 2, mask=emask, other=0.0)[:, None]
            d0 = tl.load(dij_ptr + e * 5, mask=emask, other=0.0)[:, None]
            d1 = tl.load(dij_ptr + e * 5 + 1, mask=emask, other=0.0)[:, None]
            d2 = tl.load(dij_ptr + e * 5 + 2, mask=emask, other=0.0)[:, None]
            d3 = tl.load(dij_ptr + e * 5 + 3, mask=emask, other=0.0)[:, None]
            d4 = tl.load(dij_ptr + e * 5 + 4, mask=emask, other=0.0)[:, None]
            # s channel
            rs = tl.load(R_row, mask=m2, other=0.0)
            xs = tl.load(xs_ptr + x_row, mask=m2, other=0.0)
            tl.store(gR_row, gs * xs, mask=m2)
            tl.atomic_add(gxs_ptr + x_row, gs * rs, mask=m2)
            # p channel
            rp = tl.load(R_row + F, mask=m2, other=0.0)
            xp = tl.load(xp_ptr + x_row, mask=m2, other=0.0)
            gmp = p0 * gp0 + p1 * gp1 + p2 * gp2
            tl.store(gR_row + F, gmp * xp, mask=m2)
            tl.atomic_add(gxp_ptr + x_row, gmp * rp, mask=m2)
            mp = rp * xp
            tl.store(gpij_ptr + e * 3, tl.sum(gp0 * mp, axis=1), mask=emask)
            tl.store(gpij_ptr + e * 3 + 1, tl.sum(gp1 * mp, axis=1), mask=emask)
            tl.store(gpij_ptr + e * 3 + 2, tl.sum(gp2 * mp, axis=1), mask=emask)
            # d channel
            rd = tl.load(R_row + 2 * F, mask=m2, other=0.0)
            xd = tl.load(xd_ptr + x_row, mask=m2, other=0.0)
            gmd = d0 * gd0 + d1 * gd1 + d2 * gd2 + d3 * gd3 + d4 * gd4
            tl.store(gR_row + 2 * F, gmd * xd, mask=m2)
            tl.atomic_add(gxd_ptr + x_row, gmd * rd, mask=m2)
            md = rd * xd
            tl.store(gdij_ptr + e * 5, tl.sum(gd0 * md, axis=1), mask=emask)
            tl.store(gdij_ptr + e * 5 + 1, tl.sum(gd1 * md, axis=1), mask=emask)
            tl.store(gdij_ptr + e * 5 + 2, tl.sum(gd2 * md, axis=1), mask=emask)
            tl.store(gdij_ptr + e * 5 + 3, tl.sum(gd3 * md, axis=1), mask=emask)
            tl.store(gdij_ptr + e * 5 + 4, tl.sum(gd4 * md, axis=1), mask=emask)


FWD_BLOCK_E = 16
FWD_BLOCK_F = 64
BWD_BLOCK_E = 8


class FusedLocalMessage(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, R: Tensor, pij: Tensor, dij: Tensor, xs: Tensor, xp: Tensor, xd: Tensor,
        idx_j: Tensor, offsets: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N, F = xs.shape
        R, pij, dij = R.contiguous(), pij.contiguous(), dij.contiguous()
        xs, xp, xd = xs.contiguous(), xp.contiguous(), xd.contiguous()
        s = xs.new_empty(N, F)
        p = xs.new_empty(N, 3, F)
        d = xs.new_empty(N, 5, F)
        if N > 0:
            grid = (N, triton.cdiv(F, FWD_BLOCK_F))
            _local_message_fwd_kernel[grid](
                R, pij, dij, xs, xp, xd, idx_j, offsets, s, p, d,
                F=F, BLOCK_E=FWD_BLOCK_E, BLOCK_F=FWD_BLOCK_F,
            )
        ctx.save_for_backward(R, pij, dij, xs, xp, xd, idx_j, offsets)
        return s, p, d

    @staticmethod
    @once_differentiable
    def backward(ctx, gs: Tensor, gp: Tensor, gd: Tensor):
        R, pij, dij, xs, xp, xd, idx_j, offsets = ctx.saved_tensors
        N, F = xs.shape
        gs = torch.zeros_like(xs) if gs is None else gs.contiguous()
        gp = xs.new_zeros(N, 3, F) if gp is None else gp.contiguous()
        gd = xs.new_zeros(N, 5, F) if gd is None else gd.contiguous()
        gR = torch.empty_like(R)
        gpij = torch.empty_like(pij)
        gdij = torch.empty_like(dij)
        gxs, gxp, gxd = torch.zeros_like(xs), torch.zeros_like(xp), torch.zeros_like(xd)
        if N > 0:
            _local_message_bwd_kernel[(N,)](
                R, pij, dij, xs, xp, xd, idx_j, offsets, gs, gp, gd,
                gR, gpij, gdij, gxs, gxp, gxd,
                F=F, BLOCK_E=BWD_BLOCK_E, BLOCK_F=triton.next_power_of_2(F),
            )
        return gR, gpij, gdij, gxs, gxp, gxd, None, None


def csr_offsets(idx_i: Tensor, N: int) -> Tensor:
    '''Row offsets of pairs sorted by `idx_i`.'''
    return torch.searchsorted(idx_i, torch.arange(N + 1, device=idx_i.device, dtype=idx_i.dtype))


def fused_local_message(
    R: Tensor, pij: Tensor, dij: Tensor, xs: Tensor, xp: Tensor, xd: Tensor,
    idx_j: Tensor, offsets: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    return FusedLocalMessage.apply(R, pij, dij, xs, xp, xd, idx_j, offsets)
