# Adapted from IQuestLab/UBio-MolFM (MIT)
# https://github.com/IQuestLab/UBio-MolFM

# -*- coding: utf-8 -*-


import torch
import triton
import triton.language as tl

# -----------------------

# -----------------------
def make_cfgs_n_tiled():
    cfgs = []
    for BH in (32, 128):          # H tile
        for BK in (4, ):        # K tile
            for BD in (8, 16):        
                
                if BH*BK + BH*BD + BK*BD > 64_000:
                    continue
                for nw in (4, ):
                    for ns in (2, 3):
                        cfgs.append(
                            triton.Config({'BLOCK_H': BH, 'BLOCK_K': BK, 'BLOCK_D': BD},
                                          num_warps=nw, num_stages=ns)
                        )
    
    keyset, out = set(), []
    for c in cfgs:
        t = (c.kwargs['BLOCK_H'], c.kwargs['BLOCK_K'], c.kwargs['BLOCK_D'], c.num_warps, c.num_stages)
        if t not in keyset:
            keyset.add(t); out.append(c)
    return out

AUTOTUNE_CFGS = make_cfgs_n_tiled()

# -----------------------
# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _fwd_n_tiled(
    Q, K, IDX, GATE, OUT, DOT,
    scale,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_gn, s_gk, s_gh,
    s_on, s_ok, s_oh,
    s_dn, s_dk, s_dh,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    k0 = 0
    while k0 < Kdim:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim
        idx_ptr  = IDX + pid_n * s_in + k_offs * s_ik
        idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)

        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)         # [BH]
            h_mask = h_offs < H

            acc = tl.zeros((BLOCK_K, BLOCK_H), dtype=tl.float32)  # [BK,BH]

            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)     # [BD]
                d_mask = d_offs < D

                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

                k_base = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr  = K + k_base
                kmask  = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * q_tile[None, :, :], axis=2)   # [BK,BH]
                d0  += BLOCK_D

            
            dot_ptr = DOT + pid_n*s_dn + k_offs[:, None]*s_dk + h_offs[None, :]*s_dh
            tl.store(dot_ptr, acc, mask=k_mask[:, None] & h_mask[None, :])

            
            gate_ptr  = GATE + pid_n*s_gn + k_offs[:, None]*s_gk + h_offs[None, :]*s_gh
            gate_tile = tl.load(gate_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)
            out_tile  = (acc * gate_tile * scale).to(OUT_DTYPE)
            out_ptr   = OUT + pid_n*s_on + k_offs[:, None]*s_ok + h_offs[None, :]*s_oh
            tl.store(out_ptr, out_tile, mask=k_mask[:, None] & h_mask[None, :])

            h0 += BLOCK_H
        k0 += BLOCK_K

# -----------------------

# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'])
@triton.jit
def _bwdq_n_tiled(
    K, IDX, ALPHA, dQ,
    N, H, Kdim, D,
    s_kn, s_kh, s_kd,
    s_in, s_ik,
    s_an, s_ak, s_ah,
    s_qn, s_qh, s_qd,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N: 
        return

    h0 = 0
    while h0 < H:
        h_offs = h0 + tl.arange(0, BLOCK_H)             # [BH]
        h_mask = h_offs < H

        d0 = 0
        while d0 < D:
            d_offs = d0 + tl.arange(0, BLOCK_D)         # [BD]
            d_mask = d_offs < D
            acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)  # [BH,BD]

            k0 = 0
            while k0 < Kdim:
                k_offs = k0 + tl.arange(0, BLOCK_K)     # [BK]
                k_mask = k_offs < Kdim

                a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
                a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)

                idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
                idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)
                k_base   = idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                k_ptr    = K + k_base
                kmask    = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                k_tile   = tl.load(k_ptr, mask=kmask, other=0.).to(tl.float32)

                acc += tl.sum(k_tile * a_tile[:, :, None], axis=0)  # [BH,BD]
                k0  += BLOCK_K

            dq_ptr = dQ + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
            tl.store(dq_ptr, acc, mask=h_mask[:, None] & d_mask[None, :])

            d0 += BLOCK_D
        h0 += BLOCK_H

# -----------------------

# -----------------------
@triton.autotune(configs=AUTOTUNE_CFGS, key=['H','Kdim','D'],reset_to_zero=["dK"])
@triton.jit
def _bwdk_n_tiled(
    Q, IDX, ALPHA, dK,
    N, H, Kdim, D,
    s_qn, s_qh, s_qd,
    s_in, s_ik,
    s_an, s_ak, s_ah,
    s_kn, s_kh, s_kd,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= N:
        return

    k0 = 0
    while k0 < Kdim:
        k_offs = k0 + tl.arange(0, BLOCK_K)             # [BK]
        k_mask = k_offs < Kdim

        h0 = 0
        while h0 < H:
            h_offs = h0 + tl.arange(0, BLOCK_H)         # [BH]
            h_mask = h_offs < H

            a_ptr  = ALPHA + pid_n*s_an + k_offs[:, None]*s_ak + h_offs[None, :]*s_ah
            a_tile = tl.load(a_ptr, mask=k_mask[:, None] & h_mask[None, :], other=0.).to(tl.float32)  # [BK,BH]

            idx_ptr  = IDX + pid_n*s_in + k_offs*s_ik
            idx_vals = tl.load(idx_ptr, mask=k_mask, other=0).to(tl.int32)                             # [BK]

            d0 = 0
            while d0 < D:
                d_offs = d0 + tl.arange(0, BLOCK_D)     # [BD]
                d_mask = d_offs < D

                q_ptr  = Q + pid_n*s_qn + h_offs[:, None]*s_qh + d_offs[None, :]*s_qd
                q_tile = tl.load(q_ptr, mask=h_mask[:, None] & d_mask[None, :], other=0.).to(tl.float32)

                contrib = a_tile[:, :, None] * q_tile[None, :, :]

                
                dk_ptrs = dK + idx_vals[:, None, None]*s_kn + h_offs[None, :, None]*s_kh + d_offs[None, None]*s_kd
                kmask   = k_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]
                tl.atomic_add(dk_ptrs, contrib, mask=kmask)

                d0 += BLOCK_D
            h0 += BLOCK_H
        k0 += BLOCK_K

# -----------------------

# -----------------------
class SparseQK_N_Tiled_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, idx, gate, scale: float):
        assert query.is_cuda and key.is_cuda and gate.is_cuda and idx.is_cuda
        idx = idx.int()

        assert idx.dtype == torch.int32
        N, H, D = query.shape
        Kdim = idx.shape[1]
        q, k, g = query.contiguous(), key.contiguous(), gate.contiguous()

        out = torch.empty((N, Kdim, H), dtype=q.dtype, device=q.device)
        dot = torch.empty((N, Kdim, H), dtype=torch.float32, device=q.device)

        s_qn, s_qh, s_qd = q.stride()
        s_kn, s_kh, s_kd = k.stride()
        s_in, s_ik       = idx.stride()
        s_gn, s_gk, s_gh = g.stride()
        s_on, s_ok, s_oh = out.stride()
        s_dn, s_dk, s_dh = dot.stride()

        if out.dtype == torch.float16: out_dtype = tl.float16
        elif out.dtype == torch.bfloat16: out_dtype = tl.bfloat16
        elif out.dtype == torch.float32: out_dtype = tl.float32
        else: raise TypeError(f"unsupported dtype: {out.dtype}")

        grid = lambda meta: (N,)

        _fwd_n_tiled[grid](
            q, k, idx, g, out, dot,
            scale,
            N, H, Kdim, D,
            s_qn, s_qh, s_qd,
            s_kn, s_kh, s_kd,
            s_in, s_ik,
            s_gn, s_gk, s_gh,
            s_on, s_ok, s_oh,
            s_dn, s_dk, s_dh,
            OUT_DTYPE=out_dtype,
        )

        ctx.save_for_backward(query, key, idx, gate, dot)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, idx, g, dot = ctx.saved_tensors
        scale = ctx.scale
        N, H, D = q.shape
        Kdim = idx.shape[1]
        device = q.device

        if torch.is_grad_enabled():
            # Keep the VJP differentiable for force training. In particular
            # dots must retain the original query/key dependencies.
            neighbors = idx.long()
            keys = k[neighbors].float()
            queries = q.float().unsqueeze(1)
            upstream = grad_out.float()
            weight = (upstream * g.float() * scale).unsqueeze(-1)
            dQ = (weight * keys).sum(dim=1)
            dK = torch.zeros_like(k, dtype=torch.float32).index_add(
                0, neighbors.flatten(), (weight * queries).flatten(0, 1)
            )
            dgate = upstream * (queries * keys).sum(dim=-1) * scale
        else:
            alpha = (g * grad_out * scale).to(torch.float32)
            dQ = torch.empty_like(q, dtype=torch.float32)
            dK = torch.zeros_like(k, dtype=torch.float32)

            s_kn, s_kh, s_kd = k.stride()
            s_in, s_ik       = idx.stride()
            s_an, s_ak, s_ah = alpha.stride()
            s_qn, s_qh, s_qd = dQ.stride()
            s_qn0, s_qh0, s_qd0 = q.stride()
            s_kn0, s_kh0, s_kd0 = dK.stride()

            grid = lambda meta: (N,)
            _bwdq_n_tiled[grid](
                k, idx, alpha, dQ,
                N, H, Kdim, D,
                s_kn, s_kh, s_kd,
                s_in, s_ik,
                s_an, s_ak, s_ah,
                s_qn, s_qh, s_qd,
            )

            _bwdk_n_tiled[grid](
                q, idx, alpha, dK,
                N, H, Kdim, D,
                s_qn0, s_qh0, s_qd0,
                s_in, s_ik,
                s_an, s_ak, s_ah,
                s_kn0, s_kh0, s_kd0,
            )
            dgate = dot * grad_out.float() * scale

        return dQ.to(q.dtype), dK.to(k.dtype), None, dgate.to(g.dtype), None


def sparse_qk_triton_n_tiled(query, key, idx, gate, scale=0.1):
    return SparseQK_N_Tiled_Fn.apply(query, key, idx, gate, scale)
