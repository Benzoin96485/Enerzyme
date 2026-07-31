"""SO(2)-equivariant convolution for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
Supports degree_channel radial mixing, multi-head attention, gated activation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .activation import GatedActivation
from .attention import segment_envelope_gated_softmax
from .edge_cache import EdgeCache
from .indexing import (
    build_m_major_index,
    build_m_major_l_index,
    build_rotate_inv_rescale,
    get_so3_dim,
    project_D_to_m,
    project_Dt_from_m,
)


class SO2Linear(nn.Module):
    """SO(2)-equivariant linear in m-major reduced layout.

    Weight is block-diagonal over |m| groups:
    - m=0: unconstrained cross-l mixing
    - |m|>0: SO(2)-constrained 2x2 coupling of (-m, +m) pairs
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        in_channels: int,
        out_channels: int,
        n_focus: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_focus = n_focus
        self.use_bias = bias

        # m=0 block: (lmax+1) degrees
        num_l_m0 = lmax + 1
        self.weight_m0 = nn.Parameter(
            torch.empty(n_focus, num_l_m0 * in_channels, num_l_m0 * out_channels))
        nn.init.xavier_uniform_(self.weight_m0)

        if bias:
            self.bias0 = nn.Parameter(torch.zeros(n_focus, out_channels))
        else:
            self.bias0 = None

        # |m|>0 blocks
        self.weight_m = nn.ParameterList()
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            w = nn.Parameter(torch.empty(n_focus, num_l * in_channels, 2 * num_l * out_channels))
            nn.init.xavier_uniform_(w)
            w.data *= 1.0 / math.sqrt(2.0)
            self.weight_m.append(w)

        # Precompute block sizes for m-major layout
        self.reduced_dim = (lmax + 1) + sum(2 * (lmax - m + 1) for m in range(1, mmax + 1))

    def forward(self, x: Tensor) -> Tensor:
        """x: (F, E, D_m_trunc, Cin) -> (F, E, D_m_trunc, Cout)."""
        F, E = x.shape[0], x.shape[1]
        num_l_m0 = self.lmax + 1

        # m=0 block
        x_m0 = x[:, :, :num_l_m0, :]  # (F, E, L+1, Cin)
        x_m0_flat = x_m0.reshape(F, E, num_l_m0 * self.in_channels)
        out_m0 = torch.bmm(x_m0_flat, self.weight_m0)  # (F, E, L+1 * Cout)
        out_m0 = out_m0.reshape(F, E, num_l_m0, self.out_channels)

        if self.use_bias and self.bias0 is not None:
            out_m0[:, :, 0, :] = out_m0[:, :, 0, :] + self.bias0.unsqueeze(1)

        blocks = [out_m0]
        offset = num_l_m0
        for m_idx, m in enumerate(range(1, self.mmax + 1)):
            num_l = self.lmax - m + 1
            # Neg and pos blocks
            x_neg = x[:, :, offset:offset + num_l, :]
            x_pos = x[:, :, offset + num_l:offset + 2 * num_l, :]
            offset += 2 * num_l

            x_neg_flat = x_neg.reshape(F, E, num_l * self.in_channels)
            x_pos_flat = x_pos.reshape(F, E, num_l * self.in_channels)

            w = self.weight_m[m_idx]  # (F, num_l*Cin, 2*num_l*Cout)
            w_u = w[:, :, :num_l * self.out_channels]
            w_v = w[:, :, num_l * self.out_channels:]

            # SO(2) coupling: [W_u, W_v; -W_v, W_u]
            out_neg = torch.bmm(x_neg_flat, w_u) + torch.bmm(x_pos_flat, w_v)
            out_pos = torch.bmm(x_pos_flat, w_u) - torch.bmm(x_neg_flat, w_v)

            out_neg = out_neg.reshape(F, E, num_l, self.out_channels)
            out_pos = out_pos.reshape(F, E, num_l, self.out_channels)
            blocks.append(out_neg)
            blocks.append(out_pos)

        return torch.cat(blocks, dim=2)


class DynamicRadialDegreeMixer(nn.Module):
    """Edge-conditioned degree mixer in reduced layout.

    mode="degree_channel" with rank=1: factorized per-channel cross-degree kernel.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        channels: int,
        mode: str = "degree_channel",
        rank: int = 1,
        n_radial_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.channels = channels
        self.mode = mode
        self.rank = rank

        # For each |m| group, we need a (num_l_in, num_l_out) kernel
        # conditioned on edge features. With rank-1 factorization:
        # W[l_in, l_out, c] = sum_r u[l_in, r, c] * v[l_out, r, c] * s[r, c]
        # For simplicity in v1, use elementwise radial modulation (degree-diagonal)
        # which is the "none" mode baseline. degree_channel rank-1 adds cross-degree.

        if mode == "degree_channel" and rank > 0:
            # Per |m| group, we produce a (num_l, rank, channels) factor pair
            self.mlps = nn.ModuleList()
            for m in range(mmax + 1):
                num_l = lmax - m + 1
                out_dim = 2 * num_l * rank * channels  # u and v factors
                self.mlps.append(nn.Sequential(
                    nn.Linear(n_radial_hidden, out_dim, bias=False),
                ))
            self.n_l_per_m = [lmax - m + 1 for m in range(mmax + 1)]
        else:
            # Elementwise: just produce per-degree per-channel scales
            total_coeffs = sum(lmax - m + 1 for m in range(mmax + 1))
            self.radial_proj = nn.Linear(n_radial_hidden, total_coeffs * channels, bias=False)
            self.n_l_per_m = [lmax - m + 1 for m in range(mmax + 1)]

    def forward(self, x: Tensor, edge_cond: Tensor) -> Tensor:
        """x: (F, E, D_m, C), edge_cond: (E, hidden) -> (F, E, D_m, C)."""
        if self.mode == "degree_channel" and self.rank > 0:
            return self._forward_rank(x, edge_cond)
        return self._forward_elementwise(x, edge_cond)

    def _forward_elementwise(self, x: Tensor, edge_cond: Tensor) -> Tensor:
        scales = self.radial_proj(edge_cond)  # (E, total*C)
        # Reshape to (1, E, total, C) and multiply
        total = sum(self.n_l_per_m)
        # But D_m includes doubled entries for |m|>0
        # Actually, simpler: just produce scales for the full reduced_dim
        F_dim, E_dim = x.shape[0], x.shape[1]
        D_m = x.shape[2]
        C = x.shape[3]
        # Produce scale per (D_m, C)
        s = scales.reshape(1, E_dim, -1, C)
        # Pad or truncate to match D_m
        if s.shape[2] < D_m:
            # For |m|>0, double the scales (same scale for neg and pos)
            parts = []
            offset = 0
            for m in range(self.mmax + 1):
                nl = self.n_l_per_m[m]
                chunk = s[:, :, offset:offset + nl, :]
                if m == 0:
                    parts.append(chunk)
                else:
                    parts.append(chunk)  # neg
                    parts.append(chunk)  # pos (same)
                offset += nl
            s = torch.cat(parts, dim=2)
        return x * s

    def _forward_rank(self, x: Tensor, edge_cond: Tensor) -> Tensor:
        # Simplified: for each |m| group, produce rank-1 cross-degree kernel
        F_dim, E_dim = x.shape[0], x.shape[1]
        C = x.shape[3]
        parts = []
        offset = 0
        for m_idx, m in enumerate(range(self.mmax + 1)):
            nl = self.n_l_per_m[m_idx]
            raw = self.mlps[m_idx](edge_cond)  # (E, 2*nl*rank*C)
            raw = raw.reshape(E_dim, 2, nl, self.rank, C)
            u = raw[:, 0, :, :, :]  # (E, nl, rank, C)
            v = raw[:, 1, :, :, :]  # (E, nl, rank, C)
            # kernel: (E, nl_out, nl_in, C) = sum_r v[nl_out,r,c] * u[nl_in,r,c]
            kernel = torch.einsum('eorc,eirc->eoic', v, u)  # (E, nl, nl, C)

            if m == 0:
                x_block = x[:, :, offset:offset + nl, :]  # (F, E, nl, C)
                # y[f,e,o,c] = sum_i kernel[e,o,i,c] * x[f,e,i,c]
                y_block = torch.einsum('eoic,feic->feoc', kernel, x_block)
                parts.append(y_block)
                offset += nl
            else:
                x_neg = x[:, :, offset:offset + nl, :]
                x_pos = x[:, :, offset + nl:offset + 2 * nl, :]
                y_neg = torch.einsum('eoic,feic->feoc', kernel, x_neg)
                y_pos = torch.einsum('eoic,feic->feoc', kernel, x_pos)
                parts.append(y_neg)
                parts.append(y_pos)
                offset += 2 * nl

        return torch.cat(parts, dim=2)


class SO2Convolution(nn.Module):
    """SO(2) convolution: the main message-passing operator in DPA4.

    Flow: pre_mix C->H -> rotate to local -> radial modulate -> SO2Linear stack
    -> focus compete -> rotate back -> envelope-gated attn scatter -> post_mix H->C

    v1 simplification: message_node_so3=False (skip SO3GridNet message path).
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        channels: int,
        n_focus: int = 1,
        focus_dim: int = 0,
        mixing_layers: int = 3,
        n_atten_head: int = 1,
        radial_so2_mode: str = "degree_channel",
        radial_so2_rank: int = 1,
        n_radial: int = 16,
        glu_activation: bool = True,
        activation: str = "silu",
        eps: float = 1e-7,
        message_node_so3: bool = False,
    ) -> None:
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.channels = channels
        self.n_focus = n_focus
        self.focus_dim = focus_dim if focus_dim > 0 else channels
        self.Cf = self.focus_dim
        self.mixing_layers = mixing_layers
        self.n_atten_head = n_atten_head
        self.eps = eps
        self.message_node_so3 = message_node_so3

        self.ebed_dim = get_so3_dim(lmax)
        degree_idx_m = build_m_major_l_index(lmax, mmax)
        coeff_idx_m = build_m_major_index(lmax, mmax)
        self.register_buffer("coeff_index_m", torch.from_numpy(coeff_idx_m).long())
        self.register_buffer("degree_index_m", torch.from_numpy(degree_idx_m).long())
        inv_rescale = build_rotate_inv_rescale(lmax, mmax, degree_idx_m)
        self.register_buffer("inv_rescale", torch.from_numpy(inv_rescale).float())
        self.D_m_trunc = coeff_idx_m.shape[0]

        H = n_focus * self.Cf  # total hidden width

        # Pre-mix: C -> H (expand channels to focus streams)
        self.pre_mix = nn.Linear(channels, H, bias=False)
        # Post-mix: H -> C (contract back)
        self.post_mix = nn.Linear(H, channels, bias=False)
        nn.init.zeros_(self.post_mix.weight)

        # Radial conditioning MLP: n_radial -> hidden
        radial_hidden = 64
        self.radial_mlp = nn.Sequential(
            nn.Linear(n_radial, radial_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(radial_hidden, radial_hidden, bias=False),
        )

        # Edge type conditioning
        self.edge_type_proj = nn.Linear(channels, radial_hidden, bias=False)

        # Radial feature projection for each degree (radial_hidden -> (lmax+1) * Cf)
        self.radial_feat_proj = nn.Linear(radial_hidden, (lmax + 1) * self.Cf, bias=False)

        # Dynamic radial degree mixer
        if radial_so2_mode != "none":
            self.radial_mixer = DynamicRadialDegreeMixer(
                lmax=lmax, mmax=mmax, channels=self.Cf,
                mode=radial_so2_mode, rank=radial_so2_rank,
                n_radial_hidden=radial_hidden,
            )
        else:
            self.radial_mixer = None

        # SO2 mixing layers
        self.so2_layers = nn.ModuleList()
        self.so2_acts = nn.ModuleList()
        for i in range(mixing_layers):
            self.so2_layers.append(SO2Linear(
                lmax=lmax, mmax=mmax,
                in_channels=self.Cf, out_channels=self.Cf * (2 if glu_activation else 1),
                n_focus=n_focus,
            ))
            self.so2_acts.append(GatedActivation(
                lmax=lmax, channels=self.Cf, n_focus=n_focus,
                mmax=mmax, activation=activation, layout="fndc",
            ))

        # Attention head
        if n_atten_head > 0:
            self.attn_proj = nn.Linear(radial_hidden, n_focus * n_atten_head, bias=False)
            self.z_bias = nn.Parameter(torch.zeros(n_focus, n_atten_head))
        else:
            self.attn_proj = None

    def forward(self, x: Tensor, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            x: (N, D, C) node features in packed layout
            edge_cache: EdgeCache
            radial_feat: (E, lmax+1, C) per-edge radial features

        Returns:
            (N, D, C) updated features
        """
        N, D, C = x.shape
        device = x.device
        src = edge_cache.src  # (E,)
        dst = edge_cache.dst  # (E,)
        E = src.shape[0]

        # Step 1: Pre-mix to expand channels: (N, D, C) -> (N, D, H=F*Cf)
        x_h = self.pre_mix(x)  # (N, D, H)
        x_h = x_h.reshape(N, D, self.n_focus, self.Cf)  # (N, D, F, Cf)

        # Step 2: Gather source features for edges
        x_src = x_h[src]  # (E, D, F, Cf)

        # Step 3: Rotate to local frame using m-major projection
        # Project D_full to m-major: (E, D_m, D) then x_local = D_to_m @ x_src
        D_to_m = project_D_to_m(
            edge_cache.D_full, self.coeff_index_m, self.ebed_dim
        )  # (E, D_m, D)
        x_local = torch.einsum('emd,edfg->emfg', D_to_m[:, :, :D], x_src)
        # x_local: (E, D_m, F, Cf)

        # Reorder to focus-major: (F, E, D_m, Cf)
        x_local = x_local.permute(2, 0, 1, 3)

        # Step 4: Edge conditioning
        edge_cond = self.radial_mlp(edge_cache.edge_rbf)  # (E, hidden)

        # Step 5: Radial degree modulation
        if self.radial_mixer is not None:
            x_local = self.radial_mixer(x_local, edge_cond)

        # Step 6: SO2 mixing stack
        for so2_lin, so2_act in zip(self.so2_layers, self.so2_acts):
            y = so2_lin(x_local)  # (F, E, D_m, Cf*2 or Cf)
            if y.shape[-1] == 2 * self.Cf:
                nc = self.Cf
                x_local = so2_act(y[..., :nc], gate=y[..., nc:])
            else:
                x_local = so2_act(y)

        # Step 7: Inverse rescale for truncated mmax
        x_local = x_local * self.inv_rescale.reshape(1, 1, -1, 1)

        # Step 8: Rotate back to global frame
        # (F, E, D_m, Cf) -> (E, D_m, F, Cf) then project back
        x_local = x_local.permute(1, 0, 2, 3)  # (E, F, D_m, Cf)
        x_local = x_local.permute(0, 2, 1, 3)  # (E, D_m, F, Cf)

        Dt_from_m = project_Dt_from_m(
            edge_cache.Dt_full, self.coeff_index_m, self.ebed_dim
        )  # (E, D, D_m)
        x_global = torch.einsum('edm,emfg->edfg', Dt_from_m[:, :D, :], x_local)
        # x_global: (E, D, F, Cf)

        # Step 9: Apply envelope gating
        msg = x_global * edge_cache.edge_env.unsqueeze(-1).unsqueeze(-1)

        # Step 10: Attention scatter
        if self.attn_proj is not None and self.n_atten_head > 0:
            logits = self.attn_proj(edge_cond).reshape(E, self.n_focus, self.n_atten_head)
            attn_w = segment_envelope_gated_softmax(
                logits, edge_cache.edge_env, dst, N, self.z_bias, self.eps)
            # Average over heads
            attn_w = attn_w.mean(dim=-1)  # (E, F)
            msg = msg * attn_w.unsqueeze(1).unsqueeze(-1)  # (E, D, F, Cf)

        # Step 11: Scatter to nodes
        msg_flat = msg.reshape(E, D, self.n_focus * self.Cf)
        out = torch.zeros(N, D, self.n_focus * self.Cf, device=device, dtype=msg_flat.dtype)
        dst_exp = dst.unsqueeze(-1).unsqueeze(-1).expand_as(msg_flat)
        out.scatter_add_(0, dst_exp, msg_flat)

        # Step 12: Post-mix back to C channels
        out = self.post_mix(out)  # (N, D, C)

        return out
