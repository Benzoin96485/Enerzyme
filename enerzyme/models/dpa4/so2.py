"""SO(2)-equivariant convolution for DPA4.

Reimplemented in PyTorch from DPA4/SeZM concepts (arXiv:2606.02419).
Supports degree_channel radial mixing, multi-head attention, gated activation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

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

        num_l_m0 = lmax + 1
        self.weight_m0 = nn.Parameter(
            torch.empty(n_focus, num_l_m0 * in_channels, num_l_m0 * out_channels)
        )
        nn.init.xavier_uniform_(self.weight_m0)

        if bias:
            self.bias0 = nn.Parameter(torch.zeros(n_focus, out_channels))
        else:
            self.bias0 = None

        self.weight_m = nn.ParameterList()
        for m in range(1, mmax + 1):
            num_l = lmax - m + 1
            w = nn.Parameter(
                torch.empty(n_focus, num_l * in_channels, 2 * num_l * out_channels)
            )
            nn.init.xavier_uniform_(w)
            w.data *= 1.0 / math.sqrt(2.0)
            self.weight_m.append(w)

        self.reduced_dim = (lmax + 1) + sum(
            2 * (lmax - m + 1) for m in range(1, mmax + 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (F, E, D_m_trunc, Cin) -> (F, E, D_m_trunc, Cout)."""
        F, E = x.shape[0], x.shape[1]
        num_l_m0 = self.lmax + 1

        x_m0 = x[:, :, :num_l_m0, :]
        x_m0_flat = x_m0.reshape(F, E, num_l_m0 * self.in_channels)
        out_m0 = torch.bmm(x_m0_flat, self.weight_m0)
        out_m0 = out_m0.reshape(F, E, num_l_m0, self.out_channels)

        if self.use_bias and self.bias0 is not None:
            out_m0 = torch.cat(
                [
                    out_m0[:, :, :1, :] + self.bias0[:, None, None, :],
                    out_m0[:, :, 1:, :],
                ],
                dim=2,
            )

        blocks = [out_m0]
        offset = num_l_m0
        for m_idx, m in enumerate(range(1, self.mmax + 1)):
            num_l = self.lmax - m + 1
            x_neg = x[:, :, offset : offset + num_l, :]
            x_pos = x[:, :, offset + num_l : offset + 2 * num_l, :]
            offset += 2 * num_l

            x_neg_flat = x_neg.reshape(F, E, num_l * self.in_channels)
            x_pos_flat = x_pos.reshape(F, E, num_l * self.in_channels)

            w = self.weight_m[m_idx]
            w_u = w[:, :, : num_l * self.out_channels]
            w_v = w[:, :, num_l * self.out_channels :]

            # Complex multiply (u+iv): out_neg = a u - b v, out_pos = a v + b u
            out_neg = torch.bmm(x_neg_flat, w_u) - torch.bmm(x_pos_flat, w_v)
            out_pos = torch.bmm(x_neg_flat, w_v) + torch.bmm(x_pos_flat, w_u)

            blocks.append(out_neg.reshape(F, E, num_l, self.out_channels))
            blocks.append(out_pos.reshape(F, E, num_l, self.out_channels))

        return torch.cat(blocks, dim=2)


class DynamicRadialDegreeMixer(nn.Module):
    """Edge-conditioned degree mixer on m-major ``(E, D_m, C)`` features.

    Matches deepmd ``degree_channel`` with optional low-rank channel factorization:
    kernels are produced from the m=0 radial slice ``(E, (lmax+1), C)``.
    """

    def __init__(
        self,
        lmax: int,
        mmax: int,
        channels: int,
        mode: str = "degree_channel",
        rank: int = 1,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.channels = int(channels)
        self.mode = str(mode).lower()
        self.rank = int(rank)
        if self.mode not in {"degree", "degree_channel"}:
            raise ValueError("`mode` must be 'degree' or 'degree_channel'")
        if self.rank < 0:
            raise ValueError("`rank` must be non-negative")

        self.reduced_dim = (self.lmax + 1) + sum(
            2 * (self.lmax - m + 1) for m in range(1, self.mmax + 1)
        )
        self.degree_kernel_size = sum(
            (self.lmax - m + 1) ** 2 for m in range(self.mmax + 1)
        )
        self.input_dim = (self.lmax + 1) * self.channels
        if self.mode == "degree":
            proj_out = self.degree_kernel_size
        elif self.rank > 0:
            proj_out = self.degree_kernel_size * self.rank
        else:
            proj_out = self.degree_kernel_size * self.channels

        self.weight = nn.Parameter(torch.empty(self.input_dim, proj_out))
        nn.init.xavier_uniform_(self.weight)
        if self.mode == "degree_channel" and self.rank > 0:
            self.channel_basis = nn.Parameter(torch.empty(self.rank, self.channels))
            nn.init.xavier_uniform_(self.channel_basis)
        else:
            self.register_parameter("channel_basis", None)

        compact_idx, dense_idx = self._build_dense_scatter_indices()
        self.register_buffer(
            "kernel_compact_index", torch.as_tensor(compact_idx, dtype=torch.long)
        )
        self.register_buffer(
            "kernel_dense_index", torch.as_tensor(dense_idx, dtype=torch.long)
        )

    def _build_dense_scatter_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        compact_indices: List[int] = []
        dense_indices: List[int] = []
        compact_offset = 0
        reduced_dim = self.reduced_dim

        def append_block(start_in: int, start_out: int, num_l: int) -> None:
            nonlocal compact_offset
            for l_in in range(num_l):
                for l_out in range(num_l):
                    compact_indices.append(compact_offset + l_in * num_l + l_out)
                    dense_indices.append(
                        (start_out + l_out) * reduced_dim + start_in + l_in
                    )

        num_l0 = self.lmax + 1
        append_block(0, 0, num_l0)
        compact_offset += num_l0 * num_l0
        offset = num_l0
        for m in range(1, self.mmax + 1):
            num_l = self.lmax - m + 1
            neg_start = offset
            pos_start = offset + num_l
            append_block(neg_start, neg_start, num_l)
            append_block(pos_start, pos_start, num_l)
            compact_offset += num_l * num_l
            offset += 2 * num_l
        return (
            np.asarray(compact_indices, dtype=np.int64),
            np.asarray(dense_indices, dtype=np.int64),
        )

    def forward(self, x_local: Tensor, radial_feat: Tensor) -> Tensor:
        """``x_local`` / ``radial_feat``: ``(E, D_m, C)``."""
        if x_local.shape != radial_feat.shape:
            raise ValueError("`x_local` and `radial_feat` must have the same shape")
        if x_local.shape[1] != self.reduced_dim or x_local.shape[2] != self.channels:
            raise ValueError("Input shape is incompatible with this mixer")

        radial_m0 = radial_feat[:, : self.lmax + 1, :].reshape(
            radial_feat.shape[0], self.input_dim
        )
        kernel_flat = radial_m0 @ self.weight

        if self.mode == "degree":
            kernel = self._scatter_degree_kernel(kernel_flat)
            return torch.matmul(kernel, x_local)

        if self.rank > 0:
            compact = kernel_flat.reshape(
                x_local.shape[0], self.degree_kernel_size, self.rank
            )
            return self._mix_rank_compact(compact, x_local)

        compact = kernel_flat.reshape(
            x_local.shape[0], self.degree_kernel_size, self.channels
        )
        kernel = self._scatter_channel_kernel(compact)
        return (kernel * x_local[:, None, :, :]).sum(dim=2)

    def _scatter_degree_kernel(self, compact: Tensor) -> Tensor:
        n_edge = compact.shape[0]
        source = compact.index_select(1, self.kernel_compact_index)
        dense = torch.zeros(
            self.reduced_dim * self.reduced_dim,
            n_edge,
            dtype=compact.dtype,
            device=compact.device,
        )
        dense.index_add_(0, self.kernel_dense_index, source.transpose(0, 1))
        return dense.transpose(0, 1).reshape(
            n_edge, self.reduced_dim, self.reduced_dim
        )

    def _scatter_rank_kernel(self, compact: Tensor) -> Tensor:
        n_edge = compact.shape[0]
        source = compact.index_select(1, self.kernel_compact_index)
        dense = torch.zeros(
            self.reduced_dim * self.reduced_dim,
            n_edge,
            self.rank,
            dtype=compact.dtype,
            device=compact.device,
        )
        dense.index_add_(0, self.kernel_dense_index, source.permute(1, 0, 2))
        return dense.permute(1, 0, 2).reshape(
            n_edge, self.reduced_dim, self.reduced_dim, self.rank
        )

    def _scatter_channel_kernel(self, compact: Tensor) -> Tensor:
        n_edge = compact.shape[0]
        source = compact.index_select(1, self.kernel_compact_index)
        dense = torch.zeros(
            self.reduced_dim * self.reduced_dim,
            n_edge,
            self.channels,
            dtype=compact.dtype,
            device=compact.device,
        )
        dense.index_add_(0, self.kernel_dense_index, source.permute(1, 0, 2))
        return dense.permute(1, 0, 2).reshape(
            n_edge, self.reduced_dim, self.reduced_dim, self.channels
        )

    def _mix_rank_compact(self, compact: Tensor, x_local: Tensor) -> Tensor:
        kernel = self._scatter_rank_kernel(compact)
        kernel_or = kernel.permute(0, 1, 3, 2).reshape(
            x_local.shape[0], self.reduced_dim * self.rank, self.reduced_dim
        )
        mixed = torch.matmul(kernel_or, x_local)
        mixed = mixed.reshape(
            x_local.shape[0], self.reduced_dim, self.rank, self.channels
        )
        return (mixed * self.channel_basis.view(1, 1, self.rank, self.channels)).sum(
            dim=2
        )


class SO2Convolution(nn.Module):
    """SO(2) convolution: the main message-passing operator in DPA4.

    Flow: pre_mix C->H -> rotate to local -> radial_feat A1 mix -> SO2Linear stack
    -> rotate back -> envelope-gated attn scatter -> post_mix H->C
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
        # n_radial kept for API compatibility with interaction/core builders.
        del n_radial

        self.ebed_dim = get_so3_dim(lmax)
        degree_idx_m = build_m_major_l_index(lmax, mmax)
        coeff_idx_m = build_m_major_index(lmax, mmax)
        self.register_buffer("coeff_index_m", torch.from_numpy(coeff_idx_m).long())
        self.register_buffer("degree_index_m", torch.from_numpy(degree_idx_m).long())
        inv_rescale = build_rotate_inv_rescale(lmax, mmax, degree_idx_m)
        # Applied on full packed D after rotate-back (deepmd rotate_inv_rescale_full).
        full_rescale = torch.ones(self.ebed_dim)
        for i, packed_idx in enumerate(coeff_idx_m.tolist()):
            full_rescale[packed_idx] = float(inv_rescale[i])
        self.register_buffer("inv_rescale_full", full_rescale)
        self.D_m_trunc = int(coeff_idx_m.shape[0])

        H = n_focus * self.Cf
        self.hidden_channels = H

        self.pre_mix = nn.Linear(channels, H, bias=False)
        self.post_mix = nn.Linear(H, channels, bias=False)
        nn.init.zeros_(self.post_mix.weight)

        # Lift Core radial_feat (E, lmax+1, C) → hidden width H = F*Cf.
        self.radial_hidden_proj = nn.Linear(channels, H, bias=False)

        if radial_so2_mode != "none":
            self.radial_mixer = DynamicRadialDegreeMixer(
                lmax=lmax,
                mmax=mmax,
                channels=H,
                mode=radial_so2_mode,
                rank=radial_so2_rank,
            )
        else:
            self.radial_mixer = None

        self.so2_layers = nn.ModuleList()
        self.so2_acts = nn.ModuleList()
        for _ in range(mixing_layers):
            self.so2_layers.append(
                SO2Linear(
                    lmax=lmax,
                    mmax=mmax,
                    in_channels=self.Cf,
                    out_channels=self.Cf * (2 if glu_activation else 1),
                    n_focus=n_focus,
                )
            )
            self.so2_acts.append(
                GatedActivation(
                    lmax=lmax,
                    channels=self.Cf,
                    n_focus=n_focus,
                    mmax=mmax,
                    activation=activation,
                    layout="fndc",
                )
            )

        if n_atten_head > 0:
            # Radial l=0 bias into attention logits (E, F, H_attn).
            self.attn_radial_bias = nn.Linear(H, n_focus * n_atten_head, bias=False)
            self.z_bias = nn.Parameter(torch.zeros(n_focus, n_atten_head))
        else:
            self.attn_radial_bias = None

    def forward(self, x: Tensor, edge_cache: EdgeCache, radial_feat: Tensor) -> Tensor:
        """
        Args:
            x: (N, D, C) node features in packed layout
            edge_cache: EdgeCache
            radial_feat: (E, lmax+1, C) per-edge radial features from the Core

        Returns:
            (N, D, C) updated features
        """
        N, D, C = x.shape
        device = x.device
        src = edge_cache.src
        dst = edge_cache.dst
        E = src.shape[0]
        H = self.hidden_channels

        # Step 1: Pre-mix C → H=F*Cf
        x_h = self.pre_mix(x).reshape(N, D, self.n_focus, self.Cf)
        x_src = x_h[src]  # (E, D, F, Cf)

        # Step 2: Rotate source features into m-major local frame
        D_to_m = project_D_to_m(
            edge_cache.D_full, self.coeff_index_m, self.ebed_dim
        )
        x_local = torch.einsum("emd,edfc->emfc", D_to_m[:, :, :D], x_src)
        # (E, D_m, F, Cf) → (E, D_m, H) for A1 mixer
        x_local_flat = x_local.reshape(E, self.D_m_trunc, H)

        # Step 3: Expand Core radial_feat to hidden width and m-major layout
        if radial_feat.shape[0] != E or radial_feat.shape[1] != self.lmax + 1:
            raise ValueError(
                f"`radial_feat` must have shape (E, lmax+1, C)=({E}, {self.lmax + 1}, {C}), "
                f"got {tuple(radial_feat.shape)}"
            )
        rad = self.radial_hidden_proj(radial_feat)  # (E, lmax+1, H)
        rad_m = rad.index_select(1, self.degree_index_m)  # (E, D_m, H)

        # Step 4: A1 DynamicRadialDegreeMixer (or elementwise fallback)
        if self.radial_mixer is not None:
            x_local_flat = self.radial_mixer(x_local_flat, rad_m)
        else:
            x_local_flat = x_local_flat * rad_m

        # Step 5: Focus-major SO(2) stack
        x_local = x_local_flat.reshape(E, self.D_m_trunc, self.n_focus, self.Cf)
        x_local = x_local.permute(2, 0, 1, 3).contiguous()  # (F, E, D_m, Cf)

        for so2_lin, so2_act in zip(self.so2_layers, self.so2_acts):
            residual = x_local
            y = so2_lin(x_local)
            if y.shape[-1] == 2 * self.Cf:
                nc = self.Cf
                x_local = so2_act(y[..., :nc], gate=y[..., nc:])
            else:
                x_local = so2_act(y)
            x_local = residual + x_local

        # Step 6: Rotate back + inverse-rotation rescale on full packed basis
        x_local = x_local.permute(1, 2, 0, 3).contiguous()  # (E, D_m, F, Cf)
        x_local_flat = x_local.reshape(E, self.D_m_trunc, H)
        Dt_from_m = project_Dt_from_m(
            edge_cache.Dt_full, self.coeff_index_m, self.ebed_dim
        )
        x_global = torch.einsum("edm,emh->edh", Dt_from_m[:, :D, :], x_local_flat)
        x_global = x_global * self.inv_rescale_full[:D].view(1, D, 1)
        x_global = x_global.reshape(E, D, self.n_focus, self.Cf)

        # Step 7: Envelope + optional attention aggregation
        msg = x_global * edge_cache.edge_env.unsqueeze(-1).unsqueeze(-1)
        if self.attn_radial_bias is not None and self.n_atten_head > 0:
            logits = self.attn_radial_bias(rad_m[:, 0, :]).reshape(
                E, self.n_focus, self.n_atten_head
            )
            attn_w = segment_envelope_gated_softmax(
                logits,
                edge_cache.edge_env,
                dst,
                N,
                self.z_bias,
                self.eps,
            )
            attn_w = attn_w.mean(dim=-1)  # (E, F)
            msg = msg * attn_w.unsqueeze(1).unsqueeze(-1)

        msg_flat = msg.reshape(E, D, H)
        out = torch.zeros(N, D, H, device=device, dtype=msg_flat.dtype)
        out.index_add_(0, dst, msg_flat)
        return self.post_mix(out)
