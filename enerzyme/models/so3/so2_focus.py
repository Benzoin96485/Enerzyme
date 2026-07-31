"""Focus-major SO(2) linear (DPA4 / EMFA).

Distinct from :class:`~enerzyme.models.so3.so2_ops.SO2Linear` (EquiformerV3
m-primary layout without an explicit focus axis).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class FocusSO2Linear(nn.Module):
    """SO(2)-equivariant linear in m-major reduced layout with focus streams.

    Weight is block-diagonal over |m| groups:
    - m=0: unconstrained cross-l mixing
    - |m|>0: SO(2)-constrained 2x2 coupling of (-m, +m) pairs

    Input / output layout: ``(F, E, D_m_trunc, C)``.
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
