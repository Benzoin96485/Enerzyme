"""SO(2) convolution blocks for edge-aligned spherical features.

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SO2Conv(torch.nn.Module):
    """SO(2) convolution for a single nonzero order ``m``."""

    def __init__(
        self,
        m: int,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        act,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.sphere_channels = sphere_channels
        self.num_resolutions: int = len(self.lmax_list)
        self.m = m
        self.act = act

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficients = 0
            if self.mmax_list[i] >= m:
                num_coefficients = self.lmax_list[i] - m + 1
            num_channels = num_channels + num_coefficients * self.sphere_channels

        assert num_channels > 0

        self.fc1_dist = nn.Linear(edge_channels, 2 * self.hidden_channels)
        self.fc1_r = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_r = nn.Linear(self.hidden_channels, num_channels, bias=False)
        self.fc1_i = nn.Linear(num_channels, self.hidden_channels, bias=False)
        self.fc2_i = nn.Linear(self.hidden_channels, num_channels, bias=False)

    def forward(self, x_m: torch.Tensor, x_edge: torch.Tensor) -> torch.Tensor:
        x_edge = self.act(self.fc1_dist(x_edge))
        x_edge = x_edge.view(-1, 2, self.hidden_channels)

        x_r = self.fc1_r(x_m)
        x_r = x_r * x_edge[:, 0:1, :]
        x_r = self.fc2_r(x_r)

        x_i = self.fc1_i(x_m)
        x_i = x_i * x_edge[:, 1:2, :]
        x_i = self.fc2_i(x_i)

        x_m_r = x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r[:, 1] + x_i[:, 0]

        return torch.stack((x_m_r, x_m_i), dim=1).contiguous()


class SO2Block(torch.nn.Module):
    """SO(2) block: convolutions for all orders ``m`` including ``m=0``."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        act,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions: int = len(lmax_list)
        self.act = act

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficients = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficients * self.sphere_channels

        self.fc1_dist0 = nn.Linear(edge_channels, self.hidden_channels)
        self.fc1_m0 = nn.Linear(num_channels_m0, self.hidden_channels, bias=False)
        self.fc2_m0 = nn.Linear(self.hidden_channels, num_channels_m0, bias=False)

        self.so2_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            so2_conv = SO2Conv(
                m,
                self.sphere_channels,
                self.hidden_channels,
                edge_channels,
                self.lmax_list,
                self.mmax_list,
                self.act,
            )
            self.so2_conv.append(so2_conv)

    def forward(self, x, x_edge: torch.Tensor, mappingReduced):
        num_edges = len(x_edge)

        x._m_primary(mappingReduced)

        x_edge_0 = self.act(self.fc1_dist0(x_edge))

        x_0 = x.embedding[:, 0 : mappingReduced.m_size[0]].contiguous()
        x_0 = x_0.view(num_edges, -1)

        x_0 = self.fc1_m0(x_0)
        x_0 = x_0 * x_edge_0
        x_0 = self.fc2_m0(x_0)
        x_0 = x_0.view(num_edges, -1, x.num_channels)

        x.embedding[:, 0 : mappingReduced.m_size[0]] = x_0

        offset = mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            x_m = x.embedding[
                :, offset : offset + 2 * mappingReduced.m_size[m]
            ].contiguous()
            x_m = x_m.view(num_edges, 2, -1)
            x_m = self.so2_conv[m - 1](x_m, x_edge)
            x_m = x_m.view(num_edges, -1, x.num_channels)
            x.embedding[:, offset : offset + 2 * mappingReduced.m_size[m]] = x_m
            offset = offset + 2 * mappingReduced.m_size[m]

        x._l_primary(mappingReduced)
        return x
