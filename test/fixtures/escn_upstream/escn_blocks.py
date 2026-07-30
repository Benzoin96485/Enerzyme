"""Extracted eSCN blocks from fairchem_core-1.10.0 (MIT).

Stripped of fairchem registry / GraphModelMixin. EdgeBlock consumes a precomputed
distance-feature tensor (parity harness supplies Gaussian or identity features).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from so3 import SO3_Grid


class SO2Conv(torch.nn.Module):
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

    def forward(self, x_m, x_edge) -> torch.Tensor:
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
            self.so2_conv.append(
                SO2Conv(
                    m,
                    self.sphere_channels,
                    self.hidden_channels,
                    edge_channels,
                    self.lmax_list,
                    self.mmax_list,
                    self.act,
                )
            )

    def forward(self, x, x_edge, mappingReduced):
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


class EdgeBlock(torch.nn.Module):
    """Invariant edge features from distance features + atomic numbers."""

    def __init__(
        self,
        edge_channels,
        num_distance_features: int,
        max_num_elements,
        act,
    ) -> None:
        super().__init__()
        self.in_channels = num_distance_features
        self.act = act
        self.edge_channels = edge_channels
        self.max_num_elements = max_num_elements

        self.fc1_dist = nn.Linear(self.in_channels, self.edge_channels)
        self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.fc1_edge_attr = nn.Linear(self.edge_channels, self.edge_channels)

    def forward(self, distance_features, source_element, target_element):
        x_dist = self.fc1_dist(distance_features)
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        return self.act(self.fc1_edge_attr(x_edge))


class MessageBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        num_distance_features: int,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.act = act
        self.hidden_channels = hidden_channels
        self.sphere_channels = sphere_channels
        self.SO3_grid = SO3_grid
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.edge_channels = edge_channels

        self.edge_block = EdgeBlock(
            self.edge_channels,
            num_distance_features,
            max_num_elements,
            self.act,
        )
        self.so2_block_source = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )
        self.so2_block_target = SO2Block(
            self.sphere_channels,
            self.hidden_channels,
            self.edge_channels,
            self.lmax_list,
            self.mmax_list,
            self.act,
        )

    def forward(
        self,
        x,
        atomic_numbers,
        distance_features,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        x_edge = self.edge_block(
            distance_features,
            atomic_numbers[edge_index[0]],
            atomic_numbers[edge_index[1]],
        )

        x_source = x.clone()
        x_target = x.clone()
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        x_source._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)
        x_target._rotate(SO3_edge_rot, self.lmax_list, self.mmax_list)

        x_source = self.so2_block_source(x_source, x_edge, mappingReduced)
        x_target = self.so2_block_target(x_target, x_edge, mappingReduced)

        x_target.embedding = x_source.embedding + x_target.embedding
        x_target._grid_act(self.SO3_grid, self.act, mappingReduced)
        x_target._rotate_inv(SO3_edge_rot, mappingReduced)
        x_target._reduce_edge(edge_index[1], len(x.embedding))
        return x_target


class LayerBlock(torch.nn.Module):
    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        num_distance_features: int,
        max_num_elements: int,
        SO3_grid: SO3_Grid,
        act,
    ) -> None:
        super().__init__()
        self.act = act
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels = sphere_channels
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid

        self.message_block = MessageBlock(
            self.sphere_channels,
            hidden_channels,
            edge_channels,
            self.lmax_list,
            self.mmax_list,
            num_distance_features,
            max_num_elements,
            self.SO3_grid,
            self.act,
        )
        self.fc1_sphere = nn.Linear(
            2 * self.sphere_channels_all, self.sphere_channels_all, bias=False
        )
        self.fc2_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )
        self.fc3_sphere = nn.Linear(
            self.sphere_channels_all, self.sphere_channels_all, bias=False
        )

    def forward(
        self,
        x,
        atomic_numbers,
        distance_features,
        edge_index,
        SO3_edge_rot,
        mappingReduced,
    ):
        x_message = self.message_block(
            x,
            atomic_numbers,
            distance_features,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )

        max_lmax = max(self.lmax_list)
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=3)

        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)
        return x_message
