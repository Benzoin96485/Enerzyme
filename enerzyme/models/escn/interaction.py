"""eSCN message-passing blocks (paper architecture without energy/force heads).

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
Edge scalar features consume shared Enerzyme RBF outputs instead of an internal
distance expansion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..so3 import SO2Block, SO3Grid


class EdgeBlock(torch.nn.Module):
    """Invariant edge features from RBF + atomic numbers."""

    def __init__(
        self,
        edge_channels: int,
        num_rbf: int,
        max_Za: int,
        act,
    ) -> None:
        super().__init__()
        self.act = act
        self.edge_channels = edge_channels
        self.max_Za = max_Za

        self.fc1_dist = nn.Linear(num_rbf, self.edge_channels)
        self.source_embedding = nn.Embedding(self.max_Za + 1, self.edge_channels)
        self.target_embedding = nn.Embedding(self.max_Za + 1, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.fc1_edge_attr = nn.Linear(self.edge_channels, self.edge_channels)

    def forward(
        self,
        rbf: torch.Tensor,
        source_element: torch.Tensor,
        target_element: torch.Tensor,
    ) -> torch.Tensor:
        x_dist = self.fc1_dist(rbf)
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        x_edge = self.act(source_embedding + target_embedding + x_dist)
        return self.act(self.fc1_edge_attr(x_edge))


class MessageBlock(torch.nn.Module):
    """Edge-aligned SO(2) message passing for one layer."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        num_rbf: int,
        max_Za: int,
        SO3_grid: SO3Grid,
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
            num_rbf,
            max_Za,
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
        Za: torch.Tensor,
        rbf: torch.Tensor,
        edge_index: torch.Tensor,
        SO3_edge_rot,
        mappingReduced,
    ):
        x_edge = self.edge_block(
            rbf,
            Za[edge_index[0]],
            Za[edge_index[1]],
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
    """One eSCN layer: message passing + pointwise S² nonlinearity."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        edge_channels: int,
        lmax_list: list[int],
        mmax_list: list[int],
        num_rbf: int,
        max_Za: int,
        SO3_grid: SO3Grid,
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
            num_rbf,
            max_Za,
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
        Za: torch.Tensor,
        rbf: torch.Tensor,
        edge_index: torch.Tensor,
        SO3_edge_rot,
        mappingReduced,
    ):
        x_message = self.message_block(
            x,
            Za,
            rbf,
            edge_index,
            SO3_edge_rot,
            mappingReduced,
        )

        max_lmax = max(self.lmax_list)
        x_grid_message = x_message.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = x.to_grid(self.SO3_grid, lmax=max_lmax)
        x_grid = torch.cat([x_grid, x_grid_message], dim=2)

        x_grid = self.act(self.fc1_sphere(x_grid))
        x_grid = self.act(self.fc2_sphere(x_grid))
        x_grid = self.fc3_sphere(x_grid)

        x_message._from_grid(x_grid, self.SO3_grid, lmax=max_lmax)
        return x_message
