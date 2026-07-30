"""Spherical harmonic channel embeddings with Wigner rotate helpers.

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
"""

from __future__ import annotations

import torch


class SO3_Embedding(torch.nn.Module):
    """Container for multi-resolution spherical harmonic node / edge features.

    Args:
        length: Number of nodes (or edges after expand).
        lmax_list: Maximum degree ``l`` per resolution.
        num_channels: Number of spherical channels.
        device: Device for the embedding tensor.
        dtype: Dtype for the embedding tensor.
    """

    def __init__(
        self,
        length: int,
        lmax_list: list[int],
        num_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.num_resolutions = len(lmax_list)

        self.num_coefficients = 0
        for i in range(self.num_resolutions):
            self.num_coefficients = self.num_coefficients + int((lmax_list[i] + 1) ** 2)

        embedding = torch.zeros(
            length,
            self.num_coefficients,
            self.num_channels,
            device=self.device,
            dtype=self.dtype,
        )

        self.set_embedding(embedding)
        self.set_lmax_mmax(lmax_list, lmax_list.copy())

    def clone(self) -> SO3_Embedding:
        clone = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        clone.set_embedding(self.embedding.clone())
        return clone

    def set_embedding(self, embedding: torch.Tensor) -> None:
        self.length = len(embedding)
        self.embedding = embedding

    def set_lmax_mmax(self, lmax_list: list[int], mmax_list: list[int]) -> None:
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list

    def _expand_edge(self, edge_index: torch.Tensor) -> None:
        embedding = self.embedding[edge_index]
        self.set_embedding(embedding)

    def expand_edge(self, edge_index: torch.Tensor) -> SO3_Embedding:
        x_expand = SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.num_channels,
            self.device,
            self.dtype,
        )
        x_expand.set_embedding(self.embedding[edge_index])
        return x_expand

    def _reduce_edge(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        new_embedding = torch.zeros(
            num_nodes,
            self.num_coefficients,
            self.num_channels,
            device=self.embedding.device,
            dtype=self.embedding.dtype,
        )
        new_embedding.index_add_(0, edge_index, self.embedding)
        self.set_embedding(new_embedding)

    def _m_primary(self, mapping) -> None:
        self.embedding = torch.einsum("nac,ba->nbc", self.embedding, mapping.to_m)

    def _l_primary(self, mapping) -> None:
        self.embedding = torch.einsum("nac,ab->nbc", self.embedding, mapping.to_m)

    def _rotate(self, SO3_rotation, lmax_list: list[int], mmax_list: list[int]) -> None:
        embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)

        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            embedding_i = self.embedding[:, offset : offset + num_coefficients]
            embedding_rotate = torch.cat(
                [
                    embedding_rotate,
                    SO3_rotation[i].rotate(embedding_i, lmax_list[i], mmax_list[i]),
                ],
                dim=1,
            )
            offset = offset + num_coefficients

        self.embedding = embedding_rotate
        self.set_lmax_mmax(lmax_list.copy(), mmax_list.copy())

    def _rotate_inv(self, SO3_rotation, mappingReduced) -> None:
        embedding_rotate = torch.tensor([], device=self.device, dtype=self.dtype)

        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]
            embedding_i = self.embedding[:, offset : offset + num_coefficients]
            embedding_rotate = torch.cat(
                [
                    embedding_rotate,
                    SO3_rotation[i].rotate_inv(
                        embedding_i, self.lmax_list[i], self.mmax_list[i]
                    ),
                ],
                dim=1,
            )
            offset = offset + num_coefficients

        self.embedding = embedding_rotate

        for i in range(self.num_resolutions):
            self.mmax_list[i] = int(self.lmax_list[i])

        self.set_lmax_mmax(self.lmax_list, self.mmax_list)

    def _grid_act(self, SO3_grid, act, mappingReduced) -> None:
        offset = 0
        for i in range(self.num_resolutions):
            num_coefficients = mappingReduced.res_size[i]

            x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_to_grid_mat(self.device)
            from_grid_mat = SO3_grid[self.lmax_list[i]][
                self.mmax_list[i]
            ].get_from_grid_mat(self.device)
            x_grid = torch.einsum("bai,zic->zbac", to_grid_mat, x_res)
            x_grid = act(x_grid)
            x_res = torch.einsum("bai,zbac->zic", from_grid_mat, x_grid)

            self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients

    def to_grid(self, SO3_grid, lmax: int = -1) -> torch.Tensor:
        if lmax == -1:
            lmax = max(self.lmax_list)

        to_grid_mat_lmax = SO3_grid[lmax][lmax].get_to_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        x_grid = torch.tensor([], device=self.device)

        for i in range(self.num_resolutions):
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            x_res = self.embedding[:, offset : offset + num_coefficients].contiguous()
            to_grid_mat = to_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i]),
            ]
            x_grid = torch.cat(
                [x_grid, torch.einsum("bai,zic->zbac", to_grid_mat, x_res)],
                dim=3,
            )
            offset = offset + num_coefficients

        return x_grid

    def _from_grid(self, x_grid: torch.Tensor, SO3_grid, lmax: int = -1) -> None:
        if lmax == -1:
            lmax = max(self.lmax_list)

        from_grid_mat_lmax = SO3_grid[lmax][lmax].get_from_grid_mat(self.device)
        grid_mapping = SO3_grid[lmax][lmax].mapping

        offset = 0
        offset_channel = 0
        for i in range(self.num_resolutions):
            from_grid_mat = from_grid_mat_lmax[
                :,
                :,
                grid_mapping.coefficient_idx(self.lmax_list[i], self.lmax_list[i]),
            ]
            x_res = torch.einsum(
                "bai,zbac->zic",
                from_grid_mat,
                x_grid[
                    :,
                    :,
                    :,
                    offset_channel : offset_channel + self.num_channels,
                ],
            )
            num_coefficients = int((self.lmax_list[i] + 1) ** 2)
            self.embedding[:, offset : offset + num_coefficients] = x_res
            offset = offset + num_coefficients
            offset_channel = offset_channel + self.num_channels
