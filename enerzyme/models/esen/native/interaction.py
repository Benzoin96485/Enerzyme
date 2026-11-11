"""eSEN edgewise / nodewise blocks (Fu et al., 2025).

The edgewise path is two fairchem-style SO(2) convolutions
(:class:`~enerzyme.models.so3.SO2_Convolution`) with a SiLU gate between
them and a polynomial envelope on the message. The nodewise path is an
equivariant linear pair plus the same gate (:class:`SO3Linear`,
:class:`GateActivation`). Neither path samples an S² grid unless the
ablation flags ``mlp_type="grid"`` or ``act_type="s2"`` are set.

Adapted from fairchem v1 ``models/esen`` (MIT). Energy and force heads stay
outside this module.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from ...blocks.radial_mlp import RadialFunction
from ...cutoff import polynomial_envelope
from ...so3 import (
    CoefficientMapping,
    GateActivation,
    SO2_Convolution,
    SO3_Embedding,
    SO3Grid,
    SO3Linear,
    SeparableS2Activation,
)


def _as_embedding(
    values: Tensor,
    lmax_list: List[int],
    num_channels: int,
    mmax_list: Optional[List[int]] = None,
) -> SO3_Embedding:
    emb = SO3_Embedding(
        0,
        list(lmax_list),
        num_channels,
        device=values.device,
        dtype=values.dtype,
    )
    emb.set_embedding(values)
    if mmax_list is None:
        mmax_list = list(lmax_list)
    emb.set_lmax_mmax(list(lmax_list), list(mmax_list))
    return emb


class EdgeDegreeEmbedding(nn.Module):
    """Lift invariant edge features into an equivariant one-hop embedding.

    Same radial MLP and m=0 padding as EquiformerV2's edge-degree block.
    The polynomial envelope is the eSEN addition: messages and their
    derivatives vanish at the cutoff.
    """

    def __init__(
        self,
        sphere_channels: int,
        lmax_list: List[int],
        mmax_list: List[int],
        mapping: CoefficientMapping,
        edge_channels_list: List[int],
        rescale_factor: float,
        cutoff: float,
        use_envelope: bool,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax_list = list(lmax_list)
        self.mmax_list = list(mmax_list)
        self.mapping = mapping
        self.rescale_factor = float(rescale_factor)
        self.cutoff = float(cutoff)
        self.use_envelope = bool(use_envelope)
        self.m0 = int(mapping.m_size[0])
        self.m_all = int(mapping.l_harmonic.shape[0])
        channels = list(edge_channels_list) + [self.m0 * sphere_channels]
        self.rad_func = RadialFunction(channels)
        self.sph_dim = int((max(self.lmax_list) + 1) ** 2)

    def forward(
        self,
        x_edge: Tensor,
        edge_distance: Tensor,
        edge_index: Tensor,
        rotation,
        num_nodes: int,
    ) -> Tensor:
        if edge_index.shape[1] == 0:
            return x_edge.new_zeros((num_nodes, self.sph_dim, self.sphere_channels))

        lifted = self.rad_func(x_edge).reshape(-1, self.m0, self.sphere_channels)
        pad = lifted.new_zeros(
            (lifted.shape[0], self.m_all - self.m0, self.sphere_channels)
        )
        coeffs = torch.cat((lifted, pad), dim=1)
        emb = _as_embedding(
            coeffs, self.lmax_list, self.sphere_channels, self.mmax_list
        )
        emb._l_primary(self.mapping)
        emb._rotate_inv(rotation, self.mapping)
        if self.use_envelope:
            env = polynomial_envelope(edge_distance, self.cutoff, 5)
            emb.embedding = emb.embedding * env.view(-1, 1, 1)
        emb._reduce_edge(edge_index[1], num_nodes)
        return emb.embedding / self.rescale_factor


class Edgewise(nn.Module):
    """Concatenate source and target, then two SO(2) convolutions.

    Messages are aggregated onto ``edge_index[1]`` (the receiver).
    """

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mapping: CoefficientMapping,
        edge_channels_list: List[int],
        cutoff: float,
        act_type: str,
        use_envelope: bool,
        SO3_grid=None,
    ) -> None:
        super().__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.lmax_list = [lmax]
        self.mmax_list = [mmax]
        self.mapping = mapping
        self.cutoff = float(cutoff)
        self.use_envelope = bool(use_envelope)
        self.act_type = act_type
        self.sph_dim = (lmax + 1) ** 2
        if act_type == "gate":
            self.act = GateActivation(lmax, mmax, hidden_channels)
            extra_m0 = lmax * hidden_channels
            self.s2_act = None
        elif act_type == "s2":
            self.act = None
            self.s2_act = SeparableS2Activation(lmax, mmax)
            extra_m0 = hidden_channels
            if SO3_grid is None:
                raise ValueError("act_type='s2' requires an SO3 grid table")
        else:
            raise ValueError(f"Unknown act_type {act_type!r}; expected 'gate' or 's2'")
        # The grid table is owned by the Core. Do not re-register it here.
        object.__setattr__(self, "SO3_grid", SO3_grid)

        self.so2_conv_1 = SO2_Convolution(
            2 * sphere_channels,
            hidden_channels,
            self.lmax_list,
            self.mmax_list,
            mapping,
            internal_weights=False,
            edge_channels_list=list(edge_channels_list),
            extra_m0_output_channels=extra_m0,
        )
        self.so2_conv_2 = SO2_Convolution(
            hidden_channels,
            sphere_channels,
            self.lmax_list,
            self.mmax_list,
            mapping,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

    def forward(
        self,
        node: Tensor,
        x_edge: Tensor,
        edge_distance: Tensor,
        edge_index: Tensor,
        rotation,
    ) -> Tensor:
        num_nodes = node.shape[0]
        if edge_index.shape[1] == 0:
            return node.new_zeros((num_nodes, self.sph_dim, self.sphere_channels))

        src = node[edge_index[0]]
        dst = node[edge_index[1]]
        paired = _as_embedding(
            torch.cat((src, dst), dim=2),
            self.lmax_list,
            2 * self.sphere_channels,
        )
        paired._rotate(rotation, self.lmax_list, self.mmax_list)
        hidden, gating = self.so2_conv_1(paired, x_edge)
        if self.act_type == "gate":
            activated = self.act(gating, hidden.embedding)
        else:
            activated = self.s2_act(gating, hidden.embedding, self.SO3_grid)
        hidden2 = _as_embedding(
            activated, self.lmax_list, self.hidden_channels, self.mmax_list
        )
        out = self.so2_conv_2(hidden2, x_edge)
        if self.use_envelope:
            env = polynomial_envelope(edge_distance, self.cutoff, 5)
            out.embedding = out.embedding * env.view(-1, 1, 1)
        out._rotate_inv(rotation, self.mapping)
        out._reduce_edge(edge_index[1], num_nodes)
        return out.embedding


class SpectralAtomwise(nn.Module):
    """Two equivariant linears with a SiLU gate. No S² grid."""

    def __init__(self, sphere_channels: int, hidden_channels: int, lmax: int) -> None:
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(sphere_channels, lmax * hidden_channels, bias=True),
            nn.SiLU(),
        )
        self.linear_1 = SO3Linear(sphere_channels, hidden_channels, lmax=lmax)
        self.act = GateActivation(lmax, lmax, hidden_channels)
        self.linear_2 = SO3Linear(hidden_channels, sphere_channels, lmax=lmax)

    def forward(self, node: Tensor) -> Tensor:
        gating = self.scalar_mlp(node.narrow(1, 0, 1))
        hidden = self.linear_1(node)
        hidden = self.act(gating, hidden)
        return self.linear_2(hidden)


class GridAtomwise(nn.Module):
    """Pointwise MLP on an S² grid. Ablation only; the paper default is spectral."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        grid: SO3Grid,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "grid", grid)
        self.mlp = nn.Sequential(
            nn.Linear(sphere_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_channels, sphere_channels, bias=False),
        )

    def forward(self, node: Tensor) -> Tensor:
        return self.grid.from_grid(self.mlp(self.grid.to_grid(node)))


class eSENBlock(nn.Module):
    """Pre-norm residual edgewise convolution, then pre-norm residual node FFN."""

    def __init__(
        self,
        sphere_channels: int,
        hidden_channels: int,
        lmax: int,
        mmax: int,
        mapping: CoefficientMapping,
        edge_channels_list: List[int],
        cutoff: float,
        norm_type: str,
        act_type: str,
        mlp_type: str,
        use_envelope: bool,
        SO3_grid=None,
    ) -> None:
        super().__init__()
        from ...so3 import get_normalization_layer

        self.norm_1 = get_normalization_layer(
            norm_type, lmax=lmax, num_channels=sphere_channels
        )
        self.edge_wise = Edgewise(
            sphere_channels,
            hidden_channels,
            lmax,
            mmax,
            mapping,
            edge_channels_list,
            cutoff,
            act_type,
            use_envelope,
            SO3_grid=SO3_grid,
        )
        self.norm_2 = get_normalization_layer(
            norm_type, lmax=lmax, num_channels=sphere_channels
        )
        if mlp_type == "spectral":
            self.atom_wise = SpectralAtomwise(sphere_channels, hidden_channels, lmax)
        elif mlp_type == "grid":
            if SO3_grid is None:
                raise ValueError("mlp_type='grid' requires an SO3 grid table")
            self.atom_wise = GridAtomwise(
                sphere_channels, hidden_channels, SO3_grid[lmax][lmax]
            )
        else:
            raise ValueError(
                f"Unknown mlp_type {mlp_type!r}; expected 'spectral' or 'grid'"
            )

    def forward(
        self,
        node: Tensor,
        x_edge: Optional[Tensor],
        edge_distance: Optional[Tensor],
        edge_index: Tensor,
        rotation,
    ) -> Tensor:
        residual = node
        updated = self.edge_wise(
            self.norm_1(node),
            x_edge if x_edge is not None else node.new_zeros((0, 1)),
            edge_distance if edge_distance is not None else node.new_zeros((0,)),
            edge_index,
            rotation,
        )
        node = residual + updated
        residual = node
        node = residual + self.atom_wise(self.norm_2(node))
        return node
