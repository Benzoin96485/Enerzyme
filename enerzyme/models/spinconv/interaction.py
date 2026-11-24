"""SpinConv blocks: species mixture, distance map, and the spin convolution.

The mixture is the released embedding block with one shared species vector.
Each block learns its own source and target maps of that vector. The
convolution is the grid ``fullconv`` path: a circular kernel as wide as the
longitude axis, SiLU, then average pooling over longitude.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import AvgPool1d, Conv1d, Embedding, GroupNorm, Linear, Module

from ..mp.nn.activation import get_activation_fn
from .geometry import sphere_projection, splat_sphere


def _swish() -> Module:
    """Paper Swish: ``x * sigmoid(x)``, with fixed scale and temperature 1."""
    return get_activation_fn(
        "swish",
        {"dim_feature": 1, "initial_alpha": 1.0, "initial_beta": 1.0, "learnable": False},
    )


class SpeciesMixture(Module):
    """Softmax weights over ``num_basis`` experts from the two endpoints.

    The source and target maps start near zero, so the mixture begins at
    the bias of the logit layer. That matches the released embedding tables,
    which are drawn from ``U(-1e-4, 1e-4)``.
    """

    def __init__(self, dim_embedding: int, embedding_size: int, num_basis: int) -> None:
        super().__init__()
        self.source = Linear(int(dim_embedding), int(embedding_size), bias=False)
        self.target = Linear(int(dim_embedding), int(embedding_size), bias=False)
        self.logits = Linear(2 * int(embedding_size), int(num_basis))
        torch.nn.init.uniform_(self.source.weight, -1e-4, 1e-4)
        torch.nn.init.uniform_(self.target.weight, -1e-4, 1e-4)

    def forward(
        self,
        atom_embedding: Tensor,
        idx_source: Tensor,
        idx_target: Tensor,
    ) -> Tensor:
        if idx_source.numel() == 0:
            return atom_embedding.new_zeros((0, self.logits.out_features))
        source = self.source(atom_embedding[idx_source])
        target = self.target(atom_embedding[idx_target])
        return torch.softmax(self.logits(torch.cat((source, target), dim=-1)), dim=-1)


class EmbeddingBlock(Module):
    """Mixture-of-experts map used after the spin convolution and the distance block.

    ``fc1`` and ``fc2`` expand the input to ``num_basis`` vectors of width
    ``mid_channels``. The species weights reduce that stack, and ``fc3``
    maps the result to ``out_channels``. ``project=False`` returns the mixed
    vector and is the cut where :class:`SimpleReadout` replaces the energy
    block's last linear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        dim_embedding: int,
        embedding_size: int,
        num_basis: int,
        project: bool = True,
    ) -> None:
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.num_basis = int(num_basis)
        self.project = bool(project)
        self.act = _swish()
        self.fc1 = Linear(int(in_channels), self.mid_channels)
        self.fc2 = Linear(self.mid_channels, self.num_basis * self.mid_channels)
        self.fc3 = Linear(self.mid_channels, int(out_channels)) if self.project else None
        self.mixture = SpeciesMixture(dim_embedding, embedding_size, num_basis)

    def forward(
        self,
        features: Tensor,
        atom_embedding: Tensor,
        idx_source: Tensor,
        idx_target: Tensor,
    ) -> Tensor:
        weights = self.mixture(atom_embedding, idx_source, idx_target)
        hidden = self.act(self.fc2(self.act(self.fc1(features))))
        hidden = hidden.view(-1, self.num_basis, self.mid_channels)
        mixed = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        if self.fc3 is None:
            return mixed
        return self.fc3(mixed)


def _reject_atomic_numbers(za: Tensor, span: int) -> None:
    """Require every atomic number to index the species-pair table."""
    if za.numel() == 0:
        return
    low = int(za.min())
    high = int(za.max())
    if low < 0 or high >= span:
        raise ValueError(
            f"atomic numbers must lie in 0..max_Za ({span - 1}), "
            f"got min {low} and max {high}"
        )


class PairDistanceScale(Module):
    """Per-species-pair gain and offset applied before the Gaussian expansion.

    The gain lies in ``[1 / scale_max, scale_max]``. Both tables start near
    zero, so the gain starts at 1 and the offset starts at 0.
    """

    def __init__(self, max_Za: int, scale_max: float) -> None:
        super().__init__()
        if float(scale_max) <= 0:
            raise ValueError(f"distance_scale_max must be positive, got {scale_max}")
        self.span = int(max_Za) + 1
        self.scale_max = float(scale_max)
        self.log_scale = math.log(self.scale_max)
        self.gain = Embedding(self.span * self.span, 1)
        self.offset = Embedding(self.span * self.span, 1)
        torch.nn.init.uniform_(self.gain.weight, -1e-4, 1e-4)
        torch.nn.init.uniform_(self.offset.weight, -1e-4, 1e-4)

    def forward(self, distance: Tensor, za_source: Tensor, za_target: Tensor) -> Tensor:
        _reject_atomic_numbers(za_source, self.span)
        _reject_atomic_numbers(za_target, self.span)
        index = za_source.long() * self.span + za_target.long()
        raw = 2.0 * torch.sigmoid(self.gain(index).view(-1)) - 1.0
        gain = torch.exp(self.log_scale * raw)
        offset = self.offset(index).view(-1)
        return gain * distance + offset


class SpinConvolution(Module):
    """Project incoming source messages, convolve around the bond, and pool.

    The kernel width equals the longitude count, with circular padding of
    half that width. SiLU is applied before the pool, so the kernel shape
    is used: a longitude roll leaves the pool unchanged, and a reflection
    does not. GroupNorm uses ``hidden_channels / group_channels`` groups,
    which is 4 for the published width 32 and ``group_channels`` 8.
    """

    def __init__(
        self,
        hidden_channels: int,
        mid_channels: int,
        n_lat: int,
        n_long: int,
        group_channels: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.mid_channels = int(mid_channels)
        self.n_lat = int(n_lat)
        self.n_long = int(n_long)
        self.cutoff = float(cutoff)
        groups = self.hidden_channels // int(group_channels)
        self.groups = groups
        self.act = _swish()
        self.conv = Conv1d(
            self.hidden_channels * self.n_lat,
            self.mid_channels,
            kernel_size=self.n_long,
            groups=groups,
            padding=self.n_long // 2,
            padding_mode="circular",
        )
        self.pool = AvgPool1d(self.n_long)
        self.norm = GroupNorm(groups, self.mid_channels)

    def pool_grid(self, grid: Tensor) -> Tensor:
        """Circular convolution, SiLU, longitude pool, and GroupNorm.

        ``grid`` has shape ``(E, hidden, lat, long)``. An empty edge list
        skips GroupNorm and returns ``(0, mid)``.
        """
        if grid.shape[0] == 0:
            return grid.new_zeros((0, self.mid_channels))
        hidden = grid.reshape(grid.shape[0], self.hidden_channels * self.n_lat, self.n_long)
        hidden = self.act(self.conv(hidden))
        hidden = self.pool(hidden[:, :, : self.n_long])
        hidden = hidden.reshape(grid.shape[0], self.mid_channels)
        return self.norm(hidden)

    def forward(
        self,
        messages: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        vij: Tensor,
        distance: Tensor,
        num_nodes: int,
        projection=None,
    ) -> Tensor:
        if messages.shape[0] == 0:
            return messages.new_zeros((0, self.mid_channels))
        if projection is None:
            projection = sphere_projection(vij, idx_i, idx_j, distance, self.cutoff, num_nodes, self.n_lat, self.n_long)
        grid = splat_sphere(messages, messages.shape[0], self.n_lat, self.n_long, *projection)
        return self.pool_grid(grid)


class MessageBlock(Module):
    """One spin update of the edge state.

    The convolution is mixed with the endpoint species, added to a two-layer
    map of the distance embedding, and mixed again down to ``hidden_channels``.
    """

    def __init__(
        self,
        hidden_channels: int,
        mid_channels: int,
        dim_embedding: int,
        embedding_size: int,
        num_basis: int,
        n_lat: int,
        n_long: int,
        group_channels: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.act = _swish()
        self.spin = SpinConvolution(
            hidden_channels, mid_channels, n_lat, n_long, group_channels, cutoff
        )
        self.mix_spin = EmbeddingBlock(
            mid_channels,
            mid_channels,
            mid_channels,
            dim_embedding,
            embedding_size,
            num_basis,
        )
        self.dist1 = Linear(mid_channels, mid_channels)
        self.dist2 = Linear(mid_channels, mid_channels)
        self.mix_out = EmbeddingBlock(
            mid_channels,
            hidden_channels,
            mid_channels,
            dim_embedding,
            embedding_size,
            num_basis,
        )

    def forward(
        self,
        messages: Tensor,
        distance_embed: Tensor,
        atom_embedding: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        vij: Tensor,
        distance: Tensor,
        num_nodes: int,
        projection=None,
    ) -> Tensor:
        hidden = self.spin(messages, idx_i, idx_j, vij, distance, num_nodes, projection)
        hidden = self.mix_spin(hidden, atom_embedding, idx_j, idx_i)
        radial = self.dist2(self.act(self.dist1(distance_embed)))
        hidden = self.act(hidden + radial)
        return self.mix_out(hidden, atom_embedding, idx_j, idx_i)
