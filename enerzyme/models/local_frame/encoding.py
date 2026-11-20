"""Local substructure encoding (LSE) shared by LEFTNet and AlphaNet.

The edge weight is built once from the initial scalar state and reused in
every layer, as both released models do. It concatenates a scalarized
description of each endpoint, the windowed radial filter, and the raw
radial basis.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .geometry import reduce_nodes, scalarize


def init_linears(module: nn.Module) -> None:
    """Xavier-uniform linear weights and zero biases, as in both codebases."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class RadialFilter(nn.Module):
    """``Linear → SiLU → Linear`` applied to the radial basis, then the window."""

    def __init__(self, num_rbf: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        init_linears(self)

    def forward(self, rbf: Tensor, window: Tensor) -> Tensor:
        return self.net(rbf) * window.unsqueeze(-1)


class StructureVector(nn.Module):
    """Sum of neighbor directions weighted by the radial filter and a scalar MLP."""

    def __init__(self, hidden: int, normalize: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hidden, hidden)]
        if normalize:
            layers.append(nn.LayerNorm(hidden, elementwise_affine=False))
        layers.append(nn.SiLU())
        self.mix = nn.Sequential(*layers)
        init_linears(self)

    def forward(
        self,
        scalar: Tensor,
        edge_unit: Tensor,
        radial_hidden: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        hidden = self.mix(scalar)
        message = (
            edge_unit.unsqueeze(-1)
            * radial_hidden.unsqueeze(1)
            * hidden[idx_j].unsqueeze(1)
        )
        return reduce_nodes(message, idx_i, scalar.shape[0])


class LocalSubstructure(nn.Module):
    """Edge weights ``A_ij`` from overlapping one-hop neighborhoods.

    ``endpoint='source_target'`` is LEFTNet (sender scalar, then receiver).
    ``endpoint='target_source'`` is AlphaNet, which names those atoms in the
    opposite order. ``scale=True`` divides by ``sqrt(F)``, which only
    AlphaNet does.
    """

    def __init__(
        self,
        hidden: int,
        normalize: bool,
        scale: bool,
        endpoint: str,
    ) -> None:
        super().__init__()
        if endpoint not in {"source_target", "target_source"}:
            raise ValueError(
                f"endpoint must be 'source_target' or 'target_source', got {endpoint!r}"
            )
        self.structure = StructureVector(hidden, normalize=normalize)
        bottleneck = max(hidden // 4, 1)
        self.lin = nn.Sequential(
            nn.Linear(3, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )
        self.endpoint = endpoint
        self.scale = math.sqrt(hidden) if scale else 1.0
        self.hidden = hidden
        init_linears(self.lin)

    def _endpoint(self, environment: Tensor, frame: Tensor, nodes: Tensor) -> Tensor:
        projected = scalarize(environment[nodes], frame)
        mixed = self.lin(projected.transpose(1, 2)).squeeze(-1)
        return (mixed + projected[:, 0, :]) / self.scale

    def forward(
        self,
        scalar: Tensor,
        edge_unit: Tensor,
        edge_frame: Tensor,
        radial_hidden: Tensor,
        rbf: Tensor,
        window: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        environment = self.structure(
            scalar, edge_unit, radial_hidden, idx_i, idx_j
        )
        source = self._endpoint(environment, edge_frame, idx_j)
        target = self._endpoint(environment, edge_frame, idx_i)
        if self.endpoint == "source_target":
            pair = torch.cat((source, target), dim=-1)
        else:
            pair = torch.cat((target, source), dim=-1)
        pair = pair * window.unsqueeze(-1)
        return torch.cat((pair, radial_hidden, rbf), dim=-1)


def aggregate_neighbors(
    embedding: Tensor,
    radial_hidden: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
) -> Tensor:
    """Initial scalar state: the atom embedding plus a radial neighbor sum."""
    if idx_i.numel() == 0:
        return embedding
    message = radial_hidden * embedding[idx_j]
    return embedding + reduce_nodes(message, idx_i, embedding.shape[0])
