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
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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

    def _mix_three_axes(self, axes: Tensor) -> Tensor:
        # Expose the 3-input, 1-output affine maps to the compiler so it can
        # fuse activation and reduction over the hidden channels per edge.
        first, last = self.lin[0], self.lin[2]
        x, y, z = axes.unbind(-1)
        w_x, w_y, w_z = first.weight.unbind(-1)
        hidden = x.unsqueeze(-1) * w_x + y.unsqueeze(-1) * w_y + z.unsqueeze(-1) * w_z + first.bias
        return (F.silu(hidden) * last.weight[0]).sum(-1) + last.bias[0]

    def _can_fuse_axis_layers(self, axes: Tensor) -> bool:
        # The fused expression bypasses the Sequential and its children.
        # Keep their usual call boundaries when a hook or a custom layer is present.
        from torch.nn.modules import module

        if (module._global_forward_pre_hooks or module._global_forward_hooks
                or module._global_backward_pre_hooks or module._global_backward_hooks):
            return False
        if type(self.lin) is not nn.Sequential or len(self.lin) != 3:
            return False
        first, activation, last = self.lin
        if type(first) is not nn.Linear or type(activation) is not nn.SiLU or type(last) is not nn.Linear:
            return False
        if first.bias is None or last.bias is None or first.in_features != 3 or last.out_features != 1:
            return False
        if any(layer.weight.dtype != axes.dtype or layer.bias.dtype != axes.dtype for layer in (first, last)):
            return False
        return not any(
            layer._forward_pre_hooks or layer._forward_hooks or layer._backward_pre_hooks or layer._backward_hooks
            or "forward" in layer.__dict__
            for layer in (self.lin, first, activation, last)
        )

    def _endpoint(self, environment: Tensor, frame: Tensor, nodes: Tensor) -> Tensor:
        projected = scalarize(environment[nodes], frame)
        axes = projected.transpose(1, 2)
        if (torch.compiler.is_compiling() and axes.is_cuda and not self.training
                and axes.dtype in (torch.float32, torch.float64)
                and not torch.is_autocast_enabled("cuda")
                and not torch._C._are_functorch_transforms_active()
                and self._can_fuse_axis_layers(axes)):
            mixed = checkpoint(self._mix_three_axes, axes, use_reentrant=False, preserve_rng_state=False)
        else:
            mixed = self.lin(axes).squeeze(-1)
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
