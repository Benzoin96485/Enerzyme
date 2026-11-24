"""One DPA3 repflow update on node, edge, and angle features.

The block is the released order-2 update (Zhang et al., arXiv:2506.01686,
Eqs. 7–19). Each map is one linear layer plus the scalar activation. The
three ``res_*`` rules are the DeePMD-kit residual combiners. ``optim_update``
in that code is the same linear applied to a concatenation, written as three
smaller products; this module uses the concatenation.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor, nn

from ..mp.ops import scatter_sum
from ..layers.symmetrize import dp_symmetrize


def combine_updates(
    updates: Sequence[Tensor],
    style: str,
    residuals: Optional[Sequence[Tensor]] = None,
) -> Tensor:
    """Mix the identity term ``updates[0]`` with the messages after it."""
    if not updates:
        raise ValueError("updates must be non-empty")
    if style == "res_avg":
        total = updates[0]
        for item in updates[1:]:
            total = total + item
        return total / (float(len(updates)) ** 0.5)
    if style == "res_incr":
        total = updates[0]
        if len(updates) == 1:
            return total
        scale = (float(len(updates) - 1)) ** -0.5
        for item in updates[1:]:
            total = total + scale * item
        return total
    if style == "res_residual":
        if residuals is None or len(residuals) != len(updates) - 1:
            raise ValueError(
                "res_residual needs one residual weight per message, "
                f"got {0 if residuals is None else len(residuals)} for {len(updates) - 1}"
            )
        total = updates[0]
        for weight, item in zip(residuals, updates[1:]):
            total = total + weight * item
        return total
    raise ValueError(
        f"update_style must be res_avg, res_incr, or res_residual, got {style!r}"
    )


def _residual(dim: int, scale: float, mode: str) -> nn.Parameter:
    weight = nn.Parameter(torch.empty(dim))
    if mode == "const":
        nn.init.constant_(weight, float(scale))
    elif mode == "norm":
        nn.init.normal_(weight, std=float(scale))
    else:
        raise ValueError(f"update_residual_init must be 'const' or 'norm', got {mode!r}")
    return weight



class RepFlowBlock(nn.Module):
    """Parallel or sequential node / edge / angle update."""

    def __init__(
        self,
        n_dim: int,
        e_dim: int,
        a_dim: int,
        axis_neuron: int,
        n_multi_edge_message: int,
        update_angle: bool,
        update_style: str,
        update_residual: float,
        update_residual_init: str,
        activation: Callable[[Tensor], Tensor],
        a_compress_rate: int,
        a_compress_e_rate: int,
        a_compress_use_split: bool,
        sequential_update: bool,
        smooth_edge_update: bool,
        neighbor_scale: float,
        angular_scale: float,
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.axis_neuron = axis_neuron
        self.n_multi_edge_message = n_multi_edge_message
        self.update_angle = update_angle
        self.update_style = update_style
        self.activation = activation
        self.a_compress_rate = a_compress_rate
        self.a_compress_use_split = a_compress_use_split
        self.sequential_update = sequential_update
        self.smooth_edge_update = smooth_edge_update
        if sequential_update and update_style != "res_residual":
            raise ValueError("sequential_update only supports update_style='res_residual'")
        if sequential_update and not update_angle:
            raise ValueError("sequential_update requires update_angle=True")
        if neighbor_scale <= 0.0 or angular_scale <= 0.0:
            raise ValueError("neighbor and angular scales must be positive")
        self.inv_sqrt_neighbors = float(neighbor_scale) ** -0.5
        self.neighbor_scale = float(neighbor_scale)
        self.angular_scale = float(angular_scale) ** 0.5

        if a_compress_rate == 0:
            self.n_compress = n_dim
            self.e_compress = e_dim
        else:
            self.n_compress = a_dim // a_compress_rate
            self.e_compress = a_dim // (2 * a_compress_rate) * a_compress_e_rate
        angle_in = a_dim + self.n_compress + 2 * self.e_compress

        self.node_self = nn.Linear(n_dim, n_dim)
        self.node_sym = nn.Linear(axis_neuron * (n_dim + e_dim), n_dim)
        self.node_edge = nn.Linear(2 * n_dim + e_dim, n_multi_edge_message * n_dim)
        self.edge_self = nn.Linear(2 * n_dim + e_dim, e_dim)

        node_residuals = [
            _residual(n_dim, update_residual, update_residual_init),
            _residual(n_dim, update_residual, update_residual_init),
        ]
        node_residuals.extend(
            _residual(n_dim, update_residual, update_residual_init)
            for _ in range(n_multi_edge_message)
        )
        self.node_residual = nn.ParameterList(node_residuals)
        self.edge_residual = nn.ParameterList(
            [_residual(e_dim, update_residual, update_residual_init)]
        )

        self.compress_node: Optional[nn.Linear]
        self.compress_edge: Optional[nn.Linear]
        self.angle_to_edge: Optional[nn.Linear]
        self.edge_from_angle: Optional[nn.Linear]
        self.angle_self: Optional[nn.Linear]
        if update_angle:
            if a_compress_rate != 0 and not a_compress_use_split:
                self.compress_node = nn.Linear(n_dim, self.n_compress, bias=False)
                self.compress_edge = nn.Linear(e_dim, self.e_compress, bias=False)
            else:
                self.compress_node = None
                self.compress_edge = None
            self.angle_to_edge = nn.Linear(angle_in, e_dim)
            self.edge_from_angle = nn.Linear(e_dim, e_dim)
            self.angle_self = nn.Linear(angle_in, a_dim)
            self.edge_residual.append(
                _residual(e_dim, update_residual, update_residual_init)
            )
            self.angle_residual = nn.ParameterList(
                [_residual(a_dim, update_residual, update_residual_init)]
            )
        else:
            self.compress_node = None
            self.compress_edge = None
            self.angle_to_edge = None
            self.edge_from_angle = None
            self.angle_self = None
            self.angle_residual = nn.ParameterList()

    def _act(self, values: Tensor) -> Tensor:
        return self.activation(values)

    def _node_messages(
        self,
        node: Tensor,
        edge: Tensor,
        h: Tensor,
        switch: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        num_nodes: int,
    ) -> List[Tensor]:
        node_i = node[idx_i]
        node_j = node[idx_j]
        self_message = self._act(self.node_self(node))
        gated = switch.unsqueeze(-1)
        sym_edge = dp_symmetrize(
            edge * gated, h, idx_i, num_nodes, self.axis_neuron, self.inv_sqrt_neighbors
        )
        sym_node = dp_symmetrize(
            node_j * gated, h, idx_i, num_nodes, self.axis_neuron, self.inv_sqrt_neighbors
        )
        sym_message = self._act(self.node_sym(torch.cat([sym_edge, sym_node], dim=-1)))

        pair = torch.cat([node_i, node_j, edge], dim=-1)
        gathered = scatter_sum(self._act(self.node_edge(pair)) * gated, idx_i, num_nodes)
        gathered = gathered / self.neighbor_scale
        heads = gathered.view(num_nodes, self.n_multi_edge_message, self.n_dim)
        messages = [self_message, sym_message]
        messages.extend(heads[:, head] for head in range(self.n_multi_edge_message))
        return messages

    def _edge_self(self, node: Tensor, edge: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        pair = torch.cat([node[idx_i], node[idx_j], edge], dim=-1)
        return self._act(self.edge_self(pair))

    def _compress(self, node: Tensor, edge: Tensor) -> tuple:
        if self.a_compress_rate == 0:
            return node, edge
        if self.a_compress_use_split:
            return node[..., : self.n_compress], edge[..., : self.e_compress]
        assert self.compress_node is not None and self.compress_edge is not None
        return self.compress_node(node), self.compress_edge(edge)

    def _angle_hidden(
        self,
        angle: Tensor,
        node: Tensor,
        edge: Tensor,
        edge_ij: Tensor,
        edge_ik: Tensor,
        center: Tensor,
        target: str,
    ) -> Tensor:
        node_c, edge_c = self._compress(node, edge)
        packed = torch.cat(
            [angle, node_c[center], edge_c[edge_ik], edge_c[edge_ij]], dim=-1
        )
        if target == "edge":
            assert self.angle_to_edge is not None
            return self._act(self.angle_to_edge(packed))
        assert self.angle_self is not None
        return self._act(self.angle_self(packed))

    def _reduce_angle(
        self,
        hidden: Tensor,
        edge: Tensor,
        edge_ij: Tensor,
        edge_ik: Tensor,
        angle_switch: Tensor,
        angular_edge: Tensor,
        num_edges: int,
    ) -> Tensor:
        assert self.edge_from_angle is not None
        weight = angle_switch[edge_ij] * angle_switch[edge_ik]
        reduced = scatter_sum(hidden * weight.unsqueeze(-1), edge_ij, num_edges)
        reduced = reduced / self.angular_scale
        if not self.smooth_edge_update:
            reduced = torch.where(angular_edge.unsqueeze(-1), reduced, edge)
        return self._act(self.edge_from_angle(reduced))

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        angle: Tensor,
        h: Tensor,
        switch: Tensor,
        angle_switch: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        edge_ij: Tensor,
        edge_ik: Tensor,
        center: Tensor,
        angular_edge: Tensor,
    ) -> tuple:
        num_nodes = int(node.shape[0])
        num_edges = int(edge.shape[0])
        if self.sequential_update:
            return self._forward_sequential(
                node, edge, angle, h, switch, angle_switch, idx_i, idx_j,
                edge_ij, edge_ik, center, angular_edge, num_nodes, num_edges,
            )

        node_messages = self._node_messages(node, edge, h, switch, idx_i, idx_j, num_nodes)
        node_out = combine_updates(
            [node, *node_messages], self.update_style, self.node_residual
        )
        edge_messages = [self._edge_self(node, edge, idx_i, idx_j)]
        if self.update_angle:
            hidden = self._angle_hidden(angle, node, edge, edge_ij, edge_ik, center, "edge")
            edge_messages.append(
                self._reduce_angle(
                    hidden, edge, edge_ij, edge_ik, angle_switch, angular_edge, num_edges
                )
            )
        edge_out = combine_updates(
            [edge, *edge_messages], self.update_style, self.edge_residual
        )
        if self.update_angle:
            angle_message = self._angle_hidden(
                angle, node, edge, edge_ij, edge_ik, center, "angle"
            )
            angle_out = combine_updates(
                [angle, angle_message], self.update_style, self.angle_residual
            )
        else:
            angle_out = angle
        return node_out, edge_out, angle_out

    def _forward_sequential(
        self,
        node: Tensor,
        edge: Tensor,
        angle: Tensor,
        h: Tensor,
        switch: Tensor,
        angle_switch: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        edge_ij: Tensor,
        edge_ik: Tensor,
        center: Tensor,
        angular_edge: Tensor,
        num_nodes: int,
        num_edges: int,
    ) -> tuple:
        edge = edge + self.edge_residual[0] * self._edge_self(node, edge, idx_i, idx_j)
        angle_message = self._angle_hidden(
            angle, node, edge, edge_ij, edge_ik, center, "angle"
        )
        angle = angle + self.angle_residual[0] * angle_message
        hidden = self._angle_hidden(angle, node, edge, edge_ij, edge_ik, center, "edge")
        edge = edge + self.edge_residual[1] * self._reduce_angle(
            hidden, edge, edge_ij, edge_ik, angle_switch, angular_edge, num_edges
        )
        node_messages = self._node_messages(node, edge, h, switch, idx_i, idx_j, num_nodes)
        node = combine_updates([node, *node_messages], self.update_style, self.node_residual)
        return node, edge, angle
