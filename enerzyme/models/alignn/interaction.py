"""Edge-gated graph convolution used by ALIGNN.

The released module in ``usnistgov/alignn`` (and the atomwise force field)
is the reference, including the per-channel gate. The paper writes a scalar
gate on the previous edge feature and applies the two convolutions in the
opposite order. ``update_order`` selects that order; the convolution itself
stays the released one.

``src`` is the neighbor that sends the message. ``dst`` is the node that
receives it. On the atom graph those are ``idx_j`` and ``idx_i``. On the
line graph they index bonds, not atoms.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Sequential
from torch.utils.checkpoint import checkpoint

from ..mp.nn.norm import feature_norm as normalization
from ..mp.ops import scatter_sum


def _apply_norm(norm: Optional[Module], features: Tensor) -> Tensor:
    if norm is None or features.shape[0] == 0:
        return features
    return norm(features)


class FeatureMLP(Module):
    """Linear, normalization, then SiLU.

    This is the released ``MLPLayer``. The norm is skipped on an empty
    batch so a molecule with no bonds still builds.
    """

    def __init__(self, in_features: int, out_features: int, norm: Optional[str]) -> None:
        super().__init__()
        self.linear = Linear(int(in_features), int(out_features))
        self.norm = normalization(norm, int(out_features))

    def forward(self, features: Tensor) -> Tensor:
        return F.silu(_apply_norm(self.norm, self.linear(features)))


def embedding_mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    norm: Optional[str],
) -> Sequential:
    """Two ``MLPLayer`` blocks. Bond and angle inputs use this path."""
    return Sequential(
        FeatureMLP(in_features, hidden_features, norm),
        FeatureMLP(hidden_features, out_features, norm),
    )


class EdgeGatedGraphConv(Module):
    """One edge-gated update of nodes and edges.

    The edge message is ``W_src h_src + W_dst h_dst + W_e e``. Its sigmoid
    is normalized separately on each feature channel by the sum of incoming
    gates at the destination, plus ``gate_eps``. The node residual is
    ``SiLU(Norm(W_self h + aggregated))``. The edge residual is
    ``SiLU(Norm(message))``. Both residuals are added outside the norm.
    """

    def __init__(
        self,
        features: int,
        norm: Optional[str],
        residual: bool = True,
        gate_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        width = int(features)
        if width < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        if float(gate_eps) <= 0:
            raise ValueError(f"gate_eps must be positive, got {gate_eps}")
        self.residual = bool(residual)
        self.gate_eps = float(gate_eps)
        self.src_gate = Linear(width, width)
        self.dst_gate = Linear(width, width)
        self.edge_gate = Linear(width, width)
        self.src_update = Linear(width, width)
        self.dst_update = Linear(width, width)
        self.norm_nodes = normalization(norm, width)
        self.norm_edges = normalization(norm, width)

    def _gate_message(self, node, edge, src, dst):
        return self.src_gate(node)[src] + self.dst_gate(node)[dst] + self.edge_gate(edge)

    def _can_recompute_gates(self):
        from torch.nn.modules import module

        if (module._global_forward_pre_hooks or module._global_forward_hooks
                or module._global_backward_pre_hooks or module._global_backward_hooks):
            return False
        return all(
            type(layer) is Linear and "forward" not in layer.__dict__
            and not (layer._forward_pre_hooks or layer._forward_hooks or layer._backward_pre_hooks or layer._backward_hooks)
            for layer in (self.src_gate, self.dst_gate, self.edge_gate)
        )

    def forward(
        self,
        node: Tensor,
        edge: Tensor,
        src: Tensor,
        dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (torch.compiler.is_compiling() and node.is_cuda and not self.training
                and not torch._C._are_functorch_transforms_active()):
            from torch._dynamo import config
            from ..layers.compiled_grad import compiling_fullgraph

            if getattr(config, "skip_nnmodule_hook_guards", False) and not compiling_fullgraph():
                return self._with_live_gate_hooks(node, edge, src, dst)
        return self._convolve(node, edge, src, dst)

    @torch.compiler.disable(recursive=False)
    def _with_live_gate_hooks(self, node, edge, src, dst):
        from torch._dynamo import config

        # A standalone compiled convolution also needs to retrace when a gate
        # acquires a hook after warmup, before it chooses whether to checkpoint.
        with config.patch(skip_nnmodule_hook_guards=False):
            return self._convolve(node, edge, src, dst)

    def _convolve(self, node, edge, src, dst):
        updated = self.src_update(node)
        if edge.shape[0] == 0:
            updated = F.silu(_apply_norm(self.norm_nodes, updated))
            if self.residual:
                updated = node + updated
            return updated, edge
        if (torch.compiler.is_compiling() and node.is_cuda and not self.training
                and not torch._C._are_functorch_transforms_active() and self._can_recompute_gates()):
            # Rebuild the edge projections during the force backward instead of
            # retaining the hidden state of every line-graph edge.
            message = checkpoint(
                self._gate_message, node, edge, src, dst,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            message = self._gate_message(node, edge, src, dst)
        gate = torch.sigmoid(message)
        neighbor = self.dst_update(node)[src] * gate
        summed = scatter_sum(neighbor, dst, node.shape[0])
        gate_sum = scatter_sum(gate, dst, node.shape[0])
        updated = updated + summed / (gate_sum + self.gate_eps)
        updated = F.silu(_apply_norm(self.norm_nodes, updated))
        updated_edge = F.silu(_apply_norm(self.norm_edges, message))
        if self.residual:
            updated = node + updated
            updated_edge = edge + updated_edge
        return updated, updated_edge


class ALIGNNConv(Module):
    """One ALIGNN layer: an atom-graph update and a line-graph update.

    ``update_order='code'`` is the released module: the atom graph runs
    first, and its edge output is the node input of the line graph. The
    line-graph node output becomes the next bond state.

    ``update_order='paper'`` follows equations (7) and (8) of Choudhary and
    DeCost: the line graph updates bonds and angles first, and that bond
    state is the edge input of the atom graph. The atom graph's edge output
    is the bond state passed forward.
    """

    def __init__(
        self,
        features: int,
        norm: Optional[str],
        residual: bool = True,
        gate_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.node_update = EdgeGatedGraphConv(features, norm, residual, gate_eps)
        self.edge_update = EdgeGatedGraphConv(features, norm, residual, gate_eps)

    def forward(
        self,
        atom: Tensor,
        bond: Tensor,
        angle: Tensor,
        atom_src: Tensor,
        atom_dst: Tensor,
        line_src: Tensor,
        line_dst: Tensor,
        update_order: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if update_order == "code":
            atom, bond_message = self.node_update(atom, bond, atom_src, atom_dst)
            bond, angle = self.edge_update(bond_message, angle, line_src, line_dst)
            return atom, bond, angle
        if update_order == "paper":
            bond_message, angle = self.edge_update(bond, angle, line_src, line_dst)
            atom, bond = self.node_update(atom, bond_message, atom_src, atom_dst)
            return atom, bond, angle
        raise ValueError(f"update_order must be 'code' or 'paper', got {update_order!r}")
