"""Geodite interaction, edge update, and equivariant coupling.

The blocks follow IBM ``models/pos_egnn/mlip`` (Apache-2.0): SiLU attention
instead of softmax, bias-free edge maps, a single RMS statistic over all
spherical-harmonic degrees, and a gate that reads the sum of squares of the
steerable state instead of dividing by a per-degree norm. There is no
Clebsch–Gordan tensor product. Receivers are ``idx_i`` and senders are
``idx_j``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity, LayerNorm, Linear, Module, ModuleList, Sequential

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.head import Head
from ..mp.nn.field_norm import TensorLayerNorm
from ..mp.aggregate import Aggregate
from .geometry import angular_dim, geodite_spherical_harmonics
from ..mp.ops.rejection import vector_rejection
from .geometry import split_degrees


_EDGE_PARTS = {
    "gated",
    "gatedt",
    "norej",
    "norm",
    "mlp",
    "mlpa",
    "act",
    "linw",
    "linwa",
    "ln",
    "postln",
}


@dataclass(frozen=True)
class EdgeUpdateSpec:
    """Parsed ``edge_updates`` string. Geodite-MP uses ``linw_mlpa_ln``."""

    reject: bool = True
    widen: bool = False
    activate_last: bool = True
    project: int = 0
    project_norm: int = 0
    gate: str = ""


def parse_edge_updates(spec: Union[bool, str]) -> EdgeUpdateSpec:
    """Parse the reference edge-update flags.

    ``True`` is the GotenNet-style update (rejection, SiLU on a square map,
    no extra projection). A string such as ``linw_mlpa_ln`` is the
    Geodite-MP update. ``False`` is accepted and ignored by the caller when
    edge updates are switched off.
    """
    if spec is False or spec is True or spec == "":
        return EdgeUpdateSpec()
    if not isinstance(spec, str):
        raise TypeError(f"edge_updates must be a bool or string, got {type(spec)}")
    parts = [part for part in spec.split("_") if part]
    unknown = [part for part in parts if part not in _EDGE_PARTS]
    if unknown:
        allowed = ", ".join(sorted(_EDGE_PARTS))
        raise ValueError(f"Unknown edge update parts {unknown}. Allowed: {allowed}")
    project = 0
    if "linw" in parts:
        project = 1
    if "linwa" in parts:
        project = 2
    project_norm = 0
    if "ln" in parts:
        project_norm = 1
    if "postln" in parts:
        project_norm = 2
    gate = ""
    if "gated" in parts:
        gate = "gated"
    if "gatedt" in parts:
        gate = "gatedt"
    if "act" in parts:
        gate = "act"
    return EdgeUpdateSpec(
        reject="norej" not in parts,
        widen="mlp" in parts or "mlpa" in parts,
        activate_last="mlp" not in parts,
        project=project,
        project_norm=project_norm,
        gate=gate,
    )


class Dense(Module):
    """Bias-free linear map, Xavier-uniform, then optional layer norm and activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable] = None,
        norm: str = "",
    ) -> None:
        super().__init__()
        if norm not in ("", "layer"):
            raise ValueError(f"norm must be '' or 'layer', got {norm!r}")
        self.linear = Linear(in_features, out_features, bias=False)
        self.norm = LayerNorm(out_features) if norm == "layer" else None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.linear(inputs)
        if self.norm is not None:
            output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


def _mlp(
    widths: Sequence[int],
    activation: Callable,
    last_activation: Optional[Callable],
    hidden_norm: str = "",
) -> Sequential:
    layers: List[Module] = []
    for index in range(len(widths) - 2):
        layers.append(
            Dense(
                widths[index],
                widths[index + 1],
                activation=activation,
                norm=hidden_norm,
            )
        )
    layers.append(Dense(widths[-2], widths[-1], activation=last_activation))
    return Sequential(*layers)


class NodeInit(Module):
    """Environment-conditioned scalar state from the species embedding.

    The center embedding is an input. A second species table mixes into the
    neighbor message only; it is not the atom's identity embedding.
    """

    def __init__(self, channels: int, num_rbf: int, max_Za: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.neighbor_embedding = torch.nn.Embedding(int(max_Za) + 1, self.channels)
        self.radial_proj = Dense(int(num_rbf), self.channels)
        self.aggregate = Aggregate(mode="density")
        self.density = Dense(int(num_rbf), 1, activation=F.silu)
        self.update = _mlp(
            (2 * self.channels, self.channels, self.channels),
            activation=F.silu,
            last_activation=None,
        )

    def forward(
        self,
        atom_embedding: Tensor,
        atomic_numbers: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        number, channels = atom_embedding.shape
        if idx_j.numel() == 0:
            message = atom_embedding.new_zeros((number, channels))
        else:
            neighbor = self.neighbor_embedding(atomic_numbers.long())
            radial = self.radial_proj(rbf)
            density = torch.tanh(self.density(rbf).square())
            message = self.aggregate.reduce(neighbor[idx_j] * radial, idx_i, number, density)
        return self.update(torch.cat([atom_embedding, message], dim=-1))


class EdgeInit(Module):
    """``t_ij = (h_i + h_j) W φ_ij``. No bias, so a zero radial basis stays zero."""

    def __init__(self, channels: int, num_rbf: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.radial_proj = Dense(int(num_rbf), self.channels)

    def forward(
        self,
        scalars: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        if idx_j.numel() == 0:
            return scalars.new_zeros((0, self.channels))
        return (scalars[idx_i] + scalars[idx_j]) * self.radial_proj(rbf)


class EquivariantCoupling(Module):
    """Mix invariant and steerable channels without a tensor product.

    The gate reads ``(h, Σ_m (XW)²)``. Sum of squares is smooth where the
    steerable state is zero; the block does not divide by a per-degree norm.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mix = Dense(self.channels, self.channels)
        self.gate = _mlp(
            (2 * self.channels, self.channels, 2 * self.channels),
            activation=F.silu,
            last_activation=None,
        )

    def forward(self, scalars: Tensor, steerable: Tensor) -> tuple[Tensor, Tensor]:
        mixed = self.mix(steerable)
        magnitude = mixed.square().sum(dim=1)
        context = self.gate(torch.cat([scalars, magnitude], dim=-1))
        scalar_gate, vector_gate = context.split(self.channels, dim=-1)
        return scalars + scalar_gate, steerable + vector_gate.unsqueeze(1) * mixed


class InteractionBlock(Module):
    """One Geodite message-passing block.

    Scalar messages are a SiLU attention residual. Steerable messages gate
    the edge harmonics and the sender's steerable state. Both are normalized
    by ``1 + ρ_i`` before the residual add. The last block skips the edge update.
    """

    def __init__(
        self,
        channels: int,
        num_rbf: int,
        lmax: int,
        num_heads: int,
        dropout: float,
        epsilon: float,
        edge_spec: EdgeUpdateSpec,
        update_edges: bool,
        evec_dim: int,
        emlp_dim: int,
        edge_norm: str = "",
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels {channels} must be divisible by num_heads {num_heads}"
            )
        if update_edges and edge_spec.project == 0 and evec_dim != channels:
            raise ValueError(
                "evec_dim must equal the hidden width when edge_updates has no linw projection"
            )
        self.channels = int(channels)
        self.num_rbf = int(num_rbf)
        self.lmax = int(lmax)
        self.num_heads = int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.register_buffer(
            "_value_head", torch.arange(3 * self.channels) // (3 * self.head_dim), persistent=False
        )
        self.dropout = float(dropout)
        self.update_edges = bool(update_edges)
        self.edge_spec = edge_spec
        self.evec_dim = int(evec_dim)
        multiplier = 3 * self.channels

        self.norm = TensorLayerNorm(self.lmax, self.channels, eps=epsilon)
        self.scalar_filter = _mlp(
            (self.channels, self.channels, multiplier),
            activation=F.silu,
            last_activation=None,
        )
        self.value = _mlp(
            (self.channels, self.channels, multiplier),
            activation=F.silu,
            last_activation=None,
        )
        self.query = Dense(self.channels, self.channels)
        self.key = Dense(self.channels, self.channels)
        self.edge_attention = Dense(self.channels, self.channels, activation=F.silu)
        self.edge_filter = Dense(self.channels, multiplier)
        self.aggregate = Aggregate(mode="density")
        self.density = Dense(self.num_rbf, 1, activation=F.silu)

        if self.update_edges:
            hidden = int(emlp_dim)
            widths = (
                (self.channels, hidden, self.channels)
                if edge_spec.widen
                else (self.channels, self.channels)
            )
            self.edge_mlp = _mlp(
                widths,
                activation=F.silu,
                last_activation=F.silu if edge_spec.activate_last else None,
                hidden_norm=edge_norm,
            )
            self.value_query = Dense(self.channels, self.evec_dim)
            self.value_key = ModuleList(
                Dense(self.channels, self.evec_dim) for _ in range(self.lmax)
            )
            gate: List[Module] = []
            if edge_spec.project > 0:
                if edge_spec.project_norm == 1:
                    gate.append(LayerNorm(self.evec_dim))
                if edge_spec.project == 2:
                    gate.append(torch.nn.SiLU())
                projected = Dense(self.evec_dim, self.channels)
                if edge_spec.project_norm == 2:
                    gate.extend((projected, LayerNorm(self.channels)))
                else:
                    gate.append(projected)
            if edge_spec.gate == "gated":
                gate.append(torch.nn.Sigmoid())
            elif edge_spec.gate == "gatedt":
                gate.append(torch.nn.Tanh())
            elif edge_spec.gate == "act":
                gate.append(torch.nn.SiLU())
            self.edge_gate = Sequential(*gate) if gate else Identity()

    def forward(
        self,
        scalars: Tensor,
        steerable: Tensor,
        edges: Tensor,
        harmonics: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scalars, steerable = self.norm(scalars, steerable)
        number = scalars.shape[0]
        if idx_j.numel() == 0:
            updated_edges = edges
        else:
            scalars, steerable, updated_edges = self._message(
                scalars, steerable, edges, harmonics, rbf, idx_i, idx_j, number
            )
        return scalars, steerable, updated_edges

    def _message(
        self,
        scalars: Tensor,
        steerable: Tensor,
        edges: Tensor,
        harmonics: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        number: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self.query(scalars).reshape(number, self.num_heads, self.head_dim)
        key = self.key(scalars).reshape(number, self.num_heads, self.head_dim)
        filtered = self.scalar_filter(scalars)
        values = self.value(scalars)
        attended = self.edge_attention(edges).reshape(-1, self.num_heads, self.head_dim)
        logits = (query[idx_i] * key[idx_j] * attended).sum(dim=-1, keepdim=True)
        attention = F.silu(logits) / math.sqrt(self.channels)
        if torch.compiler.is_compiling() and self.training and 0.0 < self.dropout < 1.0:
            # The composite dropout checks numel() under the coordinate VJP.
            # The native operator accepts a symbolic edge count.
            attention = torch.ops.aten.native_dropout.default(attention, self.dropout, True)[0]
        else:
            attention = F.dropout(attention, p=self.dropout, training=self.training)
        neighbor_values = values[idx_j]
        if torch.compiler.is_compiling() and values.device.type == "cpu":
            # A scalar/vector split can cut through a head. CPU Inductor drops
            # the partial head when fusing the flattened broadcast into a scatter.
            channel_weights = attention.squeeze(-1).index_select(1, self._value_head)
            mixed = channel_weights * neighbor_values
        else:
            value_heads = neighbor_values.reshape(-1, self.num_heads, 3 * self.head_dim)
            mixed = (attention * value_heads).reshape(-1, 3 * self.channels)
        mixed = mixed + self.edge_filter(edges) * filtered[idx_j]
        scalar_message, direction_gate, feature_gate = mixed.split(self.channels, dim=-1)
        steerable_message = (
            direction_gate.unsqueeze(1) * harmonics.unsqueeze(-1)
            + feature_gate.unsqueeze(1) * steerable[idx_j]
        )
        density = torch.tanh(self.density(rbf / self.num_rbf).square())
        norm = self.aggregate.normalizer(idx_i, number, scalar_message, density)
        scalars = scalars + self.aggregate.reduce_with_normalizer(scalar_message, idx_i, number, norm)
        steerable = steerable + self.aggregate.reduce_with_normalizer(steerable_message, idx_i, number, norm)
        if self.update_edges:
            edges = edges + self._edge_update(steerable, harmonics, edges, idx_i, idx_j)
        return scalars, steerable, edges

    def _edge_update(
        self,
        steerable: Tensor,
        harmonics: Tensor,
        edges: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        query = self.value_query(steerable)
        key_parts = [
            linear(component)
            for linear, component in zip(
                self.value_key, split_degrees(steerable, self.lmax, dim=1)
            )
        ]
        key = torch.cat(key_parts, dim=1)
        query_i = split_degrees(query[idx_i], self.lmax, dim=1)
        key_j = split_degrees(key[idx_j], self.lmax, dim=1)
        direction = split_degrees(harmonics, self.lmax, dim=1)
        weight = query.new_zeros(edges.shape[0], self.evec_dim)
        for receiver, sender, axis in zip(query_i, key_j, direction):
            if self.edge_spec.reject:
                receiver = vector_rejection(receiver, axis)
                sender = vector_rejection(sender, -axis)
            weight = weight + (receiver * sender).sum(dim=1)
        return self.edge_mlp(edges) * self.edge_gate(weight)


class GeoditeEmbed(Embed):
    input_fields = ("atom_embedding", "Za")

    def __init__(self, width, num_rbf, max_Za, lmax):
        super().__init__()
        self.width = width
        self.num_rbf = num_rbf
        self.lmax = lmax
        self.angular_dim = angular_dim(lmax)
        self.node_init = NodeInit(width, num_rbf, max_Za)
        self.edge_init = EdgeInit(width, num_rbf)

    def forward(self, inputs, ctx):
        x = inputs["atom_embedding"]
        z = inputs["Za"].long()
        if x.shape[-1] != self.width:
            raise ValueError(f"atom_embedding width {x.shape[-1]} != dim_embedding {self.width}")
        if ctx.rbf.ndim != 2 or ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf must have shape [E, {self.num_rbf}], got {tuple(ctx.rbf.shape)}")
        if x.shape[0] != z.shape[0]:
            raise ValueError(f"atom_embedding has {x.shape[0]} atoms but Za has {z.shape[0]}")
        scalar = self.node_init(x, z, ctx.rbf, ctx.receiver, ctx.sender)
        edges = self.edge_init(scalar, ctx.rbf, ctx.receiver, ctx.sender)
        vector = scalar.new_zeros((ctx.num_nodes, self.angular_dim, self.width))
        harmonics = (geodite_spherical_harmonics(ctx.unit(), self.lmax) if ctx.num_edges else
                     ctx.vec.new_zeros((0, self.angular_dim)))
        ctx.cache["geodite_harmonics"] = harmonics

        # Evaluate the isolated-atom reference once per species using the same learned updates.
        unique, inverse = torch.unique(z, sorted=True, return_inverse=True)
        choice = torch.empty_like(unique)
        choice.scatter_(0, inverse, torch.arange(ctx.num_nodes, device=z.device))
        empty = z.new_empty(0)
        isolated = self.node_init(x.index_select(0, choice), unique, ctx.rbf[:0], empty, empty)
        isolated_vector = isolated.new_zeros(unique.shape[0], self.angular_dim, self.width)
        ctx.cache["isolated_species"] = inverse
        return BlockState(scalar, vector, edges, {"isolated": isolated, "isolated_vector": isolated_vector})


class GeoditeBlock(Block):
    def __init__(self, message, coupling):
        super().__init__()
        self.message = message
        self.coupling = coupling

    def forward(self, state, ctx):
        x, v, edges = self.message(state.scalar, state.equi, state.edge,
                                   ctx.cache["geodite_harmonics"], ctx.rbf, ctx.receiver, ctx.sender)
        x, v = self.coupling(x, v)
        x0, v0 = self.message.norm(state.aux["isolated"], state.aux["isolated_vector"])
        x0, v0 = self.coupling(x0, v0)
        return BlockState(x, v, edges, {"isolated": x0, "isolated_vector": v0})


class GeoditeHead(Head):
    needs_taps = True
    output_fields = ("atom_feature", "isolated_atom_feature", "atom_steerable_feature")

    def forward(self, state, ctx, taps, inputs):
        scalars = torch.stack([tap.scalar for tap in taps[1:]], dim=-1)
        isolated = torch.stack([tap.aux["isolated"] for tap in taps[1:]], dim=-1)
        return {"atom_feature": scalars, "isolated_atom_feature": isolated.index_select(0, ctx.cache["isolated_species"]),
                "atom_steerable_feature": state.equi}
