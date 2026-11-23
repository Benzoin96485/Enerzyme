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
from torch.nn import Identity, LayerNorm, Linear, Module, ModuleList, Parameter, Sequential

from ..functional import reduce_edge
from ..layers.rejection import vector_rejection
from .geometry import degree_widths, split_degrees


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


class TensorLayerNorm(Module):
    """One RMS over every degree, with a per-degree affine gain.

    The degree weight ``1 / (2l + 1)`` gives each ``l`` the same vote. The
    statistic is then averaged over channels, so a symmetric geometry where
    one degree is zero does not make its own norm singular.
    """

    def __init__(self, lmax: int, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.eps = float(eps)
        self.channels = int(num_channels)
        self.split_sizes = [1, *degree_widths(self.lmax)]
        degree_index: List[int] = []
        balance: List[float] = []
        for degree, size in enumerate(self.split_sizes):
            degree_index.extend([degree] * size)
            balance.extend([1.0 / size] * size)
        self.register_buffer(
            "degree_index", torch.tensor(degree_index, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "balance", torch.tensor(balance).view(1, -1, 1), persistent=False
        )
        self.affine = Parameter(torch.ones(self.lmax + 1, self.channels))

    def forward(self, scalars: Tensor, steerable: Tensor) -> tuple[Tensor, Tensor]:
        packed = torch.cat([scalars.unsqueeze(1), steerable], dim=1)
        # [N, 1, C] then a scalar per atom.
        squared = (packed.square() * self.balance).sum(dim=1, keepdim=True)
        inverse_rms = (squared.mean(dim=2, keepdim=True) + self.eps).pow(-0.5)
        gain = self.affine.index_select(0, self.degree_index).unsqueeze(0)
        normalized = packed * inverse_rms * gain
        return normalized[:, 0, :], normalized[:, 1:, :]


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
            pooled = reduce_edge(
                neighbor[idx_j] * radial, idx_i, (number, channels)
            )
            norm = reduce_edge(density, idx_i, (number, 1))
            message = pooled / (norm + 1)
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
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        value_heads = values[idx_j].reshape(
            -1, self.num_heads, (3 * self.channels) // self.num_heads
        )
        mixed = (attention * value_heads).reshape(-1, 3 * self.channels)
        mixed = mixed + self.edge_filter(edges) * filtered[idx_j]
        scalar_message, direction_gate, feature_gate = mixed.split(self.channels, dim=-1)
        steerable_message = (
            direction_gate.unsqueeze(1) * harmonics.unsqueeze(-1)
            + feature_gate.unsqueeze(1) * steerable[idx_j]
        )
        density = torch.tanh(self.density(rbf / self.num_rbf).square())
        scalar_sum = reduce_edge(scalar_message, idx_i, (number, self.channels))
        steerable_sum = reduce_edge(
            steerable_message, idx_i, (number, steerable.shape[1], self.channels)
        )
        density_sum = reduce_edge(density, idx_i, (number, 1))
        scale = density_sum + 1
        scalars = scalars + scalar_sum / scale
        steerable = steerable + steerable_sum / scale.unsqueeze(-1)
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
