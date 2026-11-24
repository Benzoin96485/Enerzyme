"""Radial channel mixing and the two CACE messages.

The raw radial basis is summed onto atoms first and then mixed by a linear
map that depends on the angular shell and the edge channel (Cheng, 2024,
Eq. 5). Message passing, when enabled, is one of

* ``b``: a linear map of the sender's invariants multiplies the raw edge basis
* ``a``: an exponential radial filter multiplies the sender's mixed density
* ``memory``: a per-shell scale of the receiver's current density

``full`` uses all three. The aggregated messages are divided by
``sqrt(avg_num_neighbors)`` and, for ``b``, mixed with the same radial map.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Union

import torch
from torch import Tensor, nn
from ..mp.ops import scatter_sum


MessageSpec = Union[str, Iterable[str]]

_MESSAGE_ALIASES = {
    "b": "b",
    "bchi": "b",
    "a": "a",
    "ar": "a",
    "m": "memory",
    "memory": "memory",
}


def parse_messages(spec: MessageSpec) -> FrozenSet[str]:
    """Normalize a message specification to ``{"memory", "a", "b"}``."""
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key == "full":
            return frozenset({"memory", "a", "b"})
        if key in {"ab", "ba"}:
            return frozenset({"a", "b"})
        if key not in _MESSAGE_ALIASES:
            known = "b, a, memory, ab, full, or a list of those names"
            raise ValueError(f"Unknown CACE messages {spec!r}; expected {known}")
        return frozenset({_MESSAGE_ALIASES[key]})
    kinds = set()
    for item in spec:
        token = str(item).strip().lower()
        if token not in _MESSAGE_ALIASES:
            raise ValueError(
                f"Unknown CACE message {item!r}; expected Bchi, Ar, or M"
            )
        kinds.add(_MESSAGE_ALIASES[token])
    if not kinds:
        raise ValueError("CACE message passing needs at least one message")
    return frozenset(kinds)



class RadialChannelMix(nn.Module):
    """Per-shell map from raw radial channels to the mixed radial basis."""

    def __init__(
        self,
        n_raw: int,
        n_radial: int,
        n_channels: int,
        shells: Tensor,
    ) -> None:
        super().__init__()
        self.n_raw = int(n_raw)
        self.n_radial = int(n_radial)
        self.n_channels = int(n_channels)
        self.register_buffer("shells", shells.detach().to(dtype=torch.long).clone())
        weight = torch.empty(self.shells.shape[0], self.n_raw, self.n_radial * self.n_channels)
        for shell in range(weight.shape[0]):
            nn.init.xavier_uniform_(weight[shell])
        self.weight = nn.Parameter(
            weight.view(self.shells.shape[0], self.n_raw, self.n_radial, self.n_channels)
        )

    def forward(self, raw: Tensor) -> Tensor:
        if raw.shape[1] != self.n_raw or raw.shape[-1] != self.n_channels:
            raise ValueError(
                f"raw density must be [N, {self.n_raw}, angular, {self.n_channels}], "
                f"got {tuple(raw.shape)}"
            )
        mixed = raw.new_zeros(raw.shape[0], self.n_radial, raw.shape[2], self.n_channels)
        for shell, (start, end) in enumerate(self.shells.tolist()):
            group = raw[:, :, start:end, :]
            mixed[:, :, start:end, :] = torch.einsum(
                "irkc,roc->iokc", group, self.weight[shell]
            )
        return mixed


class _BchiGate(nn.Module):
    """Linear (or SiLU MLP) map from flattened invariants to angular weights."""

    def __init__(
        self,
        n_in: int,
        n_channels: int,
        lmax: int,
        l_index: Tensor,
        shared_channels: bool,
        shared_l: bool,
        n_layers: int,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"bchi layers must be >= 1, got {n_layers}")
        self.shared_channels = bool(shared_channels)
        self.shared_l = bool(shared_l)
        self.n_channels = int(n_channels)
        self.n_l = 1 if self.shared_l else int(lmax) + 1
        self.n_c = 1 if self.shared_channels else self.n_channels
        self.register_buffer("l_index", l_index.detach().to(dtype=torch.long).clone())
        n_out = self.n_l * self.n_c
        if n_layers == 1:
            self.h = nn.Linear(n_in, n_out)
        else:
            hidden = n_in
            layers = []
            width = n_in
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(width, hidden), nn.SiLU()])
                width = hidden
            layers.append(nn.Linear(width, n_out))
            self.h = nn.Sequential(*layers)

    def angular_weight(self, invariant: Tensor) -> Tensor:
        flat = invariant.reshape(invariant.shape[0], -1)
        raw = self.h(flat)
        if self.shared_l and self.shared_channels:
            return raw.view(-1, 1, 1)
        if self.shared_l:
            return raw.view(-1, 1, self.n_channels)
        per_shell = raw.view(-1, self.n_l, self.n_c)
        return per_shell.index_select(1, self.l_index)


class MessagePassing(nn.Module):
    """One CACE update of the mixed atomic density."""

    def __init__(
        self,
        kinds: FrozenSet[str],
        *,
        n_raw: int,
        n_radial: int,
        n_angular: int,
        n_channels: int,
        n_invariants: int,
        lmax: int,
        shells: Tensor,
        l_index: Tensor,
        cutoff: float,
        norm: float,
        shared_channels: bool,
        shared_l: bool,
        bchi_layers: int,
    ) -> None:
        super().__init__()
        self.kinds = frozenset(kinds)
        self.n_radial = int(n_radial)
        self.n_angular = int(n_angular)
        self.n_channels = int(n_channels)
        self.norm = float(norm)
        self.register_buffer("shells", shells.detach().to(dtype=torch.long).clone())
        n_shells = self.shells.shape[0]
        if "memory" in self.kinds:
            self.memory_coef = nn.Parameter(torch.full((n_shells, n_radial, n_channels), 0.25))
        if "a" in self.kinds:
            self.prefactor = nn.Parameter(torch.rand(n_shells, n_radial, n_channels))
            self.inv_r0 = nn.Parameter(
                (1.0 / float(cutoff))
                * (torch.rand(n_shells, n_radial, n_channels) + 0.5)
            )
        if "b" in self.kinds:
            self.gate = _BchiGate(
                n_radial * n_invariants * n_channels,
                n_channels,
                lmax,
                l_index,
                shared_channels,
                shared_l,
                bchi_layers,
            )

    def _scale_shells(self, density: Tensor, coefficient: Tensor) -> Tensor:
        scaled = torch.empty_like(density)
        for shell, (start, end) in enumerate(self.shells.tolist()):
            scaled[:, :, start:end, :] = (
                density[:, :, start:end, :] * coefficient[shell][None, :, None, :]
            )
        return scaled

    def forward(
        self,
        mixed: Tensor,
        invariant: Tensor,
        edge: Tensor,
        distance: Tensor,
        window: Tensor,
        receivers: Tensor,
        senders: Tensor,
        n_atoms: int,
        mix: RadialChannelMix,
    ) -> Tensor:
        updated = mixed.new_zeros(mixed.shape)
        if "a" in self.kinds and senders.numel():
            sender = mixed.index_select(0, senders)
            filtered = torch.empty_like(sender)
            for shell, (start, end) in enumerate(self.shells.tolist()):
                decay = torch.exp(
                    -distance[:, None, None] * self.inv_r0[shell]
                )
                decay = decay * self.prefactor[shell] * window[:, None, None]
                filtered[:, :, start:end, :] = (
                    sender[:, :, start:end, :] * decay[:, :, None, :]
                )
            updated = updated + scatter_sum(filtered, receivers, n_atoms)
        if "b" in self.kinds and senders.numel():
            weight = self.gate.angular_weight(invariant).index_select(0, senders)
            gated = edge * weight[:, None, :, :]
            updated = updated + mix(scatter_sum(gated, receivers, n_atoms))
        updated = updated * self.norm
        if "memory" in self.kinds:
            updated = updated + self._scale_shells(mixed, self.memory_coef)
        return updated
