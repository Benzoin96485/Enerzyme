"""Residual atomic networks used to refresh REANN orbital coefficients.

The block order follows ``NNMod`` / ``ResBlock`` in
https://github.com/zhangylch/REANN ``src/MODEL.py``: a projection into the
residual width, then residual blocks whose last linear is zero, then an
activation and a zero linear. LayerNorm sits on the activated features.
``nblock == 0`` keeps the projection and the final map, with no shortcut.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from ..activation import get_activation_fn


_ACTIVATION_ALIAS = {
    "swish": "swish",
    "relu_like": "relu_like",
    "relu-like": "relu_like",
    "tanh_like": "tanh_like",
    "tanh-like": "tanh_like",
}


def resolve_reann_activation(name: str) -> str:
    key = str(name).lower()
    if key not in _ACTIVATION_ALIAS:
        known = ", ".join(sorted(set(_ACTIVATION_ALIAS)))
        raise ValueError(f"activation must be one of {known}, got {name!r}")
    return _ACTIVATION_ALIAS[key]


def _tempered(name: str, width: int, fan: int) -> nn.Module:
    fan = max(int(fan), 1)
    if name == "tanh_like":
        alpha = 1.0 / math.sqrt(float(fan))
    else:
        alpha = 1.0
    return get_activation_fn(
        name,
        {
            "dim_feature": int(width),
            "initial_alpha": alpha,
            "initial_beta": 1.0 / float(fan),
            "learnable": True,
        },
    )


class REANNSubLayer(nn.Module):
    """Activation, optional LayerNorm and dropout, then a linear map."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        fan: int,
        activation: str,
        layer_norm: bool,
        dropout: float,
        zero_init: bool,
    ) -> None:
        super().__init__()
        self.activation = _tempered(activation, width_in, fan)
        self.norm = nn.LayerNorm(width_in) if layer_norm else None
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else None
        self.linear = nn.Linear(width_in, width_out)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.activation(features)
        if self.norm is not None:
            hidden = self.norm(hidden)
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.linear(hidden)


def _block_widths(hidden_dims: Sequence[int]) -> list[int]:
    widths = [int(width) for width in hidden_dims]
    if not widths or any(width < 1 for width in widths):
        raise ValueError(f"hidden_dims must be positive widths, got {list(hidden_dims)}")
    return [widths[0], *widths[1:], widths[0]]


class REANNMLP(nn.Module):
    """One atomic map from a density vector to coefficients or a scalar."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden_dims: Sequence[int],
        nblock: int,
        activation: str,
        layer_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(n_in) < 1 or int(n_out) < 1:
            raise ValueError(f"n_in and n_out must be >= 1, got {n_in}, {n_out}")
        if int(nblock) < 0:
            raise ValueError(f"nblock must be >= 0, got {nblock}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.nblock = int(nblock)
        self.activation_name = resolve_reann_activation(activation)
        widths = _block_widths(hidden_dims)
        hidden0 = widths[0]
        self.proj = nn.Linear(self.n_in, hidden0)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        blocks = []
        for _ in range(self.nblock):
            layers = []
            for index, (width_in, width_out) in enumerate(zip(widths[:-1], widths[1:])):
                fan = self.n_in if index == 0 else widths[index - 1]
                layers.append(
                    REANNSubLayer(
                        width_in,
                        width_out,
                        fan,
                        self.activation_name,
                        layer_norm,
                        float(dropout),
                        zero_init=(index == len(widths) - 2),
                    )
                )
            blocks.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(blocks)
        self.out_act = _tempered(self.activation_name, hidden0, hidden0)
        self.out = nn.Linear(hidden0, self.n_out)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.proj(features)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.out(self.out_act(hidden))


class SpeciesMLP(nn.Module):
    """One shared map, or one map for each listed atomic number."""

    def __init__(
        self,
        n_out: int,
        atomic_numbers: Optional[Sequence[int]],
        factory,
    ) -> None:
        super().__init__()
        self.n_out = int(n_out)
        if atomic_numbers is None:
            self.atomic_numbers = None
            self.shared: Optional[REANNMLP] = factory()
            self.by_species = nn.ModuleDict()
        else:
            numbers = tuple(int(number) for number in atomic_numbers)
            if not numbers:
                raise ValueError("atomic_numbers is empty")
            if len(set(numbers)) != len(numbers):
                raise ValueError(f"atomic_numbers contains duplicates: {list(numbers)}")
            self.atomic_numbers = numbers
            self.shared = None
            self.by_species = nn.ModuleDict({str(number): factory() for number in numbers})

    def forward(self, features: Tensor, species: Tensor) -> Tensor:
        if self.shared is not None:
            return self.shared(features)
        out = features.new_zeros((features.shape[0], self.n_out))
        if features.shape[0] == 0:
            return out
        species = species.reshape(-1).long()
        for atomic_number in torch.unique(species).tolist():
            key = str(int(atomic_number))
            if key not in self.by_species:
                known = list(self.atomic_numbers or [])
                raise ValueError(
                    f"atomic number {int(atomic_number)} has no REANN network; "
                    f"known species are {known}"
                )
            mask = species == int(atomic_number)
            chosen = mask.nonzero(as_tuple=True)[0]
            predicted = self.by_species[key](features.index_select(0, chosen))
            out = out.index_put((chosen,), predicted)
        return out
