"""Bias-free radial map and ordered-pair FiLM for DPA4C.

One SwiGLU trunk reads the raw radial basis. A second trunk reads the two
endpoint embeddings and returns a bounded per-channel scale, shift, and
optional mode mix. The gate is the first half of the pre-activation, matching
the DPA4C equations rather than the equivariant FFN SwiGLU used by DPA4.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def swiglu_hidden_width(width: int, multiple: int = 8) -> int:
    """Post-gate width ``8 * ceil(width / 3)``, aligned to ``multiple``."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    numerator = 8 * int(width)
    denominator = 3 * int(multiple)
    return int(multiple) * ((numerator + denominator - 1) // denominator)


def swiglu_gate_first(preactivation: Tensor) -> Tensor:
    """``SiLU(gate) * value``, with the gate in the leading half."""
    gate, value = preactivation.chunk(2, dim=-1)
    return F.silu(gate) * value


class RadialAmplitude(nn.Module):
    """Shared radial map ``g(ρ)`` and optional mode profiles ``q(ρ)``."""

    def __init__(self, n_radial: int, channels: int, radial_modes: int) -> None:
        super().__init__()
        if n_radial < 1:
            raise ValueError(f"n_radial must be >= 1, got {n_radial}")
        if radial_modes < 0:
            raise ValueError(f"radial_modes must be >= 0, got {radial_modes}")
        self.channels = int(channels)
        self.radial_modes = int(radial_modes)
        hidden = swiglu_hidden_width(self.channels)
        self.hidden_width = hidden
        self.in_proj = nn.Linear(int(n_radial), 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, self.channels, bias=False)
        self.mode_head = (
            nn.Linear(hidden, self.radial_modes, bias=False)
            if self.radial_modes > 0
            else None
        )

    def forward(self, radial_basis: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        hidden = swiglu_gate_first(self.in_proj(radial_basis))
        amplitude = self.out_proj(hidden)
        if self.mode_head is None:
            return amplitude, None
        return amplitude, self.mode_head(hidden)


class OrderedPairFiLM(nn.Module):
    """Per-edge scale, shift, and mode mix from the two endpoint embeddings.

    ``γ = 1 + tanh(s)`` lies in ``(0, 2)`` and the learned part of ``β`` and
    ``U`` lies in ``(-1, 1)``. The network output is scaled by 0.1 before
    those bounds. Evaluating it on gathered embeddings is the same map as a
    finite type-pair cache when the embedding depends only on species.
    """

    OUTPUT_SCALE = 0.1

    def __init__(self, channels: int, radial_modes: int) -> None:
        super().__init__()
        if radial_modes < 0:
            raise ValueError(f"radial_modes must be >= 0, got {radial_modes}")
        self.channels = int(channels)
        self.radial_modes = int(radial_modes)
        hidden = swiglu_hidden_width(2 * self.channels)
        self.hidden_width = hidden
        out_width = self.channels * (2 + self.radial_modes)
        self.in_proj = nn.Linear(2 * self.channels, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, out_width, bias=False)

    def forward(
        self,
        center: Tensor,
        neighbor: Tensor,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        hidden = swiglu_gate_first(self.in_proj(torch.cat((center, neighbor), dim=-1)))
        logits = self.OUTPUT_SCALE * self.out_proj(hidden)
        scale = 1.0 + torch.tanh(logits[..., : self.channels])
        shift = center + neighbor + torch.tanh(logits[..., self.channels : 2 * self.channels])
        if self.radial_modes == 0:
            return scale, shift, None
        mixing = torch.tanh(logits[..., 2 * self.channels :]).reshape(
            center.shape[0],
            self.channels,
            self.radial_modes,
        )
        return scale, shift, mixing
