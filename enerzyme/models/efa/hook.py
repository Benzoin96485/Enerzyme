"""Lightweight cross-architecture EFA wiring helpers.

Future Cores can call :func:`apply_efa_if_configured` inside a layer loop once
they expose invariant ``atom_feature`` / ``features``, absolute ``Ra``, and
``batch_seg``. EFA stays **inside** the Core (not a YAML physics layer).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from .block import EFABlock, parse_era_iterations


def build_efa_blocks(
    num_layers: int,
    dim_features: int,
    era_use_in_iterations: Optional[object] = None,
    **efa_kwargs,
) -> ModuleList:
    """Build a ``ModuleList`` of length ``num_layers`` with ``EFABlock`` or ``None``.

    Entries are ``EFABlock`` where layer index is in ``era_use_in_iterations``,
    else ``None`` (stored as :class:`torch.nn.Identity` placeholders are avoided;
    callers should check ``blocks[i] is not None``).

    Because ``ModuleList`` cannot hold raw ``None``, inactive slots use a
    disabled sentinel module :class:`_NoEFA`.
    """
    active = parse_era_iterations(era_use_in_iterations)
    active_set = set(active) if active is not None else set()
    blocks: list[Module] = []
    for i in range(num_layers):
        if i in active_set:
            blocks.append(EFABlock(dim_features, **efa_kwargs))
        else:
            blocks.append(_NoEFA())
    return ModuleList(blocks)


class _NoEFA(Module):
    """Placeholder when EFA is disabled for a layer."""

    def forward(
        self, features: Tensor, positions: Tensor, batch_seg: Tensor
    ) -> Tensor:
        return features.new_zeros(features.shape)


def apply_efa_if_configured(
    features: Tensor,
    positions: Tensor,
    batch_seg: Optional[Tensor],
    efa_block: Optional[Module],
    *,
    as_delta: bool = True,
) -> Tensor:
    """Apply one EFA block if present; otherwise return zeros (delta) or features.

    Parameters
    ----------
    features, positions:
        ``[N, F]``, ``[N, 3]``.
    batch_seg:
        ``[N]``; if ``None``, treats the batch as a single molecule.
    efa_block:
        ``EFABlock``, ``_NoEFA``, or ``None``.
    as_delta:
        If True (default), returns an additive update (zeros when inactive).
    """
    if efa_block is None or isinstance(efa_block, _NoEFA):
        return features.new_zeros(features.shape) if as_delta else features
    if batch_seg is None:
        batch_seg = torch.zeros(
            features.shape[0], dtype=torch.long, device=features.device
        )
    out = efa_block(features, positions, batch_seg)
    if as_delta:
        return out
    # If the block already returns a delta, add it; if full features, return as is.
    if getattr(efa_block, "as_delta", True):
        return features + out
    return out


def efa_input_fields(enabled: bool = True) -> set[str]:
    """Extra Core ``input_fields`` required when EFA is enabled."""
    if not enabled:
        return set()
    return {"Ra", "batch_seg"}
