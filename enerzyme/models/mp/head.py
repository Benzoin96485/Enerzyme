"""Core heads: map the final state and per-block taps to the Core's output fields."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .block import BlockState
from .graph import GraphContext


class Head(nn.Module):
    """Maps the final state (and optional per-block taps) to the Core's output fields.

    ``taps[0]`` is the embedded state and ``taps[k]`` the state after block ``k``.
    """

    input_fields: Tuple[str, ...] = ()
    output_fields: Tuple[str, ...] = ("atom_feature",)
    needs_taps: bool = False

    def forward(self, state: BlockState, ctx: GraphContext, taps: Optional[List[BlockState]],
                inputs: Dict[str, Any]) -> Dict[str, Tensor]:  # pragma: no cover
        raise NotImplementedError


def _read(state: BlockState, source: str) -> Tensor:
    if source == "scalar":
        return state.scalar
    if source == "equi":
        return state.equi
    return state.aux[source]


OUTPUT_MODES = ("feature", "layers", "concat", "sum")


class FeatureHead(Head):
    """Expose node features for the readout layers.

    ``output_mode``:

    * ``"feature"``: the final state's ``source`` as ``atom_feature``
    * ``"layers"``: per-block features stacked on a new last axis ``(N, F, B)``
    * ``"concat"``: per-block features concatenated on the feature axis
    * ``"sum"``: sum of per-block features

    ``source`` is ``"scalar"``, ``"equi"`` or an ``aux`` key. ``include_embedding`` adds
    the embedded state as the first tap. ``equi_field`` also emits
    ``state.equi`` (e.g. ``atom_vector_feature``).
    """

    def __init__(self, output_mode: str = "feature", scalar_field: str = "atom_feature",
                 equi_field: Optional[str] = None, include_embedding: bool = False,
                 source: str = "scalar") -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {output_mode!r}")
        self.output_mode = output_mode
        self.scalar_field = scalar_field
        self.equi_field = equi_field
        self.include_embedding = bool(include_embedding)
        self.source = source
        self.output_fields = tuple(f for f in (scalar_field, equi_field) if f)
        self.needs_taps = output_mode != "feature"

    def forward(self, state: BlockState, ctx: GraphContext, taps: Optional[List[BlockState]],
                inputs: Dict[str, Any]) -> Dict[str, Tensor]:
        if self.output_mode == "feature":
            feature = _read(state, self.source)
        else:
            chosen = taps if self.include_embedding else taps[1:]
            features = [_read(tap, self.source) for tap in chosen]
            if self.output_mode == "layers":
                feature = torch.stack(features, dim=-1)
            elif self.output_mode == "concat":
                feature = torch.cat(features, dim=-1)
            else:
                feature = features[0]
                for item in features[1:]:
                    feature = feature + item
        out = {self.scalar_field: feature}
        if self.equi_field:
            out[self.equi_field] = state.equi
        return out

    def extra_repr(self) -> str:
        return f"output_mode={self.output_mode}, source={self.source}"


READOUT_MODES = ("final", "per_block", "sum_taps")


class ReadoutHead(Head):
    """Direct per-atom outputs (``Ea``, ``Qa``, ...) from node features.

    ``mode``:

    * ``"final"``: ``readout(final.source)`` (SchNet)
    * ``"per_block"``: ``Σ_k readouts[k](tap_k.source)`` (PhysNet, MACE)
    * ``"sum_taps"``: ``readout(Σ_k tap_k.source)`` (SpookyNet)

    The readout output has one column per entry of ``fields``; with
    ``ensemble_size > 1`` it is ``(N, F, S)`` or a flat ``(N, F*S)`` that is
    reshaped. ``nh_loss`` adds PhysNet's non-hierarchicality penalty between
    consecutive per-block outputs. ``l2_penalty`` sums ``readout.output.l2loss()``
    over the per-block readouts. ``block_keep_prob < 1`` drops whole block
    contributions per molecule during training (SpookyNet module dropout).
    """

    def __init__(
        self,
        readout: Optional[nn.Module] = None,
        readouts: Optional[Sequence[nn.Module]] = None,
        mode: str = "final",
        source: str = "scalar",
        fields: Sequence[str] = ("Ea", "Qa"),
        ensemble_size: int = 1,
        nh_loss: bool = False,
        l2_penalty: bool = False,
        block_keep_prob: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in READOUT_MODES:
            raise ValueError(f"mode must be one of {READOUT_MODES}, got {mode!r}")
        if (mode == "per_block") == (readouts is None) or (mode != "per_block") == (readout is None):
            raise ValueError("per_block takes `readouts`; final/sum_taps take `readout`")
        self.mode = mode
        self.source = source
        self.fields = tuple(fields)
        self.ensemble_size = int(ensemble_size)
        self.nh_loss = bool(nh_loss)
        self.l2_penalty = bool(l2_penalty) and self.ensemble_size > 1
        self.block_keep_prob = float(block_keep_prob)
        if readout is not None:
            self.readout = readout
        if readouts is not None:
            self.readouts = nn.ModuleList(readouts)
        self.needs_taps = mode != "final"
        outputs = list(self.fields)
        if self.nh_loss:
            outputs.append("nh_loss")
        if self.l2_penalty:
            outputs.append("l2_penalty")
        self.output_fields = tuple(outputs)

    def _shape(self, out: Tensor) -> Tensor:
        if self.ensemble_size > 1 and out.dim() == 2:
            return out.reshape(-1, len(self.fields), self.ensemble_size)
        return out

    def _split(self, out: Tensor) -> Dict[str, Tensor]:
        return {name: out[:, k] for k, name in enumerate(self.fields)}

    def forward(self, state: BlockState, ctx: GraphContext, taps: Optional[List[BlockState]],
                inputs: Dict[str, Any]) -> Dict[str, Tensor]:
        if self.mode == "final":
            return self._split(self._shape(self.readout(_read(state, self.source))))
        if self.mode == "sum_taps":
            feature = None
            keep = None
            for tap in taps[1:]:
                item = _read(tap, self.source)
                if self.training and self.block_keep_prob < 1.0:
                    if keep is None:
                        keep = item.new_ones(ctx.num_graphs(), 1)
                    item = item * keep[ctx.batch]
                    keep = keep * torch.bernoulli(torch.full_like(keep, self.block_keep_prob))
                feature = item if feature is None else feature + item
            return self._split(self._shape(self.readout(feature)))

        total = None
        nh_loss = None
        last_sq = None
        for readout, tap in zip(self.readouts, taps[1:]):
            out = self._shape(readout(_read(tap, self.source)))
            total = out if total is None else total + out
            if self.nh_loss:
                sq = out ** 2
                if last_sq is not None:
                    term = torch.mean(sq / (sq + last_sq + 1e-7))
                    nh_loss = term if nh_loss is None else nh_loss + term
                last_sq = sq
        result = self._split(total)
        if self.nh_loss:
            result["nh_loss"] = nh_loss if nh_loss is not None else total.new_zeros(())
        if self.l2_penalty:
            penalty = None
            for readout in self.readouts:
                term = readout.output.l2loss()
                penalty = term if penalty is None else penalty + term
            result["l2_penalty"] = penalty
        return result

    def extra_repr(self) -> str:
        return f"mode={self.mode}, source={self.source}, fields={self.fields}"
