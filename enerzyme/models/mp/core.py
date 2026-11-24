"""``ComposableCore``: one Core class assembled from embed, blocks and head.

Presets (``enerzyme.models.presets``) build the parts and return a
``ComposableCore``; the layer stack treats it like any other
:class:`~enerzyme.models.layers.BaseFFCore`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn

from ..layers import BaseFFCore
from .block import BlockState
from .graph import GraphContext, GraphRequires


class Embed(nn.Module):
    """Builds the initial ``BlockState`` from raw inputs; ``input_fields`` lists what it reads."""

    input_fields: Tuple[str, ...] = ()

    def forward(self, inputs: Dict[str, Any], ctx: GraphContext) -> BlockState:  # pragma: no cover
        raise NotImplementedError


class Head(nn.Module):
    """Maps the final state (and optional per-block taps) to the Core's output fields."""

    input_fields: Tuple[str, ...] = ()
    output_fields: Tuple[str, ...] = ("atom_feature",)
    needs_taps: bool = False

    def forward(self, state: BlockState, ctx: GraphContext, taps: Optional[List[BlockState]],
                inputs: Dict[str, Any]) -> Dict[str, Tensor]:  # pragma: no cover
        raise NotImplementedError


class FieldEmbed(Embed):
    """``BlockState(scalar=inputs[field])``, optionally with a zero Cartesian vector state."""

    def __init__(self, field: str = "atom_embedding", width: Optional[int] = None,
                 vector_channels: Optional[int] = None) -> None:
        super().__init__()
        self.field = field
        self.width = None if width is None else int(width)
        self.vector_channels = None if vector_channels is None else int(vector_channels)
        self.input_fields = (field,)

    def forward(self, inputs: Dict[str, Any], ctx: GraphContext) -> BlockState:
        x = inputs[self.field]
        if self.width is not None and x.shape[-1] != self.width:
            raise ValueError(f"{self.field} width {x.shape[-1]} != {self.width}")
        equi = None
        if self.vector_channels is not None:
            equi = x.new_zeros(x.shape[0], 3, self.vector_channels)
        return BlockState(scalar=x, equi=equi)


OUTPUT_MODES = ("feature", "layers", "concat", "sum")


class FeatureHead(Head):
    """Expose node features for the readout layers.

    ``output_mode``:

    * ``"feature"``: final ``state.scalar`` as ``atom_feature``
    * ``"layers"``: per-block scalars stacked on a new last axis ``(N, F, B)``
    * ``"concat"``: per-block scalars concatenated on the feature axis
    * ``"sum"``: sum of per-block scalars

    ``include_embedding`` adds the embedded state as the first tap.
    ``equi_field`` also emits ``state.equi`` (e.g. ``atom_vector_feature``).
    """

    def __init__(self, output_mode: str = "feature", scalar_field: str = "atom_feature",
                 equi_field: Optional[str] = None, include_embedding: bool = False) -> None:
        super().__init__()
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {output_mode!r}")
        self.output_mode = output_mode
        self.scalar_field = scalar_field
        self.equi_field = equi_field
        self.include_embedding = bool(include_embedding)
        self.output_fields = tuple(f for f in (scalar_field, equi_field) if f)
        self.needs_taps = output_mode != "feature"

    def forward(self, state: BlockState, ctx: GraphContext, taps: Optional[List[BlockState]],
                inputs: Dict[str, Any]) -> Dict[str, Tensor]:
        if self.output_mode == "feature":
            feature = state.scalar
        else:
            chosen = taps if self.include_embedding else taps[1:]
            scalars = [tap.scalar for tap in chosen]
            if self.output_mode == "layers":
                feature = torch.stack(scalars, dim=-1)
            elif self.output_mode == "concat":
                feature = torch.cat(scalars, dim=-1)
            else:
                feature = torch.stack(scalars, dim=0).sum(dim=0)
        out = {self.scalar_field: feature}
        if self.equi_field:
            out[self.equi_field] = state.equi
        return out

    def extra_repr(self) -> str:
        return f"output_mode={self.output_mode}"


def radius_stack_builder(core: BaseFFCore, built_layers: List[nn.Module]) -> None:
    core.adopt_radius_stack(built_layers)


class ComposableCore(BaseFFCore):
    """``embed → blocks[schedule] → head`` over a shared :class:`GraphContext`.

    ``schedule`` lists block indices in execution order, so weight sharing is
    ``blocks=[b], schedule=[0, 0, 0]`` without duplicate state_dict entries.
    ``node_field`` gives the node count and compute dtype. ``builder`` places
    the surrounding layer stack (default: :meth:`BaseFFCore.adopt_radius_stack`).
    ``attributes`` are set on the Core for readouts (``feature_irreps``,
    ``dim_feature_out``, ``vector_feature_layout``, ...).
    """

    def __init__(
        self,
        embed: Embed,
        blocks: Sequence[nn.Module],
        head: Head,
        requires: GraphRequires,
        *,
        schedule: Optional[Sequence[int]] = None,
        node_field: str = "atom_embedding",
        builder: Optional[Callable[[BaseFFCore, List[nn.Module]], None]] = None,
        banner: Optional[str] = None,
        extra_input_fields: Iterable[str] = (),
        **attributes: Any,
    ) -> None:
        input_fields: Set[str] = set(requires.input_fields())
        input_fields |= set(getattr(embed, "input_fields", ()))
        input_fields |= set(getattr(head, "input_fields", ()))
        for block in blocks:
            input_fields |= set(getattr(block, "input_fields", ()))
        input_fields |= set(extra_input_fields)
        input_fields.add(node_field)
        super().__init__(input_fields=input_fields, output_fields=set(head.output_fields))
        self.embed = embed
        self.blocks = nn.ModuleList(blocks)
        self.head = head
        self.requires = requires
        self.schedule = list(range(len(self.blocks))) if schedule is None else [int(i) for i in schedule]
        if any(i < 0 or i >= len(self.blocks) for i in self.schedule):
            raise ValueError(f"schedule {self.schedule} indexes outside {len(self.blocks)} blocks")
        self.node_field = node_field
        self._builder = builder or radius_stack_builder
        self._banner = banner
        for key, value in attributes.items():
            setattr(self, key, value)

    def __str__(self) -> str:
        return self._banner if self._banner is not None else super().__str__()

    def build(self, built_layers: List[nn.Module]) -> None:
        self._builder(self, built_layers)

    def compute_dtype(self, inputs: Dict[str, Any]) -> Optional[torch.dtype]:
        node = inputs.get(self.node_field)
        if torch.is_tensor(node) and node.is_floating_point():
            return node.dtype
        for p in self.parameters():
            return p.dtype
        return None

    def context(self, inputs: Dict[str, Any]) -> GraphContext:
        node = inputs[self.node_field]
        return GraphContext.from_inputs(inputs, self.requires, num_nodes=node.shape[0],
                                        dtype=self.compute_dtype(inputs))

    def get_output(self, **inputs: Any) -> Dict[str, Tensor]:
        ctx = self.context(inputs)
        state = self.embed(inputs, ctx)
        taps: Optional[List[BlockState]] = [state] if self.head.needs_taps else None
        for index in self.schedule:
            state = self.blocks[index](state, ctx)
            if taps is not None:
                taps.append(state)
        return self.head(state, ctx, taps, inputs)
