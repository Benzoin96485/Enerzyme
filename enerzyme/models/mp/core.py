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
from .head import FeatureHead, Head, ReadoutHead  # noqa: F401


class Embed(nn.Module):
    """Builds the initial ``BlockState`` from raw inputs; ``input_fields`` lists what it reads."""

    input_fields: Tuple[str, ...] = ()

    def forward(self, inputs: Dict[str, Any], ctx: GraphContext) -> BlockState:  # pragma: no cover
        raise NotImplementedError


class FieldEmbed(Embed):
    """Initial state from a node input field.

    ``add_fields`` are summed onto ``field`` when present (charge / spin
    embeddings). ``slot="equi"`` stores the result as the equivariant state,
    for Cores whose node features are irreps from the start.
    ``vector_channels`` adds a zero Cartesian vector state ``(N, 3, C)``.
    """

    def __init__(self, field: str = "atom_embedding", add_fields: Sequence[str] = (),
                 width: Optional[int] = None, vector_channels: Optional[int] = None,
                 slot: str = "scalar") -> None:
        super().__init__()
        if slot not in ("scalar", "equi"):
            raise ValueError(f"slot must be 'scalar' or 'equi', got {slot!r}")
        self.field = field
        self.add_fields = tuple(add_fields)
        self.width = None if width is None else int(width)
        self.vector_channels = None if vector_channels is None else int(vector_channels)
        self.slot = slot
        self.input_fields = (field,) + self.add_fields

    def forward(self, inputs: Dict[str, Any], ctx: GraphContext) -> BlockState:
        x = inputs[self.field]
        for name in self.add_fields:
            extra = inputs.get(name)
            if extra is not None:
                x = x + extra
        if self.width is not None and x.shape[-1] != self.width:
            raise ValueError(f"{self.field} width {x.shape[-1]} != {self.width}")
        if self.slot == "equi":
            return BlockState(equi=x)
        equi = None
        if self.vector_channels is not None:
            equi = x.new_zeros(x.shape[0], 3, self.vector_channels)
        return BlockState(scalar=x, equi=equi)


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
        positional_fields: Sequence[str] = (),
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
        self.positional_fields = tuple(positional_fields)
        if len(set(self.positional_fields)) != len(self.positional_fields):
            raise ValueError("positional_fields must contain unique input names")
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

    def get_output(self, *args: Any, **inputs: Any) -> Dict[str, Tensor]:
        if args:
            if len(args) > len(self.positional_fields):
                raise TypeError(f"get_output accepts at most {len(self.positional_fields)} positional inputs")
            for field, value in zip(self.positional_fields, args):
                if field in inputs:
                    raise TypeError(f"get_output got multiple values for {field!r}")
                inputs[field] = value
        ctx = self.context(inputs)
        state = self.embed(inputs, ctx)
        taps: Optional[List[BlockState]] = [state] if self.head.needs_taps else None
        for index in self.schedule:
            state = self.blocks[index](state, ctx)
            if taps is not None:
                taps.append(state)
        return self.head(state, ctx, taps, inputs)
