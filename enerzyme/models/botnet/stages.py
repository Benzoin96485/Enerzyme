"""BOTNet species-conditioned convolutions and layer-feature placement."""

import torch
from torch import Tensor
from torch.nn import functional as F
from e3nn.o3 import Irreps
from typing import List, Optional, Sequence

from ..mp.core import FieldEmbed
from ..mp.head import Head
from ..mp.block import BlockState
from .interaction import INTERACTION_CLASSES, VARIANTS


class SpeciesEmbed(FieldEmbed):
    def __init__(self, width, num_rbf, max_Za, sph):
        super().__init__(width=width, add_fields=("charge_embedding", "spin_embedding"), slot="equi")
        self.num_rbf = num_rbf
        self.max_Za = max_Za
        self.sph = sph

    def forward(self, inputs, ctx):
        embedding = inputs["atom_embedding"]
        if embedding.ndim != 2 or embedding.shape[-1] != self.width:
            raise ValueError(f"atom_embedding must be [N, dim_embedding={self.width}]")
        if ctx.receiver.shape != ctx.sender.shape or ctx.vec.shape != (ctx.num_edges, 3):
            raise ValueError("idx_i_sr, idx_j_sr and vij_sr must describe the same edges")
        if ctx.rbf.shape != (ctx.num_edges, self.num_rbf):
            raise ValueError(f"rbf must have one row per edge and width num_rbf={self.num_rbf}")
        z = inputs["Za"].long().view(-1)
        if z.shape[0] != ctx.num_nodes:
            raise ValueError("Za and atom_embedding must have the same number of atoms")
        if z.numel() and (int(z.min()) < 0 or int(z.max()) > self.max_Za):
            raise ValueError(f"Za must lie in 0..{self.max_Za}")
        x = super().forward(inputs, ctx).equi
        node_attrs = F.one_hot(z, self.max_Za + 1).to(x.dtype)
        # The BOTNet interaction uses i as sender but harmonics of R_j - R_i.
        return BlockState(scalar=node_attrs, equi=x, edge=self.sph(-ctx.vec))


def body_ordered_update(interaction, state, ctx):
    x = interaction(node_attrs=state.scalar, node_feats=state.equi, edge_attrs=state.edge,
                    edge_feats=ctx.rbf, sender=ctx.sender, receiver=ctx.receiver)
    return BlockState(scalar=state.scalar, equi=x, edge=state.edge)


class PlacedFeatureHead(Head):
    """Embed nonuniform layer outputs into their union irreps before stacking."""

    needs_taps = True

    def __init__(self, layer_irreps, union):
        super().__init__()
        self.width = union.dim
        for i, irreps in enumerate(layer_irreps):
            self.register_buffer(f"placement_{i}", _column_index(irreps, union), persistent=False)

    def forward(self, state, ctx, taps, inputs):
        features = [_place(tap.equi, getattr(self, f"placement_{i}"), self.width)
                    for i, tap in enumerate(taps[1:])]
        return {"atom_feature": torch.stack(features, dim=-1)}


def _column_index(src: Irreps, dst: Irreps) -> Tensor:
    """Columns of ``dst`` that receive each column of ``src``, in ``src`` order."""
    dst = Irreps(dst).simplify()
    src = Irreps(src).simplify()
    starts = {}
    offset = 0
    for mul, ir in dst:
        starts[ir] = (offset, mul)
        offset += mul * ir.dim
    columns: List[int] = []
    for mul, ir in src:
        if ir not in starts:
            raise ValueError(f"{ir} is in a layer output but not in the stacked irreps")
        start, dst_mul = starts[ir]
        if mul != dst_mul:
            raise ValueError(
                f"Multiplicity of {ir} is {mul} in one layer and {dst_mul} in the stack"
            )
        width = mul * ir.dim
        columns.extend(range(start, start + width))
    if len(columns) != src.dim:
        raise ValueError(
            f"Placement for {src} has {len(columns)} columns, expected {src.dim}"
        )
    return torch.tensor(columns, dtype=torch.long)


def _union_irreps(layer_irreps: Sequence[Irreps]) -> Irreps:
    multiplicity = {}
    for irreps in layer_irreps:
        for mul, ir in Irreps(irreps).simplify():
            multiplicity[ir] = max(multiplicity.get(ir, 0), mul)
    if not multiplicity:
        raise ValueError("BOTNet produced empty irreps")
    return Irreps(
        [(multiplicity[ir], ir) for ir in sorted(multiplicity, key=lambda ir: (ir.l, -ir.p))]
    ).simplify()


def _place(features: Tensor, columns: Tensor, dim_out: int) -> Tensor:
    laid = features.new_zeros(features.shape[0], dim_out)
    laid[:, columns] = features
    return laid

def _resolve_interactions(
    variant: str,
    interaction_cls_first: Optional[str],
    interaction_cls: Optional[str],
) -> tuple:
    if variant not in VARIANTS and (
        interaction_cls_first is None or interaction_cls is None
    ):
        raise ValueError(
            f"Unknown variant {variant!r}. Choices: {sorted(VARIANTS)}. "
            "Set both interaction class names to bypass the variant."
        )
    if variant in VARIANTS:
        default_first, default_rest = VARIANTS[variant]
    else:
        default_first, default_rest = interaction_cls_first, interaction_cls
    first = default_first if interaction_cls_first is None else interaction_cls_first
    rest = default_rest if interaction_cls is None else interaction_cls
    unknown = [name for name in (first, rest) if name not in INTERACTION_CLASSES]
    if unknown:
        raise KeyError(
            f"Unknown interaction class {unknown[0]!r}. "
            f"Choices: {sorted(INTERACTION_CLASSES)}"
        )
    return first, rest
