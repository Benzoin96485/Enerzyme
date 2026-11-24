"""ViSNet-LSRM Core (Li et al., ICLR 2024).

Short-range message passing is :func:`~enerzyme.models.presets.cartesian_vector.visnet`.
The long-range branch is the released serial ViSNet-LSRM block: after the
short stack, fragment states are mean-pooled from that state and atoms attend
to fragments inside ``cutoff_lr``. The two branches are normalized separately
and concatenated, which is the late fusion in Eq. (5). The scalar head stays
:class:`~enerzyme.models.layers.readout.SimpleReadout` on the concatenated
invariant, so the readout width is ``2 F``.

Adapted from ``Visnorm_shared_LSRMNorm2_2branchSerial`` (the class named
ViSNet-LSRM in ``liyy2/LSR-MP``). That repository is not imported, and its
checkpoints do not load. The short-range edge state remains ViSNet's
``(h_i + h_j) Linear(rbf)``. The LSR-MP training script drops self pairs and
uses a radial-only edge; ``include_self`` defaults to false here for that
script, and can be turned back on.

BRICS labels are a data field, ``cluster_ids``, already registered for
E2Former-LSR. Online ``kmeans`` and ``spectral`` assignment reuse
:mod:`enerzyme.models.layers.fragments`. Fragment centers are recomputed from
the current coordinates. ``center='atomic_number'`` is the released
``center_of_mass`` path (weights are atomic numbers, not isotopic masses).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear
from ..mp.ops import scatter_mean

from ..layers.rbf import ExpNormalSmearing
from ..mp.block import Block, BlockState
from ..mp.head import Head
from .interaction import VecLayerNorm, _xavier_linear
from .long_range import (
    FragmentBipartiteBlock,
    bipartite_radius_edges,
    normalize_center,
    normalize_fragment_mode,
    resolve_long_range_fragments,
)


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# Table 10 / the released training script: 6 short layers, 2 long layers,
# 128 channels, 8 heads, lmax 1, 4 Å short cutoff, 9 Å long cutoff.
# num_rbf 50 and trainable radial bases are the class defaults in lsrm_modules.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 50,
    "max_Za": 100,
    "cutoff_sr": 4.0,
    "cutoff_lr": 9.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_MD22_BUILD_PARAMS = dict(DEFAULT_BUILD_PARAMS)

PUBLISHED_MD22_CORE_PARAMS = {
    "num_layers": 6,
    "long_layers": 2,
    "num_heads": 8,
    "lmax": 1,
    "vertex_type": "none",
    "vecnorm_type": "max_min",
    "trainable_vecnorm": False,
    "neighbor_embedding": True,
    "include_self": False,
    "activation": "silu",
    "attn_activation": "silu",
    "cutoff_lr": 9.0,
    "fragment_mode": "kmeans",
    "center": "atomic_number",
    "min_nodes_per_group": 24,
    "kmeans_random_state": 0,
    "dropout": 0.0,
    "trainable_rbf_lr": True,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": True}},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": dict(PUBLISHED_MD22_CORE_PARAMS)},
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _batch_index(batch_seg: Optional[Tensor], n_atoms: int, device: torch.device) -> Tensor:
    if batch_seg is None:
        return torch.zeros(n_atoms, dtype=torch.long, device=device)
    index = batch_seg.reshape(-1).long()
    if index.shape[0] != n_atoms:
        raise ValueError(
            f"batch_seg must have length {n_atoms}, got {index.shape[0]}"
        )
    return index


class FragmentEdges(Block):
    """Seed the long-range state and prepare the atom-to-fragment graph once."""

    input_fields = ("Ra", "Za", "batch_seg", "cluster_ids")

    def __init__(self, num_rbf, width, cutoff_lr, trainable, mode, center, min_nodes, seed, active):
        super().__init__()
        self.dim_embedding = width
        self.cutoff_lr = cutoff_lr
        self.mode = mode
        self.center = center
        self.min_nodes_per_group = max(1, min_nodes)
        self.kmeans_random_state = seed
        self.active = active
        self.long_rbf = ExpNormalSmearing(num_rbf=num_rbf, cutoff_sr=cutoff_lr, trainable=trainable)
        self.long_rbf_proj = Linear(num_rbf, width)

    def forward(self, state, ctx):
        scalar = state.scalar
        vector = state.equi[:, :3].contiguous()
        n_atoms = ctx.num_nodes
        short = {"short": state}
        Ra = ctx.inputs.get("Ra")
        if self.active and Ra is None:
            raise ValueError("long-range fragments require atomic positions Ra")
        if not self.active or n_atoms == 0:
            return BlockState(scalar.new_zeros(scalar.shape), vector.new_zeros(vector.shape), None, short)
        Za = ctx.inputs.get("Za")
        batch_seg = ctx.batch
        cluster_ids = ctx.inputs.get("cluster_ids")
        batch = _batch_index(batch_seg, n_atoms, scalar.device)
        flat_ids, centers, cluster_batch = resolve_long_range_fragments(
            Ra.to(dtype=scalar.dtype),
            batch,
            self.mode,
            self.center,
            None if Za is None else Za.reshape(-1),
            cluster_ids,
            self.min_nodes_per_group,
            self.kmeans_random_state,
        )
        idx_atom, idx_fragment, distance, direction = bipartite_radius_edges(
            Ra.to(dtype=scalar.dtype),
            centers,
            batch,
            cluster_batch,
            self.cutoff_lr,
        )
        if idx_atom.numel() == 0:
            edge_attr = scalar.new_zeros(0, self.dim_embedding)
        else:
            edge_attr = self.long_rbf_proj(self.long_rbf.get_rbf(distance))
        ctx.cache["fragment_graph"] = flat_ids, idx_atom, idx_fragment, edge_attr, direction
        return BlockState(scalar, vector, None, short)


def fragment_step(block, state, ctx):
    if ctx.num_nodes == 0:
        return state
    ids, atoms, fragments, edges, directions = ctx.cache["fragment_graph"]
    vector = state.equi
    group_scalar = scatter_mean(state.scalar, ids, dim=0)
    group_vector = scatter_mean(vector.reshape(ctx.num_nodes, -1), ids, dim=0).reshape(-1, 3, vector.shape[-1])
    ds, dv = block(state.scalar, vector, group_scalar, group_vector, atoms, fragments, edges, directions)
    return BlockState(state.scalar + ds, vector + dv, None, state.aux)


class FragmentFusionHead(Head):
    def __init__(self, short_head, width):
        super().__init__()
        self.short = short_head
        self.output_fields = short_head.output_fields
        self.out_norm_long = LayerNorm(width)
        self.vec_out_norm_long = VecLayerNorm(width, trainable=False, norm_type="max_min")

    def forward(self, state, ctx, taps, inputs):
        out = self.short(state.aux["short"], ctx, taps, inputs)
        out["atom_feature"] = torch.cat((out["atom_feature"], self.out_norm_long(state.scalar)), dim=-1)
        out["atom_vector_feature"] = torch.cat((out["atom_vector_feature"], self.vec_out_norm_long(state.equi)),
                                               dim=-1).contiguous()
        return out
