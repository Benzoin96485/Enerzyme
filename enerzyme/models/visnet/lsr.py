"""ViSNet-LSRM Core (Li et al., ICLR 2024).

Short-range message passing is :class:`~enerzyme.models.visnet.core.ViSNetCore`.
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

from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import LayerNorm, Linear, ModuleList
from torch_scatter import scatter_mean

from ..layers.rbf import ExpNormalSmearing
from .core import ViSNetCore
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


class ViSNetLSRMCore(ViSNetCore):
    """ViSNet short-range stack plus serial fragment long-range fusion."""

    def __str__(self) -> str:
        return """
################################################################################
# ViSNet-LSRM (Li et al., ICLR 2024)                                           #
# short ViSNet, then atom–fragment long-range messages, late concat           #
# atom_feature [N, 2F], atom_vector_feature [N, 3, 2F]                        #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 50,
        max_Za: int = 100,
        cutoff_sr: float = 4.0,
        num_layers: int = 6,
        long_layers: int = 2,
        num_heads: int = 8,
        lmax: int = 1,
        vertex_type: object = "none",
        vecnorm_type: object = "max_min",
        trainable_vecnorm: bool = False,
        neighbor_embedding: bool = True,
        include_self: bool = False,
        activation: object = "silu",
        attn_activation: object = "silu",
        cutoff_lr: float = 9.0,
        fragment_mode: object = "kmeans",
        center: object = "atomic_number",
        min_nodes_per_group: int = 24,
        kmeans_random_state: int = 0,
        dropout: float = 0.0,
        trainable_rbf_lr: bool = True,
    ) -> None:
        if int(long_layers) < 0:
            raise ValueError(f"long_layers must be >= 0, got {long_layers}")
        if float(cutoff_lr) <= 0.0:
            raise ValueError(f"cutoff_lr must be positive, got {cutoff_lr}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        mode = normalize_fragment_mode(fragment_mode)
        center_name = normalize_center(center)
        super().__init__(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            max_Za=max_Za,
            cutoff_sr=cutoff_sr,
            num_layers=num_layers,
            num_heads=num_heads,
            lmax=lmax,
            vertex_type=vertex_type,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            neighbor_embedding=neighbor_embedding,
            include_self=include_self,
            activation=activation,
            attn_activation=attn_activation,
        )
        self._input_fields |= {"Ra", "batch_seg", "cluster_ids"}
        self._relevant_fields |= {"Ra", "batch_seg", "cluster_ids"}
        for name in ("Ra", "batch_seg", "cluster_ids"):
            self._name_mapping[name] = name

        self.long_layers = int(long_layers)
        self.cutoff_lr = float(cutoff_lr)
        self.fragment_mode = mode
        self.center = center_name
        self.min_nodes_per_group = max(int(min_nodes_per_group), 1)
        self.kmeans_random_state = int(kmeans_random_state)
        self.dropout = float(dropout)
        self.trainable_rbf_lr = bool(trainable_rbf_lr)
        # Late fusion concatenates the short and long invariants.
        self.dim_feature_out = 2 * self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"

        self.long_rbf = ExpNormalSmearing(
            num_rbf=self.num_rbf,
            cutoff_sr=self.cutoff_lr,
            trainable=self.trainable_rbf_lr,
        )
        self.long_rbf_proj = Linear(self.num_rbf, self.dim_embedding)
        self.long_blocks = ModuleList(
            [
                FragmentBipartiteBlock(
                    self.dim_embedding, self.num_heads, self.dropout
                )
                for _ in range(self.long_layers)
            ]
        )
        self.out_norm_long = LayerNorm(self.dim_embedding)
        # Section 4.1 min-max vector norm, independent of the short-range setting.
        self.vec_out_norm_long = VecLayerNorm(
            self.dim_embedding, trainable=False, norm_type="max_min"
        )
        _xavier_linear(self.long_rbf_proj)

    def _long_range(
        self,
        scalar: Tensor,
        vector_l1: Tensor,
        Ra: Tensor,
        Za: Optional[Tensor],
        batch_seg: Optional[Tensor],
        cluster_ids: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Serial long-range residuals. ``long_layers == 0`` returns zeros."""
        n_atoms = int(scalar.shape[0])
        if self.long_layers == 0 or n_atoms == 0:
            return (
                scalar.new_zeros(n_atoms, self.dim_embedding),
                vector_l1.new_zeros(n_atoms, 3, self.dim_embedding),
            )
        batch = _batch_index(batch_seg, n_atoms, scalar.device)
        flat_ids, centers, cluster_batch = resolve_long_range_fragments(
            Ra.to(dtype=scalar.dtype),
            batch,
            self.fragment_mode,
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
        long_scalar = scalar
        long_vector = vector_l1
        for block in self.long_blocks:
            group_scalar = scatter_mean(long_scalar, flat_ids, dim=0)
            group_vector = scatter_mean(
                long_vector.reshape(n_atoms, -1), flat_ids, dim=0
            ).reshape(-1, 3, self.dim_embedding)
            delta_scalar, delta_vector = block(
                long_scalar,
                long_vector,
                group_scalar,
                group_vector,
                idx_atom,
                idx_fragment,
                edge_attr,
                direction,
            )
            long_scalar = long_scalar + delta_scalar
            long_vector = long_vector + delta_vector
        return long_scalar, long_vector

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
        Za: Optional[Tensor] = None,
        Ra: Optional[Tensor] = None,
        batch_seg: Optional[Tensor] = None,
        cluster_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Short-range ViSNet, then serial long-range fusion.

        ``Ra`` is required when ``long_layers > 0`` because fragment centers
        are means of the current coordinates. ``cluster_ids`` is required only
        for ``fragment_mode='precomputed'``.
        """
        if self.long_layers > 0 and Ra is None:
            raise ValueError("long-range fragments require atomic positions Ra")
        scalar, vector = self.encode_state(
            atom_embedding,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            Dij_sr,
            cutoff_values_sr,
            Za,
        )
        long_scalar, long_vector = self._long_range(
            scalar, vector[:, :3].contiguous(), Ra, Za, batch_seg, cluster_ids
        )
        scalar = self.out_norm(scalar)
        vector = self.vec_out_norm(vector)
        long_scalar = self.out_norm_long(long_scalar)
        long_vector = self.vec_out_norm_long(long_vector)
        output = {
            "atom_feature": torch.cat((scalar, long_scalar), dim=-1),
            "atom_vector_feature": torch.cat(
                (vector[:, :3], long_vector), dim=-1
            ).contiguous(),
        }
        if self.lmax == 2:
            output["atom_l2_feature"] = vector[:, 3:].contiguous()
        return output
