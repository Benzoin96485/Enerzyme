"""LEFTNet Core (Du et al., NeurIPS 2023, arXiv:2304.04757).

The Core reads an atom embedding and the LEFTNet radial basis and emits

* ``atom_feature`` — invariant features ``[N, F]``
* ``atom_vector_feature`` — Cartesian vectors ``[N, 3, F]``

The species table, the exponential radial basis, the energy head, and
``Fa = -∇E`` stay outside. Each block is local substructure encoding
(shared, computed once), a Cartesian equivariant message, and a frame
transition. ``fte_scalarization='l2'`` is the released reduction. ``'frame'``
scalarizes the vector state on the node frame from the paper (Eq. 10)
instead of replacing that projection by a norm.

Reference: https://github.com/yuanqidu/LeftNet (``model.py``). Checkpoints
from that repository do not load.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..mp.basis.envelope import cosine_transition
from ..layers import BaseFFCore
from ..local_frame import (
    CartesianMessage,
    FrameTransition,
    LocalSubstructure,
    RadialFilter,
    aggregate_neighbors,
    leftnet_edge_frame,
    leftnet_node_frame,
    require_matching_distance,
    require_same_graph,
    subtract_graph_mean,
)


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"alpha": 1.0, "trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {"num_layers": 4, "fte_scalarization": "l2"},
    },
    {
        "name": "SimpleReadout",
        "params": {"output_fields": ["Ea"], "head_type": "dense"},
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class LEFTNetCore(BaseFFCore):
    """Local-frame message passing. Property heads stay in the layer stack."""

    def __str__(self) -> str:
        return f"""
################################################################################
# LEFTNet (Du et al., NeurIPS 2023, arXiv:2304.04757)                          #
# fte={self.fte_scalarization:<5}  atom_feature [N, F], atom_vector_feature [N, 3, F] #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 32,
        cutoff_sr: float = 5.0,
        num_layers: int = 4,
        fte_scalarization: str = "l2",
        eps: float = 1e-10,
    ) -> None:
        if int(dim_embedding) < 4:
            raise ValueError(f"dim_embedding must be >= 4, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if float(cutoff_sr) <= 0.0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if fte_scalarization not in {"l2", "frame"}:
            raise ValueError(
                f"fte_scalarization must be 'l2' or 'frame', got {fte_scalarization!r}"
            )
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
                "Ra",
                "batch_seg",
                "cutoff_values_sr",
            },
            output_fields={"atom_feature", "atom_vector_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_layers = int(num_layers)
        self.fte_scalarization = str(fte_scalarization)
        self.eps = float(eps)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.vector_feature_layout = "cartesian_n3f"

        self.radial = RadialFilter(self.num_rbf, self.dim_embedding)
        self.lse = LocalSubstructure(
            self.dim_embedding,
            normalize=False,
            scale=False,
            endpoint="source_target",
        )
        self.messages = nn.ModuleList(
            [
                CartesianMessage(self.dim_embedding, self.num_rbf)
                for _ in range(self.num_layers)
            ]
        )
        self.transitions = nn.ModuleList(
            [
                FrameTransition(self.dim_embedding, self.fte_scalarization)
                for _ in range(self.num_layers)
            ]
        )

    def build(self, built_layers: List[nn.Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        Ra: Tensor,
        batch_seg: Optional[Tensor] = None,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        self._check(atom_embedding, rbf, idx_i_sr, idx_j_sr, vij_sr, Dij_sr, Ra)
        require_same_graph(batch_seg, idx_i_sr, idx_j_sr)
        require_matching_distance(vij_sr, Dij_sr)
        window = _window(Dij_sr, self.cutoff_sr, cutoff_values_sr, rbf.dtype)
        radial_hidden = self.radial(rbf, window)
        scalar = aggregate_neighbors(atom_embedding, radial_hidden, idx_i_sr, idx_j_sr)
        vector = scalar.new_zeros(scalar.shape[0], 3, self.dim_embedding)
        node_frame = None
        if idx_i_sr.numel():
            centered = subtract_graph_mean(Ra, batch_seg)
            frame, edge_unit = leftnet_edge_frame(
                centered, vij_sr, idx_i_sr, idx_j_sr, self.eps
            )
            if self.fte_scalarization == "frame":
                node_frame = leftnet_node_frame(centered, idx_i_sr, idx_j_sr, self.eps)
            weight = self.lse(
                scalar,
                edge_unit,
                frame,
                radial_hidden,
                rbf,
                window,
                idx_i_sr,
                idx_j_sr,
            )
        else:
            weight = edge_unit = None
            if self.fte_scalarization == "frame":
                centered = subtract_graph_mean(Ra, batch_seg)
                node_frame = leftnet_node_frame(centered, idx_i_sr, idx_j_sr, self.eps)

        for message, transition in zip(self.messages, self.transitions):
            if weight is not None:
                delta_s, delta_v = message(
                    scalar, vector, rbf, weight, edge_unit, idx_i_sr, idx_j_sr
                )
                scalar = scalar + delta_s
                vector = vector + delta_v
            delta_s, delta_v = transition(scalar, vector, node_frame)
            scalar = scalar + delta_s
            vector = vector + delta_v
        return {"atom_feature": scalar, "atom_vector_feature": vector}

    def _check(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        vij: Tensor,
        dij: Tensor,
        pos: Tensor,
    ) -> None:
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}")
        if idx_i.shape != idx_j.shape or idx_i.ndim != 1:
            raise ValueError("idx_i_sr and idx_j_sr must be 1D and the same length")
        if vij.shape[0] != idx_i.shape[0] or dij.reshape(-1).shape[0] != idx_i.shape[0]:
            raise ValueError("vij_sr, Dij_sr, and the neighbor index must have one row per edge")
        if pos.shape[0] != atom_embedding.shape[0] or pos.shape[-1] != 3:
            raise ValueError("Ra must have shape [N, 3] with the same N as atom_embedding")


def _window(
    dij: Tensor,
    cutoff: float,
    given: Optional[Tensor],
    dtype: torch.dtype,
) -> Tensor:
    if given is None:
        values = cosine_transition(dij.reshape(-1), cutoff)
    else:
        values = given.reshape(-1)
    return values.to(dtype=dtype)
