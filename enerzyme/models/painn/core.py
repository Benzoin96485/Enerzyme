"""PaiNN Core (Schütt, Unke, Gastegger, ICML 2021, arXiv:2102.03150).

Equivariant message passing in Cartesian space. The Core consumes a scalar
atom embedding and a radial basis and emits:

* ``atom_feature`` — invariant features ``[N, F]`` (``feature_irreps = "Fx0e"``)
* ``atom_vector_feature`` — equivariant features ``[N, 3, F]``

Nuclear embedding, Bessel expansion, cosine cutoff values, scalar readouts,
and tensor heads stay outside the Core. Filters are linear combinations of
the radial basis, multiplied by the cosine cutoff (paper Sec. IV.1).

Adapted from SchNetPack 2.0.4 ``schnetpack.representation.painn`` (MIT):
https://github.com/atomistic-machine-learning/schnetpack
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..cutoff import cosine_transition
from ..layers import BaseFFCore
from ..layers.gated_equivariant import cartesian_dense
from .interaction import PaiNNInteraction, PaiNNMixing


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 128,
    "num_rbf": 20,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "painn", "trainable": False, "apply_cutoff_fn": False},
    },
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "num_interactions": 3,
            "shared_interactions": False,
            "shared_filters": False,
        },
    },
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


class PaiNNCore(BaseFFCore):
    """Scalar/vector message-passing Core. Property heads stay in the layer stack."""

    def __str__(self) -> str:
        return """
################################################################################
# PaiNN (Schütt, Unke, Gastegger, ICML 2021, arXiv:2102.03150)                 #
# Cartesian equivariant MP; atom_feature [N, F], atom_vector_feature [N, 3, F] #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 128,
        num_rbf: int = 20,
        cutoff_sr: float = 5.0,
        num_interactions: int = 3,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        propagate_vectors: bool = True,
        use_vector_scalar_product: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if num_interactions < 1:
            raise ValueError(f"num_interactions must be >= 1, got {num_interactions}")
        if dim_embedding < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
                "cutoff_values_sr",
            },
            output_fields={"atom_feature", "atom_vector_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.cutoff_sr = float(cutoff_sr)
        self.num_interactions = int(num_interactions)
        self.shared_interactions = bool(shared_interactions)
        self.shared_filters = bool(shared_filters)
        self.propagate_vectors = bool(propagate_vectors)
        self.use_vector_scalar_product = bool(use_vector_scalar_product)
        self.epsilon = float(epsilon)

        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        # Channel-last Cartesian vectors, not an SH / e3nn layout.
        self.vector_feature_layout = "cartesian_n3f"

        filter_out = (
            3 * self.dim_embedding
            if self.shared_filters
            else self.num_interactions * 3 * self.dim_embedding
        )
        self.filter_net = cartesian_dense(self.num_rbf, filter_out, activation=False)

        n_blocks = 1 if self.shared_interactions else self.num_interactions
        self.interactions = ModuleList(
            [
                PaiNNInteraction(
                    self.dim_embedding, propagate_vectors=self.propagate_vectors
                )
                for _ in range(n_blocks)
            ]
        )
        self.mixings = ModuleList(
            [
                PaiNNMixing(
                    self.dim_embedding,
                    epsilon=self.epsilon,
                    use_vector_scalar_product=self.use_vector_scalar_product,
                )
                for _ in range(n_blocks)
            ]
        )

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def _filters(
        self,
        rbf: Tensor,
        dij: Tensor,
        cutoff_values_sr: Optional[Tensor],
    ) -> List[Tensor]:
        if cutoff_values_sr is None:
            cutoff_values_sr = cosine_transition(dij, self.cutoff_sr)
        window = cutoff_values_sr.reshape(-1, 1).to(dtype=rbf.dtype)
        filters = self.filter_net(rbf) * window
        width = 3 * self.dim_embedding
        if self.shared_filters:
            return [filters] * self.num_interactions
        return list(torch.split(filters, width, dim=-1))

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run ``num_interactions`` message/update pairs.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``. ``cutoff_values_sr`` is optional; when it is
        omitted the Core applies the Behler cosine window to ``Dij_sr``.
        """
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}"
            )
        q = atom_embedding
        mu = q.new_zeros(q.shape[0], 3, self.dim_embedding)
        idx_i = idx_i_sr.long()
        idx_j = idx_j_sr.long()
        vij = vij_sr.to(dtype=q.dtype)
        dij = Dij_sr.to(dtype=q.dtype)
        rbf = rbf.to(dtype=q.dtype)
        norm = torch.linalg.vector_norm(vij, dim=-1, keepdim=True).clamp_min(1e-12)
        dir_ij = vij / norm
        filter_list = self._filters(rbf, dij, cutoff_values_sr)
        n_atoms = q.shape[0]
        for step in range(self.num_interactions):
            block = 0 if self.shared_interactions else step
            q, mu = self.interactions[block](
                q, mu, filter_list[step], dir_ij, idx_i, idx_j, n_atoms
            )
            q, mu = self.mixings[block](q, mu)
        return {"atom_feature": q, "atom_vector_feature": mu}
