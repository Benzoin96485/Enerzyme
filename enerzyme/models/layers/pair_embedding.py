"""Two-body chemical × radial embedding.

Center and neighbor species are embedded separately, concatenated, and
multiplied by a bias-free projection of an existing radial basis. The radial
basis itself stays in ``BesselRBF`` (Allegro uses ``flavor: nequip``). This
layer is the pair analogue of ``RandomAtomEmbedding``: any later Core can
read ``edge_embedding`` without owning the species tables.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor
from torch.nn import Embedding

from ..blocks.mlp import VariancePreservingMLP
from . import BaseFFLayer


class PairChemicalRadialLayer(BaseFFLayer):
    """``edge_embedding = cat(e_i, e_j) * Linear(rbf)``.

    ``idx_i_sr`` is the center and ``idx_j_sr`` is the neighbor, matching
    ``vij_sr = R_j - R_i``. ``dim_edge`` must be even so the two species
    tables have the same width.
    """

    def __init__(self, max_Za: int, num_rbf: int, dim_edge: int) -> None:
        super().__init__(
            input_fields={"Za", "rbf", "idx_i_sr", "idx_j_sr"},
            output_fields={"edge_embedding"},
        )
        dim_edge = int(dim_edge)
        if dim_edge < 2 or dim_edge % 2 != 0:
            raise ValueError(f"dim_edge must be a positive even integer, got {dim_edge}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(max_Za) < 1:
            raise ValueError(f"max_Za must be >= 1, got {max_Za}")
        self.max_Za = int(max_Za)
        self.num_rbf = int(num_rbf)
        self.dim_edge = dim_edge
        half = dim_edge // 2
        self.center = Embedding(self.max_Za + 1, half)
        self.neighbor = Embedding(self.max_Za + 1, half)
        self.basis = VariancePreservingMLP(
            self.num_rbf,
            dim_edge,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
        )

    def get_output(
        self,
        Za: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        if rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf width {rbf.shape[-1]} != num_rbf {self.num_rbf}"
            )
        if idx_i_sr.shape != idx_j_sr.shape:
            raise ValueError("idx_i_sr and idx_j_sr must have the same shape")
        if rbf.shape[0] != idx_i_sr.shape[0]:
            raise ValueError(
                "rbf and the neighbor index must describe the same edges"
            )
        species = Za.long()
        if species.numel() and int(species.max()) > self.max_Za:
            raise ValueError(
                f"atomic number {int(species.max())} exceeds max_Za {self.max_Za}"
            )
        centers = idx_i_sr.long()
        neighbors = idx_j_sr.long()
        chemical = torch.cat(
            (self.center(species[centers]), self.neighbor(species[neighbors])),
            dim=-1,
        )
        radial = self.basis(rbf.to(dtype=chemical.dtype))
        return {"edge_embedding": chemical * radial}
