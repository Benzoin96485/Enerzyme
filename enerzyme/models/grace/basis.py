"""One-particle GRACE basis.

The scalar-indicator basis contracts a neighbor's chemical embedding into the
radial channel and multiplies by real spherical harmonics. The equivariant
indicator basis is the same radial-angular factor, Clebsch–Gordan coupled to
the neighbor's message from the previous layer. Both are summed onto the
central atom and scaled by ``1 / avg_n_neigh``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .coupling import DegreeMix, SparseCGProduct
from .features import scatter_blocks, scale_features, zeros_like_degrees


def _scale_harmonics(radial: Tensor, harmonics: Sequence[Tensor]) -> List[Tensor]:
    """``R_{n l}(r) Y_{lm}`` with a radial channel on every degree."""
    scaled: List[Tensor] = []
    for degree, block in enumerate(harmonics):
        scaled.append(radial[:, :, degree].unsqueeze(-1) * block.unsqueeze(1))
    return scaled


class AtomicBasis(Module):
    """One ACE layer's atomic basis, before the many-body products."""

    def __init__(
        self,
        n_rad: int,
        lmax: int,
        embedding_dim: int,
        *,
        indicator_lmax: Optional[int] = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_rad = int(n_rad)
        self.lmax = int(lmax)
        self.embedding_dim = int(embedding_dim)
        self.indicator_lmax = (
            None if indicator_lmax is None else int(indicator_lmax)
        )
        self.embed = Linear(self.embedding_dim, self.n_rad, bias=False)
        self.edge_product: Optional[SparseCGProduct] = None
        self.combine: Optional[DegreeMix] = None
        if self.indicator_lmax is not None:
            self.edge_product = SparseCGProduct(
                self.n_rad, self.lmax, self.indicator_lmax, self.lmax
            )
            self.combine = DegreeMix(
                self.lmax,
                2 * self.n_rad,
                self.n_rad,
                normalize=False,
                epsilon=epsilon,
            )

    def _scalar_edges(
        self, radial: Tensor, harmonics: Sequence[Tensor], neighbor_embedding: Tensor
    ) -> List[Tensor]:
        weights = self.embed(neighbor_embedding)
        edges: List[Tensor] = []
        for degree, block in enumerate(harmonics):
            edges.append(
                radial[:, :, degree].unsqueeze(-1)
                * weights.unsqueeze(-1)
                * block.unsqueeze(1)
            )
        return edges

    def forward(
        self,
        radial: Tensor,
        harmonics: Sequence[Tensor],
        embedding: Tensor,
        receivers: Tensor,
        senders: Tensor,
        n_atoms: int,
        neighbor_scale: Tensor,
        messages: Optional[Sequence[Tensor]] = None,
    ) -> List[Tensor]:
        """Build the atomic basis on ``receivers`` from neighbors ``senders``.

        ``embedding`` and ``messages`` are per-atom. The scalar indicator is
        the neighbor embedding; an equivariant message, when present, is also
        read from the neighbor.
        """
        if radial.shape[0] == 0:
            return zeros_like_degrees(neighbor_scale, n_atoms, self.n_rad, self.lmax)
        scalar = scatter_blocks(
            self._scalar_edges(radial, harmonics, embedding[senders]),
            receivers,
            n_atoms,
        )
        scalar = scale_features(scalar, neighbor_scale)
        if self.edge_product is None:
            return scalar
        if messages is None:
            raise RuntimeError("this GRACE layer expects an equivariant indicator")
        gathered = [block[senders] for block in messages]
        coupled = self.edge_product(_scale_harmonics(radial, harmonics), gathered)
        coupled = scale_features(
            scatter_blocks(coupled, receivers, n_atoms), neighbor_scale
        )
        combined = [
            torch.cat((scalar[degree], coupled[degree]), dim=1)
            for degree in range(self.lmax + 1)
        ]
        return self.combine(combined)
