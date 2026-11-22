"""SpinConv Core (Shuaibi et al., arXiv:2106.09575).

The Core maps a species embedding and a Gaussian bond basis to a scalar
atomic state. Edge messages are initialized from the distance block, then
updated by spin convolutions on the source-centered sphere. The species
table and the Gaussian centers stay outside. The last linear of the energy
embedding block is :class:`~enerzyme.models.layers.readout.SimpleReadout`.
Conservative forces are ``Fa = -∇E`` from the shared ``Force`` layer.

The released force-centric head paints three axis-aligned spheres and
averages random rotations. It is a direct force model and is not part of
this potential.

``idx_i_sr`` is the destination and ``idx_j_sr`` is the source.
``vij_sr = R[source] - R[destination]``.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList

from ..functional import segment_sum_coo
from ..layers._base_layer import BaseFFCore
from ..layers.atom_embedding import BaseAtomEmbedding
from ..layers.rbf import GaussianRBFLayer, gaussian_expansion, spinconv_gaussian_terms
from .geometry import symmetric_neighbor_mask
from .interaction import EmbeddingBlock, MessageBlock, PairDistanceScale, _swish


def _positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


class SpinConvCore(BaseFFCore):
    """Rotation-invariant edge messages with a pooled spin convolution.

    Emits ``atom_feature`` with shape ``(N, mid_hidden_channels)`` and
    ``feature_irreps = "{D}x0e"``. ``D`` is the width inside the embedding
    block. An isolated atom still runs the energy mixture on a zero
    aggregate, so its latent depends on the species embedding and not on
    coordinates.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        cutoff_sr: float,
        max_Za: int,
        hidden_channels: int = 32,
        mid_hidden_channels: int = 256,
        num_interactions: int = 3,
        sphere_size_lat: int = 16,
        sphere_size_long: int = 12,
        max_num_neighbors: int = 40,
        embedding_size: Optional[int] = None,
        num_embedding_basis: int = 8,
        group_channels: int = 8,
        scale_distances: bool = True,
        distance_scale_max: float = 2.0,
        basis_width_scalar: float = 3.0,
        residual_first: bool = False,
    ) -> None:
        dim_embedding = _positive("dim_embedding", dim_embedding)
        num_rbf = _positive("num_rbf", num_rbf)
        hidden_channels = _positive("hidden_channels", hidden_channels)
        mid_hidden_channels = _positive("mid_hidden_channels", mid_hidden_channels)
        num_interactions = _positive("num_interactions", num_interactions)
        sphere_size_lat = _positive("sphere_size_lat", sphere_size_lat)
        sphere_size_long = _positive("sphere_size_long", sphere_size_long)
        max_num_neighbors = _positive("max_num_neighbors", max_num_neighbors)
        num_embedding_basis = _positive("num_embedding_basis", num_embedding_basis)
        group_channels = _positive("group_channels", group_channels)
        max_Za = _positive("max_Za", max_Za)
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if sphere_size_lat < 2 or sphere_size_long < 2:
            raise ValueError(
                "sphere_size_lat and sphere_size_long must be >= 2, got "
                f"{sphere_size_lat} x {sphere_size_long}"
            )
        if hidden_channels % group_channels != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"group_channels ({group_channels})"
            )
        groups = hidden_channels // group_channels
        if mid_hidden_channels % groups != 0:
            raise ValueError(
                f"mid_hidden_channels ({mid_hidden_channels}) must be divisible by "
                f"hidden_channels / group_channels ({groups})"
            )
        if float(basis_width_scalar) <= 0:
            raise ValueError(f"basis_width_scalar must be positive, got {basis_width_scalar}")
        embedding_size = dim_embedding if embedding_size is None else _positive(
            "embedding_size", embedding_size
        )

        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
                "Za",
            },
            output_fields={"atom_feature"},
        )
        self.dim_embedding = dim_embedding
        self.num_rbf = num_rbf
        self.cutoff_sr = float(cutoff_sr)
        self.max_Za = max_Za
        self.hidden_channels = hidden_channels
        self.mid_hidden_channels = mid_hidden_channels
        self.num_interactions = num_interactions
        self.sphere_size_lat = sphere_size_lat
        self.sphere_size_long = sphere_size_long
        self.max_num_neighbors = max_num_neighbors
        self.embedding_size = embedding_size
        self.num_embedding_basis = num_embedding_basis
        self.group_channels = group_channels
        self.scale_distances = bool(scale_distances)
        self.distance_scale_max = float(distance_scale_max)
        self.basis_width_scalar = float(basis_width_scalar)
        self.residual_first = bool(residual_first)
        self.message_divisor = float(max_num_neighbors) / 2.0 + 1.0
        self.dim_feature_out = mid_hidden_channels
        self.feature_irreps = f"{mid_hidden_channels}x0e"
        self.num_output_blocks = 1

        center, coeff = spinconv_gaussian_terms(
            self.cutoff_sr, self.num_rbf, self.basis_width_scalar
        )
        self.register_buffer("gaussian_center", center)
        self.register_buffer("gaussian_coeff", coeff)

        self.act = _swish()
        self.distance_map = Linear(num_rbf, mid_hidden_channels)
        self.init_fc1 = Linear(mid_hidden_channels, mid_hidden_channels)
        self.init_fc2 = Linear(mid_hidden_channels, mid_hidden_channels)
        mix_kwargs = dict(
            dim_embedding=dim_embedding,
            embedding_size=embedding_size,
            num_basis=num_embedding_basis,
        )
        self.init_mix = EmbeddingBlock(
            mid_hidden_channels,
            hidden_channels,
            mid_hidden_channels,
            **mix_kwargs,
        )
        self.interactions = ModuleList(
            [
                MessageBlock(
                    hidden_channels,
                    mid_hidden_channels,
                    dim_embedding,
                    embedding_size,
                    num_embedding_basis,
                    sphere_size_lat,
                    sphere_size_long,
                    group_channels,
                    self.cutoff_sr,
                )
                for _ in range(num_interactions)
            ]
        )
        self.energy_mix = EmbeddingBlock(
            hidden_channels,
            mid_hidden_channels,
            mid_hidden_channels,
            project=False,
            **mix_kwargs,
        )
        self.pair_scale = (
            PairDistanceScale(max_Za, self.distance_scale_max) if self.scale_distances else None
        )

    def __str__(self) -> str:
        return """
##########################################################################
# SpinConv (arXiv:2106.09575) — spin-pooled edge messages                #
# Species embedding and the Gaussian basis stay outside this Core.      #
##########################################################################
"""

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        from ..layers.readout import HierachicalNSEReadout, HierachicalReadout

        if any(isinstance(layer, (HierachicalReadout, HierachicalNSEReadout)) for layer in built_layers):
            raise ValueError(
                "SpinConv atom_feature has shape (N, mid_hidden_channels). "
                "Use SimpleReadout. HierachicalReadout expects (N, F, blocks)."
            )
        gaussians: list[GaussianRBFLayer] = []
        for layer in built_layers:
            if isinstance(layer, BaseAtomEmbedding):
                self.atom_embedding = layer
            elif isinstance(layer, GaussianRBFLayer) and getattr(layer, "flavor", None) == "spinconv":
                gaussians.append(layer)
        if len(gaussians) != 1:
            raise ValueError(
                "SpinConvCore requires exactly one GaussianRBF layer with flavor "
                f"'spinconv', found {len(gaussians)}"
            )
        radial = gaussians[0]
        if radial.apply_cutoff_fn:
            raise ValueError(
                "SpinConv GaussianRBF must set apply_cutoff_fn: false. "
                "The published expansion has no envelope."
            )
        if int(radial.num_rbf) != self.num_rbf:
            raise ValueError(
                f"GaussianRBF num_rbf ({radial.num_rbf}) != Core num_rbf ({self.num_rbf})"
            )
        if abs(float(radial.cutoff_sr) - self.cutoff_sr) > 1e-4:
            raise ValueError(
                f"GaussianRBF cutoff_sr ({radial.cutoff_sr}) != Core cutoff_sr ({self.cutoff_sr})"
            )
        scalar = getattr(radial, "basis_width_scalar", None)
        if scalar is None or abs(float(scalar) - self.basis_width_scalar) > 1e-6:
            raise ValueError(
                "GaussianRBF basis_width_scalar "
                f"({scalar}) must equal Core basis_width_scalar ({self.basis_width_scalar})"
            )
        if not torch.allclose(radial.center, self.gaussian_center):
            raise ValueError("GaussianRBF centers do not match the SpinConv distance basis")
        self.radial_basis_function = radial
        embedding = getattr(self, "atom_embedding", None)
        if embedding is None:
            raise ValueError("SpinConvCore requires a RandomAtomEmbedding layer")
        if int(embedding.dim_embedding) != self.dim_embedding:
            raise ValueError(
                f"atom embedding width {embedding.dim_embedding} != dim_embedding {self.dim_embedding}"
            )
        if int(embedding.max_Za) != self.max_Za:
            raise ValueError(
                f"atom embedding max_Za {embedding.max_Za} != Core max_Za {self.max_Za}"
            )

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        Za: Tensor,
    ) -> dict[str, Tensor]:
        self._check_inputs(atom_embedding, rbf, Dij_sr, vij_sr, idx_i_sr, idx_j_sr, Za)
        num_nodes = int(atom_embedding.shape[0])
        idx_i = idx_i_sr.long()
        idx_j = idx_j_sr.long()
        distance = Dij_sr
        vij = vij_sr
        basis = rbf
        if idx_i.numel():
            keep = symmetric_neighbor_mask(
                idx_i, idx_j, distance, num_nodes, self.max_num_neighbors
            )
            if not bool(keep.all()):
                idx_i = idx_i[keep]
                idx_j = idx_j[keep]
                distance = distance[keep]
                vij = vij[keep]
                basis = basis[keep]

        if self.pair_scale is not None:
            scaled = self.pair_scale(distance, Za[idx_j], Za[idx_i])
            basis = gaussian_expansion(scaled, self.gaussian_center, self.gaussian_coeff)
        distance_embed = self.distance_map(basis)
        hidden = self.act(self.init_fc2(self.act(self.init_fc1(distance_embed))))
        messages = self.init_mix(hidden, atom_embedding, idx_j, idx_i)
        for index, block in enumerate(self.interactions):
            updated = block(
                messages,
                distance_embed,
                atom_embedding,
                idx_i,
                idx_j,
                vij,
                distance,
                num_nodes,
            )
            if index == 0 and not self.residual_first:
                messages = updated
            else:
                messages = messages + updated

        if num_nodes == 0:
            aggregated = atom_embedding.new_zeros((0, self.hidden_channels))
        else:
            aggregated = segment_sum_coo(messages, idx_i, dim_size=num_nodes)
            aggregated = aggregated / self.message_divisor
        nodes = torch.arange(num_nodes, device=atom_embedding.device)
        atom_feature = self.energy_mix(aggregated, atom_embedding, nodes, nodes)
        return {"atom_feature": atom_feature}

    def _check_inputs(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        Za: Tensor,
    ) -> None:
        num_nodes = int(atom_embedding.shape[0])
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must have shape (N, {self.dim_embedding}), "
                f"got {tuple(atom_embedding.shape)}"
            )
        if Za.shape != (num_nodes,):
            raise ValueError(f"Za must have shape ({num_nodes},), got {tuple(Za.shape)}")
        edges = int(idx_i_sr.shape[0]) if idx_i_sr.ndim == 1 else -1
        aligned = (
            idx_i_sr.ndim == 1
            and idx_j_sr.shape == idx_i_sr.shape
            and Dij_sr.shape == idx_i_sr.shape
            and vij_sr.shape == (edges, 3)
            and rbf.shape == (edges, self.num_rbf)
        )
        if not aligned:
            raise ValueError(
                "rbf, Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same "
                f"edges (rbf width {self.num_rbf}); got rbf {tuple(rbf.shape)}, "
                f"Dij_sr {tuple(Dij_sr.shape)}, vij_sr {tuple(vij_sr.shape)}, "
                f"idx_i_sr {tuple(idx_i_sr.shape)}, idx_j_sr {tuple(idx_j_sr.shape)}"
            )
        if edges and (
            int(idx_i_sr.min()) < 0
            or int(idx_j_sr.min()) < 0
            or int(idx_i_sr.max()) >= num_nodes
            or int(idx_j_sr.max()) >= num_nodes
        ):
            raise ValueError("edge indices are outside the atom range")
