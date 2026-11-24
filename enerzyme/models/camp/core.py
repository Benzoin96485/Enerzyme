"""CAMP Core (Wen, Huang, Dai, and Adhikari, npj Comput. Mater. 2025).

The Core maps a chemical embedding and a short-range graph to scalar
features from every message-passing layer. Each layer builds Cartesian
atomic moments, contracts them into hyper moments, and writes the result
back onto the node state. The species embedding, the per-layer energy head,
and elemental references stay outside the Core.

``idx_i_sr`` is the receiver. ``idx_j_sr`` is the sender. ``vij_sr`` is
``r_sender - r_receiver``, the paper's ``r_ij``.

This is an original implementation of the published Cartesian moments.
`wengroup/camp <https://github.com/wengroup/camp>`_ is not imported, and its
checkpoints are not weight-compatible. Each layer owns a species-pair
Chebyshev basis, matching the reference module layout. ``dim_embedding`` is
the channel count, one more than the reference ``max_u``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from ..layers._base_layer import BaseFFCore
from .moment import CAMPLayer
from .presets import resolve_camp_preset
from .radial import resolve_hidden_layers, species_lookup


_MAX_RANK = 6


DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 33,
    "cutoff_sr": 5.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "water"}},
    {"name": "CAMPEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class CAMPCore(BaseFFCore):
    """Cartesian atomic-moment message passing on an external embedding.

    Emits ``atom_feature`` of shape ``[N, C, T]``. ``C`` is ``dim_embedding``
    and ``T`` is the number of layers. ``feature_irreps`` describes one layer
    of scalars. ``num_output_blocks`` is ``T``, so a hierarchical head can
    read every layer and :class:`~enerzyme.models.layers.readout.SimpleReadout`
    reads the last one.
    """

    def __str__(self) -> str:
        return """
################################################################################
# CAMP (Wen et al., 2025)                                                      #
# Cartesian atomic moments from a chemical embedding; head stays outside      #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 33,
        max_Za: int = 94,
        cutoff_sr: float = 5.0,
        preset: Optional[str] = "water",
        max_v: Optional[int] = None,
        num_layers: Optional[int] = None,
        max_chebyshev_degree: Optional[int] = None,
        envelope_p: Optional[int] = None,
        radial_mlp_hidden_layers: Optional[Sequence[int]] = None,
        avg_num_neighbors: Optional[float] = None,
        output_mlp_hidden_layers: Optional[Sequence[int]] = None,
        atomic_numbers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "Dij_sr",
                "vij_sr",
                "idx_i_sr",
                "idx_j_sr",
            },
            output_fields={"atom_feature"},
        )
        spec = resolve_camp_preset(
            preset,
            max_v=max_v,
            num_layers=num_layers,
            max_chebyshev_degree=max_chebyshev_degree,
            envelope_p=envelope_p,
            radial_mlp_hidden_layers=radial_mlp_hidden_layers,
            avg_num_neighbors=avg_num_neighbors,
            output_mlp_hidden_layers=output_mlp_hidden_layers,
        )
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if int(spec["max_v"]) < 0 or int(spec["max_v"]) > _MAX_RANK:
            raise ValueError(f"max_v must lie in 0..{_MAX_RANK}, got {spec['max_v']}")
        if int(spec["num_layers"]) < 1:
            raise ValueError(f"num_layers must be >= 1, got {spec['num_layers']}")
        if int(spec["max_chebyshev_degree"]) < 0:
            raise ValueError(
                f"max_chebyshev_degree must be >= 0, got {spec['max_chebyshev_degree']}"
            )
        if int(spec["envelope_p"]) < 1:
            raise ValueError(f"envelope_p must be >= 1, got {spec['envelope_p']}")
        if float(spec["avg_num_neighbors"]) <= 0:
            raise ValueError(
                f"avg_num_neighbors must be positive, got {spec['avg_num_neighbors']}"
            )
        lookup, n_species = species_lookup(max_Za, atomic_numbers)
        self.register_buffer("index_of_z", lookup)
        self.preset = None if preset is None else str(preset)
        self.is_camp_core = True
        self.dim_embedding = int(dim_embedding)
        self.max_Za = int(max_Za)
        self.n_species = int(n_species)
        self.cutoff_sr = float(cutoff_sr)
        self.max_v = int(spec["max_v"])
        self.num_layers = int(spec["num_layers"])
        self.max_chebyshev_degree = int(spec["max_chebyshev_degree"])
        self.envelope_p = int(spec["envelope_p"])
        self.avg_num_neighbors = float(spec["avg_num_neighbors"])
        self.output_mlp_hidden_layers = spec["output_mlp_hidden_layers"]
        self.atomic_numbers = None if atomic_numbers is None else [int(z) for z in atomic_numbers]
        hidden = resolve_hidden_layers(spec["radial_mlp_hidden_layers"], self.dim_embedding)
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            first = index == 0
            last = index == self.num_layers - 1
            self.layers.append(
                CAMPLayer(
                    n_channels=self.dim_embedding,
                    n_species=self.n_species,
                    max_v=self.max_v,
                    max_in_rank=0 if first else self.max_v,
                    max_out_rank=0 if last else self.max_v,
                    residual=not first,
                    avg_num_neighbors=self.avg_num_neighbors,
                    degree=self.max_chebyshev_degree,
                    cutoff=self.cutoff_sr,
                    envelope_p=self.envelope_p,
                    radial_hidden=hidden,
                )
            )
        self.dim_feature_out = self.dim_embedding
        self.num_output_blocks = self.num_layers
        self.feature_irreps = f"{self.dim_embedding}x0e"

    def build(self, built_layers) -> None:
        self.adopt_radius_stack(built_layers)

    def _species_index(self, atomic_numbers: Tensor) -> Tensor:
        species = atomic_numbers.long()
        if species.numel() and (
            int(species.min()) < 0 or int(species.max()) > self.max_Za
        ):
            raise ValueError(
                f"atomic numbers must lie in 0..max_Za ({self.max_Za}), "
                f"got {int(species.min())}..{int(species.max())}"
            )
        index = self.index_of_z[species.clamp(min=0)]
        if torch.any(index < 0):
            missing = species[index < 0].unique().tolist()
            raise ValueError(
                f"atomic numbers {missing} are outside the CAMP species table"
            )
        return index

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must be [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if Za.shape[0] != atom_embedding.shape[0]:
            raise ValueError(
                f"Za and atom_embedding must share the atom count, "
                f"got {Za.shape[0]} and {atom_embedding.shape[0]}"
            )
        n_atoms = int(atom_embedding.shape[0])
        n_edges = int(idx_i_sr.shape[0])
        if idx_j_sr.shape[0] != n_edges or Dij_sr.shape[0] != n_edges or vij_sr.shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {tuple(Dij_sr.shape)}, {tuple(vij_sr.shape)}, "
                f"{tuple(idx_i_sr.shape)}, {tuple(idx_j_sr.shape)}"
            )
        if n_edges and vij_sr.shape[-1] != 3:
            raise ValueError(f"vij_sr must be [E, 3], got {tuple(vij_sr.shape)}")
        species = self._species_index(Za)
        dtype = atom_embedding.dtype
        if n_edges == 0:
            distance = atom_embedding.new_zeros(0)
            vectors = atom_embedding.new_zeros(0, 3)
            receivers = idx_i_sr.new_zeros(0)
            senders = idx_j_sr.new_zeros(0)
            zi = species.new_zeros(0)
            zj = species.new_zeros(0)
        else:
            receivers = idx_i_sr.long()
            senders = idx_j_sr.long()
            if (
                int(receivers.min()) < 0
                or int(senders.min()) < 0
                or int(receivers.max()) >= n_atoms
                or int(senders.max()) >= n_atoms
            ):
                raise ValueError("neighbor indices exceed the atom count")
            distance = Dij_sr.reshape(-1).to(dtype=dtype)
            vectors = vij_sr.to(dtype=dtype)
            zi = species.index_select(0, receivers)
            zj = species.index_select(0, senders)
        features = {0: atom_embedding.transpose(0, 1).contiguous()}
        scalars = []
        for layer in self.layers:
            features = layer(
                distance, vectors, zi, zj, receivers, senders, features, n_atoms
            )
            scalars.append(features[0].transpose(0, 1))
        return {"atom_feature": torch.stack(scalars, dim=-1)}
