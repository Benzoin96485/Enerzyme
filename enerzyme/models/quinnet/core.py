"""QuinNet Core (Wang et al., NeurIPS 2023).

The Core is the neighbor embedding, the edge embedding, and the quintuple
interaction stack. It reads a species embedding and an exponential-normal
basis and emits:

* ``atom_feature`` — invariant features ``[N, F]``
* ``atom_vector_feature`` — Cartesian ``l = 1`` features ``[N, 3, F]``
* ``atom_l2_feature`` — polynomial ``l = 2`` features ``[N, 5, F]``

``ExpNormalSmearing``, the first
:class:`~enerzyme.models.layers.atom_embedding.RandomAtomEmbedding`, the
cosine window, and the scalar energy MLP stay outside. The second species
table is :class:`~enerzyme.models.et.interaction.ETNeighborEmbedding`. The
edge map is :class:`~enerzyme.models.visnet.interaction.EdgeEmbedding`. The
shared neighbor list has no self pair, so the Core inserts ``r = 0`` through
the same radial basis. The reference ``max_num_neighbors`` cap is not applied.

Inside a block the angular state has 8 channels, ``Sphere(l=2)`` without the
constant ``l = 0``. The public vector is the ``l = 1`` slice, so
:class:`~enerzyme.models.layers.tensorial.EquivariantDipoleLayer` still
applies. The reference scalar head ignores the vector; that head is
:class:`~enerzyme.models.layers.readout.SimpleReadout`.

Adapted from ``src/models/quinnet.py`` (MIT):
https://github.com/Zun-Wang/QuinNet
Checkpoints from that package do not load.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList

from ..mp.basis.envelope import cosine_transition
from ..et.interaction import ETNeighborEmbedding, normalize_distance_influence
from ..layers import BaseFFCore
from ..layers.atom_embedding import RandomAtomEmbedding
from ..layers.polynomial_harmonics import angular_dim, polynomial_spherical_harmonics
from ..layers.self_pairs import insert_missing_self_pairs, unit_direction
from ..visnet.interaction import EdgeEmbedding, normalize_activation
from .interaction import QuinNetInteraction


_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# example/hparams_aspirin.yaml. max_z: 100 there is Embedding(100), so
# Z = 0..99. This stack follows the other Cores: max_Za is the largest
# atomic number, and the table has max_Za + 1 rows.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 32,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_MD17_BUILD_PARAMS = dict(DEFAULT_BUILD_PARAMS)

PUBLISHED_MD17_CORE_PARAMS = {
    "num_layers": 5,
    "num_heads": 8,
    "distance_influence": "both",
    "neighbor_embedding": True,
    "include_self": True,
    "activation": "silu",
    "attn_activation": "silu",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": dict(PUBLISHED_MD17_CORE_PARAMS)},
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


class QuinNetCore(BaseFFCore):
    """Scalar/angular Core. Property heads stay in the layer stack."""

    def __str__(self) -> str:
        return """
################################################################################
# QuinNet (Wang et al., NeurIPS 2023)                                         #
# atom_feature [N, F], atom_vector_feature [N, 3, F]; l = 2 is atom_l2_feature #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 256,
        num_rbf: int = 32,
        max_Za: int = 100,
        cutoff_sr: float = 5.0,
        num_layers: int = 5,
        num_heads: int = 8,
        distance_influence: str = "both",
        neighbor_embedding: bool = True,
        include_self: bool = True,
        activation: object = "silu",
        attn_activation: object = "silu",
    ) -> None:
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if int(num_heads) < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if int(dim_embedding) % int(num_heads) != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if int(max_Za) < 1:
            raise ValueError(f"max_Za must be >= 1, got {max_Za}")
        if float(cutoff_sr) <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        influence = normalize_distance_influence(distance_influence)
        normalize_activation(activation)
        normalize_activation(attn_activation)
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
                "cutoff_values_sr",
                "Za",
            },
            output_fields={"atom_feature", "atom_vector_feature", "atom_l2_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.max_Za = int(max_Za)
        self.cutoff_sr = float(cutoff_sr)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.distance_influence = influence
        self.neighbor_embedding = bool(neighbor_embedding)
        self.include_self = bool(include_self)
        self.lmax = 2
        self.angular_dim = angular_dim(self.lmax)
        self.dim_feature_out = self.dim_embedding
        self.feature_irreps = f"{self.dim_feature_out}x0e"
        self.vector_feature_layout = "cartesian_n3f"
        self._radial_holder: list = []

        self.neighborhood: Optional[ETNeighborEmbedding]
        if self.neighbor_embedding:
            self.neighborhood = ETNeighborEmbedding(
                self.dim_embedding, self.num_rbf, self.max_Za
            )
        else:
            self.neighborhood = None
        self.edge_embedding = EdgeEmbedding(self.num_rbf, self.dim_embedding)
        self.quin_layers = ModuleList(
            [
                QuinNetInteraction(
                    self.dim_embedding,
                    self.num_heads,
                    self.distance_influence,
                    last_layer=(index == self.num_layers - 1),
                    activation=activation,
                    attn_activation=attn_activation,
                )
                for index in range(self.num_layers)
            ]
        )
        self.out_norm = LayerNorm(self.dim_embedding)

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)
        radial = None
        for layer in built_layers:
            if layer is self:
                break
            fields = getattr(layer, "_output_fields", ())
            if "rbf" in fields and callable(getattr(layer, "get_rbf", None)):
                radial = layer
            if isinstance(layer, RandomAtomEmbedding):
                if int(layer.dim_embedding) != self.dim_embedding:
                    raise ValueError(
                        "RandomAtomEmbedding width "
                        f"({layer.dim_embedding}) != dim_embedding ({self.dim_embedding})"
                    )
                if int(layer.max_Za) != self.max_Za:
                    raise ValueError(
                        f"RandomAtomEmbedding max_Za ({layer.max_Za}) != "
                        f"Core max_Za ({self.max_Za})"
                    )
        if radial is not None:
            radial_cutoff = getattr(radial, "cutoff_sr", None)
            if (
                radial_cutoff is not None
                and abs(float(radial_cutoff) - self.cutoff_sr) > 1e-5
            ):
                raise ValueError(
                    f"radial cutoff_sr ({radial_cutoff}) != Core cutoff_sr ({self.cutoff_sr})"
                )
            radial_width = getattr(radial, "num_rbf", None)
            if radial_width is not None and int(radial_width) != self.num_rbf:
                raise ValueError(
                    f"radial num_rbf ({radial_width}) != Core num_rbf ({self.num_rbf})"
                )
        self._radial_holder = [] if radial is None else [radial]

    def _window(self, dist: Tensor, cutoff_values_sr: Optional[Tensor]) -> Tensor:
        if cutoff_values_sr is None:
            return cosine_transition(dist, self.cutoff_sr).reshape(-1).to(dtype=dist.dtype)
        window = cutoff_values_sr.reshape(-1).to(dtype=dist.dtype)
        if window.shape[0] != dist.shape[0]:
            raise ValueError(
                "cutoff_values_sr and Dij_sr must describe the same edges, "
                f"got {window.shape[0]} and {dist.shape[0]}"
            )
        return window

    def _prepare_graph(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
        cutoff_values_sr: Optional[Tensor],
    ) -> tuple[int, Tensor, Tensor, Tensor, Tensor, Tensor]:
        n_atoms = int(atom_embedding.shape[0])
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must have shape [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must have shape [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        if idx_i_sr.shape != idx_j_sr.shape or idx_i_sr.ndim != 1:
            raise ValueError(
                "idx_i_sr and idx_j_sr must be 1D edge indices of the same length, "
                f"got {tuple(idx_i_sr.shape)} and {tuple(idx_j_sr.shape)}"
            )
        n_edges = int(idx_i_sr.shape[0])
        if tuple(vij_sr.shape) != (n_edges, 3):
            raise ValueError(
                f"vij_sr must have shape [{n_edges}, 3], got {tuple(vij_sr.shape)}"
            )
        if Dij_sr.reshape(-1).shape[0] != n_edges:
            raise ValueError(
                "Dij_sr, idx_i_sr, and idx_j_sr must share the edge count, "
                f"got {Dij_sr.reshape(-1).shape[0]} and {n_edges}"
            )
        if rbf.shape[0] != n_edges:
            raise ValueError(f"rbf has {rbf.shape[0]} rows for {n_edges} edges")

        dtype = atom_embedding.dtype
        dist = Dij_sr.reshape(-1).to(dtype=dtype)
        vij = vij_sr.to(dtype=dtype)
        basis = rbf.to(dtype=dtype)
        window = self._window(dist, cutoff_values_sr)
        radial = self._radial_holder[0] if self._radial_holder else None
        idx_i, idx_j, vij, dist, basis, window = insert_missing_self_pairs(
            n_atoms,
            idx_i_sr.long(),
            idx_j_sr.long(),
            vij,
            dist,
            basis,
            window,
            enabled=self.include_self,
            radial_basis=radial,
            num_rbf=self.num_rbf,
        )
        direction = polynomial_spherical_harmonics(
            unit_direction(vij, dist), self.lmax
        ).to(dtype=dtype)
        return n_atoms, idx_i, idx_j, direction, basis, window

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
    ) -> Dict[str, Tensor]:
        """Run the neighbor embedding and ``num_layers`` QuinNet blocks.

        Receivers are ``idx_i_sr`` and senders are ``idx_j_sr``.
        ``vij_sr = R_j - R_i``. ``Za`` is required when ``neighbor_embedding``
        is on. The radial basis is the windowed exponential-normal expansion;
        ``cutoff_values_sr`` is the cosine window on the attention score.
        """
        if self.neighbor_embedding and Za is None:
            raise ValueError("neighbor_embedding requires atomic numbers Za")
        n_atoms, idx_i, idx_j, direction, basis, window = self._prepare_graph(
            atom_embedding,
            rbf,
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            Dij_sr,
            cutoff_values_sr,
        )
        scalar = atom_embedding
        if self.neighborhood is not None:
            scalar = self.neighborhood(
                Za.long(), scalar, idx_i, idx_j, basis, window
            )
        vector = scalar.new_zeros(n_atoms, self.angular_dim, self.dim_embedding)
        edge = self.edge_embedding(idx_i, idx_j, basis, scalar)
        # The angular state starts at zero, so the six geometric channels are
        # zero in the first block. They enter once that block has written
        # directions into the state. ``num_layers`` must be at least 2 for
        # the quintuple terms to reach the scalar features.
        for layer in self.quin_layers:
            delta_scalar, delta_vector, delta_edge = layer(
                scalar, vector, edge, idx_i, idx_j, window, direction
            )
            scalar = scalar + delta_scalar
            vector = vector + delta_vector
            if delta_edge is not None:
                edge = edge + delta_edge
        scalar = self.out_norm(scalar)
        return {
            "atom_feature": scalar,
            "atom_vector_feature": vector[:, :3].contiguous(),
            "atom_l2_feature": vector[:, 3:].contiguous(),
        }
