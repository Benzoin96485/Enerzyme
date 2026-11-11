"""GRACE Core (Lysogorskiy, Bochkarev, Drautz, 2025).

The Core maps a chemical embedding and a short-range graph to invariant atomic
densities. One layer is an ACE star graph. Further layers pass an equivariant
indicator to the neighbors and repeat the product recursion (semi-local tree
graphs). The embedding table, the energy head, and pair priors such as ZBL
stay in the layer stack.

This is an original implementation of the published basis. TensorPotential is
Academic Software License and is not imported or vendored; checkpoints from
that package are not weight-compatible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import BaseFFCore
from ..so3.spherical_harmonics import spherical_harmonics
from .basis import AtomicBasis
from .coupling import (
    DensityProjection,
    EquivariantIndicator,
    OrderSpec,
    ProductLadder,
)
from .features import split_spherical_harmonics
from .presets import resolve_grace_preset
from .radial import RadialEmbedding


DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "dim_embedding": 128,
    "cutoff_sr": 6.0,
    "cutoff_fn": "cosine",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"preset": "1L"}},
    {"name": "GRACEEnergyReadout"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class _GRACELayer(Module):
    """Radial embedding, atomic basis, and product ladder for one message layer.

    ``incoming_lmax`` is the angular momentum of the neighbor message this
    layer consumes. ``outgoing_lmax`` is the message it emits for the next
    layer; ``None`` means this is the last layer.
    """

    def __init__(
        self,
        orders: Sequence[OrderSpec],
        *,
        n_basis: int,
        cutoff: float,
        cutoff_order: int,
        basis: str,
        radial_hidden: int,
        radial_activation: str,
        embedding_dim: int,
        incoming_lmax: Optional[int],
        outgoing_lmax: Optional[int],
        indicator_channels: int,
        indicator_out: Optional[int],
        n_species: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.orders = list(orders)
        self.lmax = int(orders[0].lmax)
        self.n_rad = int(orders[0].channels)
        self.incoming_lmax = None if incoming_lmax is None else int(incoming_lmax)
        self.outgoing_lmax = None if outgoing_lmax is None else int(outgoing_lmax)
        self.radial = RadialEmbedding(
            n_basis,
            self.n_rad,
            self.lmax,
            cutoff,
            cutoff_order,
            basis=basis,
            hidden=radial_hidden,
            activation=radial_activation,
        )
        self.atomic = AtomicBasis(
            self.n_rad,
            self.lmax,
            embedding_dim,
            indicator_lmax=self.incoming_lmax,
            epsilon=epsilon,
        )
        self.ladder = ProductLadder(self.orders, epsilon=epsilon)
        self.indicator: Optional[EquivariantIndicator] = None
        if indicator_out is not None:
            if self.outgoing_lmax is None:
                raise ValueError("a message-passing layer needs outgoing indicator_lmax")
            self.indicator = EquivariantIndicator(
                [order.channels for order in self.orders],
                [order.lmax for order in self.orders],
                self.outgoing_lmax,
                indicator_channels,
                int(indicator_out),
                n_species,
            )

    def forward(
        self,
        distance: Tensor,
        harmonics: List[Tensor],
        embedding: Tensor,
        receivers: Tensor,
        senders: Tensor,
        species: Tensor,
        n_atoms: int,
        neighbor_scale: Tensor,
        messages: Optional[Sequence[Tensor]],
    ):
        radial = self.radial(distance)
        atomic = self.atomic(
            radial,
            harmonics,
            embedding,
            receivers,
            senders,
            n_atoms,
            neighbor_scale,
            messages,
        )
        orders = self.ladder(atomic)
        outgoing = None
        if self.indicator is not None:
            outgoing = self.indicator(orders, species)
        return orders, outgoing


class GRACECore(BaseFFCore):
    """Recursive ACE on a chemical embedding.

    Emits ``atom_feature`` of shape ``[N, n_densities]`` with
    ``feature_irreps = "{n_densities}x0e"``. For the foundation head the first
    channel is the linear density and the rest enter the MLP. For the
    Finnis–Sinclair head there is one density per published power.
    """

    def __str__(self) -> str:
        return """
################################################################################
# GRACE (Lysogorskiy, Bochkarev, Drautz, 2025)                                 #
# Recursive ACE densities from a chemical embedding; head stays outside        #
################################################################################
"""

    def __init__(
        self,
        max_Za: int = 94,
        dim_embedding: int = 128,
        cutoff_sr: float = 6.0,
        preset: Optional[str] = "1L",
        avg_n_neigh: float = 1.0,
        rms_epsilon: float = 1e-6,
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
        name, config = resolve_grace_preset(preset)
        if dim_embedding < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if cutoff_sr <= 0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if avg_n_neigh <= 0:
            raise ValueError(f"avg_n_neigh must be positive, got {avg_n_neigh}")
        self.preset = name
        self.max_Za = int(max_Za)
        self.dim_embedding = int(dim_embedding)
        self.cutoff_sr = float(cutoff_sr)
        self.basis_type = str(config["basis"])
        self.n_rad_base = int(config["n_rad_base"])
        self.cutoff_order = int(config["cutoff_order"])
        self.radial_hidden = int(config["radial_hidden"])
        self.radial_activation = str(config["radial_activation"])
        self.readout_head = str(config["readout"])
        self.readout_hidden = int(config["readout_hidden"])
        self.readout_activation = str(config["radial_activation"])
        self.recommended_embedding = int(config["embedding_size"])
        self.recommended_cutoff = float(config["cutoff"])
        self.fs_terms = config["fs_terms"]
        self.rms_epsilon = float(rms_epsilon)
        self.is_grace_core = True
        layer_specs = list(config["layers"])
        if self.readout_head == "linear_mlp":
            self.n_densities = int(config["n_mlp_dens"]) + 1
        elif self.readout_head == "finnis_sinclair":
            self.n_densities = len(self.fs_terms)
        else:
            raise ValueError(f"Unknown readout {self.readout_head!r} in preset {name}")
        if self.n_densities < 1:
            raise ValueError("GRACE must emit at least one density")
        self.num_layers = len(layer_specs)
        self.order_channels = [
            [order.channels for order in spec["orders"]] for spec in layer_specs
        ]
        self.order_lmax = [
            [order.lmax for order in spec["orders"]] for spec in layer_specs
        ]
        self.indicator_lmax = [spec["indicator_lmax"] for spec in layer_specs]
        self.dim_feature_out = self.n_densities
        self.feature_irreps = f"{self.n_densities}x0e"
        self.num_blocks = 1
        self.register_buffer("avg_n_neigh", torch.tensor(float(avg_n_neigh)))

        built: List[_GRACELayer] = []
        for index, spec in enumerate(layer_specs):
            incoming = None
            if index > 0:
                incoming = layer_specs[index - 1]["indicator_lmax"]
                if incoming is None:
                    raise ValueError(
                        f"GRACE layer {index - 1} must set indicator_lmax to feed the next layer"
                    )
                incoming = int(incoming)
            outgoing_channels = None
            outgoing_lmax = None
            if index + 1 < len(layer_specs):
                if spec["indicator_lmax"] is None:
                    raise ValueError(
                        f"GRACE layer {index} must set indicator_lmax to feed the next layer"
                    )
                outgoing_lmax = int(spec["indicator_lmax"])
                outgoing_channels = int(layer_specs[index + 1]["orders"][0].channels)
            built.append(
                _GRACELayer(
                    spec["orders"],
                    n_basis=self.n_rad_base,
                    cutoff=self.cutoff_sr,
                    cutoff_order=self.cutoff_order,
                    basis=self.basis_type,
                    radial_hidden=self.radial_hidden,
                    radial_activation=self.radial_activation,
                    embedding_dim=self.dim_embedding,
                    incoming_lmax=incoming,
                    outgoing_lmax=outgoing_lmax,
                    indicator_channels=int(spec["indicator_channels"]),
                    indicator_out=outgoing_channels,
                    n_species=self.max_Za + 1,
                    epsilon=self.rms_epsilon,
                )
            )
        self.layers = ModuleList(built)
        flat_channels = [
            channels for layer_channels in self.order_channels for channels in layer_channels
        ]
        self.density = DensityProjection(
            flat_channels, self.n_densities, self.max_Za + 1
        )
        self.max_l = max(layer.lmax for layer in self.layers)

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        n_atoms = int(atom_embedding.shape[0])
        species = Za.long()
        receivers = idx_i_sr.long()
        senders = idx_j_sr.long()
        distance = Dij_sr.to(dtype=atom_embedding.dtype)
        vectors = vij_sr.to(dtype=atom_embedding.dtype)
        if distance.numel() > 0:
            packed = spherical_harmonics(
                vectors,
                list(range(self.max_l + 1)),
                layout="e3nn",
                normalization="component",
            )
        else:
            packed = atom_embedding.new_zeros((0, (self.max_l + 1) ** 2))
        neighbor_scale = (1.0 / self.avg_n_neigh).to(dtype=atom_embedding.dtype)
        messages = None
        collected = []
        for layer in self.layers:
            width = (layer.lmax + 1) ** 2
            harmonics = split_spherical_harmonics(packed[:, :width], layer.lmax)
            orders, messages = layer(
                distance,
                harmonics,
                atom_embedding,
                receivers,
                senders,
                species,
                n_atoms,
                neighbor_scale,
                messages,
            )
            collected.append(orders)
        return {"atom_feature": self.density(collected, species)}
