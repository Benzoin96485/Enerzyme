"""Geodite Core (Reschützegger et al., J. Chem. Theory Comput. 2026).

The Core maps a species embedding and an effective radial basis to latent
scalar states. It does not own the Bessel expansion, the polynomial
envelope, the covalent distance weight, the energy head, or the ZBL wall.
Those stay in the layer stack so the same priors can be reused.

Geodite-MP, the published MPtrj model, is the default:

* 192 channels, 4 layers, ``lmax = 2``, 8 heads, dropout 0.2
* edge update ``linw_mlpa_ln``
* 12 Bessel functions and a 6 Å cutoff, applied outside this module

Each layer emits a scalar state. ``atom_feature`` stacks them as
``[N, C, K]``. ``isolated_atom_feature`` is the same stack with no neighbors,
so :class:`~enerzyme.models.layers.isolated.IsolatedAtomResidualLayer` can
form Eq. 1. ``atom_steerable_feature`` is the final equivariant state in the
Geodite polynomial basis, ``[N, (lmax+1)²-1, C]``, with ``l = 1`` stored as
``(x, y, z)``.

Adapted from IBM ``materials`` ``models/pos_egnn/mlip`` (Apache-2.0). This is
a reimplementation: those checkpoints do not load.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from ..layers import BaseFFCore
from .geometry import angular_dim, geodite_spherical_harmonics
from .interaction import (
    EdgeInit,
    EquivariantCoupling,
    InteractionBlock,
    NodeInit,
    parse_edge_updates,
)


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 192,
    "num_rbf": 12,
    "max_Za": 100,
    "cutoff_sr": 6.0,
    # eV, so the screened ZBL prefactor is 14.3996 eV·Å. Use 1.0 for Hartree.
    "Hartree_in_E": 27.211386245988,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"trainable": False, "apply_cutoff_fn": False},
    },
    {
        "name": "RadialWindow",
        "params": {
            "envelope": "polynomial",
            "envelope_p": 6,
            "attenuation": "covalent",
            "trainable_attenuation": True,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core"},
    {"name": "IsolatedAtomResidual"},
    {
        "name": "HierachicalReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "use_bias": False,
            "initial_weight": "xavier_uniform",
        },
    },
    {"name": "ScreenedZBLRepulsionEnergy", "params": {"envelope_p": 5}},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


class GeoditeCore(BaseFFCore):
    """Scalar/steerable message passing without Clebsch–Gordan products."""

    def __str__(self) -> str:
        return """
################################################################################
# Geodite (Reschützegger et al., J. Chem. Theory Comput. 2026)                 #
# SiLU attention + inner products; atom_feature [N, C, K]                      #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 192,
        num_rbf: int = 12,
        max_Za: int = 100,
        num_layers: int = 4,
        lmax: int = 2,
        num_heads: int = 8,
        attn_dropout: float = 0.2,
        edge_updates: Union[bool, str] = "linw_mlpa_ln",
        evec_dim: Optional[int] = None,
        emlp_dim: Optional[int] = None,
        edge_norm: str = "",
        epsilon: float = 1e-6,
    ) -> None:
        if int(num_layers) < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(num_rbf) < 1:
            raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
        if int(lmax) < 1 or int(lmax) > 4:
            raise ValueError(f"lmax must be in 1..4, got {lmax}")
        if int(num_heads) < 1 or int(dim_embedding) % int(num_heads) != 0:
            raise ValueError(
                f"dim_embedding {dim_embedding} must be divisible by num_heads {num_heads}"
            )
        if not 0.0 <= float(attn_dropout) <= 1.0:
            raise ValueError(f"attn_dropout must be in [0, 1], got {attn_dropout}")
        if float(epsilon) <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        super().__init__(
            input_fields={
                "atom_embedding",
                "rbf",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Za",
            },
            output_fields={
                "atom_feature",
                "isolated_atom_feature",
                "atom_steerable_feature",
            },
        )
        self.dim_embedding = int(dim_embedding)
        self.num_rbf = int(num_rbf)
        self.max_Za = int(max_Za)
        self.num_layers = int(num_layers)
        self.lmax = int(lmax)
        self.num_heads = int(num_heads)
        self.attn_dropout = float(attn_dropout)
        self.epsilon = float(epsilon)
        self.angular_dim = angular_dim(self.lmax)
        self.edge_updates = edge_updates
        self.edge_spec = parse_edge_updates(edge_updates)
        self.update_edges = edge_updates is not False
        self.evec_dim = self.dim_embedding if evec_dim is None else int(evec_dim)
        self.emlp_dim = self.dim_embedding if emlp_dim is None else int(emlp_dim)
        self.dim_feature_out = self.dim_embedding
        self.num_output_blocks = self.num_layers
        self.feature_irreps = f"{self.dim_embedding}x0e"
        self.emits_isolated_atom_feature = True
        # Not an e3nn / SPHC layout. l = 1 is (x, y, z) of the unit edge.
        self.steerable_layout = "geodite_polynomial"

        self.node_init = NodeInit(self.dim_embedding, self.num_rbf, self.max_Za)
        self.edge_init = EdgeInit(self.dim_embedding, self.num_rbf)
        self.interactions = ModuleList(
            [
                InteractionBlock(
                    channels=self.dim_embedding,
                    num_rbf=self.num_rbf,
                    lmax=self.lmax,
                    num_heads=self.num_heads,
                    dropout=self.attn_dropout,
                    epsilon=self.epsilon,
                    edge_spec=self.edge_spec,
                    update_edges=self.update_edges and layer < self.num_layers - 1,
                    evec_dim=self.evec_dim,
                    emlp_dim=self.emlp_dim,
                    edge_norm=edge_norm,
                )
                for layer in range(self.num_layers)
            ]
        )
        self.couplings = ModuleList(
            EquivariantCoupling(self.dim_embedding) for _ in range(self.num_layers)
        )

    def build(self, built_layers: List[Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def _edges(
        self, vij: Tensor
    ) -> Tensor:
        if vij.numel() == 0:
            return vij.new_zeros((0, self.angular_dim))
        norm = torch.linalg.vector_norm(vij, dim=-1, keepdim=True).clamp_min(1e-12)
        return geodite_spherical_harmonics(vij / norm, self.lmax)

    def _run(
        self,
        atom_embedding: Tensor,
        atomic_numbers: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        vij: Tensor,
    ) -> tuple[Tensor, Tensor]:
        scalars = self.node_init(atom_embedding, atomic_numbers, rbf, idx_i, idx_j)
        edges = self.edge_init(scalars, rbf, idx_i, idx_j)
        steerable = scalars.new_zeros(
            (scalars.shape[0], self.angular_dim, self.dim_embedding)
        )
        harmonics = self._edges(vij)
        states = []
        for interaction, coupling in zip(self.interactions, self.couplings):
            scalars, steerable, edges = interaction(
                scalars, steerable, edges, harmonics, rbf, idx_i, idx_j
            )
            scalars, steerable = coupling(scalars, steerable)
            states.append(scalars)
        return torch.stack(states, dim=-1), steerable

    def _isolated(self, atom_embedding: Tensor, atomic_numbers: Tensor) -> Tensor:
        """Neighbor-free states for each species present in the batch."""
        number = atomic_numbers.shape[0]
        if number == 0:
            return atom_embedding.new_zeros(
                (0, self.dim_embedding, self.num_layers)
            )
        unique, inverse = torch.unique(
            atomic_numbers.long(), sorted=True, return_inverse=True
        )
        choice = torch.empty(unique.shape[0], dtype=torch.long, device=atomic_numbers.device)
        choice.scatter_(
            0, inverse, torch.arange(number, device=atomic_numbers.device)
        )
        embedding = atom_embedding.index_select(0, choice)
        empty = atomic_numbers.new_zeros((0,))
        radial = atom_embedding.new_zeros((0, self.num_rbf))
        vectors = atom_embedding.new_zeros((0, 3))
        states, _ = self._run(embedding, unique, radial, empty, empty, vectors)
        return states.index_select(0, inverse)

    def get_output(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Za: Tensor,
    ) -> Dict[str, Tensor]:
        """Run every layer. ``idx_i_sr`` receives the message from ``idx_j_sr``.

        ``vij_sr = R_j - R_i``. ``rbf`` is the effective basis: Bessel
        functions already multiplied by the cutoff envelope and the covalent
        distance weight.
        """
        if atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding width {atom_embedding.shape[-1]} != "
                f"dim_embedding {self.dim_embedding}"
            )
        if rbf.ndim != 2 or rbf.shape[-1] != self.num_rbf:
            raise ValueError(
                f"rbf must have shape [E, {self.num_rbf}], got {tuple(rbf.shape)}"
            )
        if atom_embedding.shape[0] != Za.shape[0]:
            raise ValueError(
                f"atom_embedding has {atom_embedding.shape[0]} atoms but Za has {Za.shape[0]}"
            )
        receivers = idx_i_sr.long()
        senders = idx_j_sr.long()
        states, steerable = self._run(
            atom_embedding, Za.long(), rbf, receivers, senders, vij_sr
        )
        return {
            "atom_feature": states,
            "isolated_atom_feature": self._isolated(atom_embedding, Za),
            "atom_steerable_feature": steerable,
        }
