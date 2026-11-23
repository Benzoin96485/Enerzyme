"""REANN Core: recursive embedded-atom densities.

Zhang, Xia, and Jiang, J. Chem. Phys. 156, 114801 (2022). The theory is
Phys. Rev. Lett. 127, 156002 (2021). The contraction follows
https://github.com/zhangylch/REANN. That package is not imported, and its
checkpoints do not load.

The Core reads ``atom_embedding`` and the short-range geometry. It emits the
final invariant density ``atom_feature`` ``[N, n_orbit]``. Species embedding,
the energy map, charges, dipoles, and polarizabilities stay in the layer stack.
``num_iterations = 0`` is the embedded-atom network without coefficient updates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from ..layers import BaseFFCore
from .density import (
    angular_momentum_index,
    default_n_orbit,
    embedded_atom_density,
)
from .network import SpeciesMLP, resolve_reann_activation


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 5.0,
}

_READOUT = {
    "output_fields": ["Ea"],
    "head_type": "mlp",
    "hidden_dims": [128, 128],
    "activation_fn": "relu_like",
    "activation_params": {
        "dim_feature": 1,
        "initial_alpha": 1.0,
        "initial_beta": 1.0,
        "learnable": False,
    },
    "keep_feature": False,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "lmax": 2,
            "nwave": 8,
            "num_iterations": 1,
            "nblock": 1,
            "hidden_dims": [128, 128],
            "activation": "relu_like",
            "layer_norm": True,
            "dropout": 0.0,
            "elemental": False,
            "coeff_init": 0.1,
        },
    },
    {"name": "SimpleReadout", "params": dict(_READOUT)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

# MD17 column of Table I. ``elemental: true`` also needs ``atomic_numbers``
# for the species present in that molecule. The energy network is the shared
# readout, with the same hidden widths.
PUBLISHED_MD17_CORE_PARAMS = {
    "lmax": 2,
    "nwave": 20,
    "num_iterations": 1,
    "nblock": 1,
    "hidden_dims": [128, 128],
    "activation": "relu_like",
    "layer_norm": True,
    "dropout": 0.0,
    "elemental": True,
}
PUBLISHED_MD17_CUTOFF = 8.0


class REANNCore(BaseFFCore):
    """Recursive embedded-atom density. Property heads stay outside."""

    def __str__(self) -> str:
        return (
            "REANN (Zhang, Xia, and Jiang, J. Chem. Phys. 2022)\n"
            f"lmax={self.lmax}  nwave={self.nwave}  n_orbit={self.n_orbit}  "
            f"iterations={self.num_iterations}  elemental={self.elemental}"
        )

    def __init__(
        self,
        dim_embedding: int = 32,
        max_Za: int = 94,
        cutoff_sr: float = 5.0,
        lmax: int = 2,
        nwave: int = 8,
        n_orbit: Optional[int] = None,
        num_iterations: int = 1,
        nblock: int = 1,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "relu_like",
        layer_norm: bool = True,
        dropout: float = 0.0,
        elemental: bool = False,
        atomic_numbers: Optional[Sequence[int]] = None,
        coeff_init: float = 0.1,
    ) -> None:
        if int(dim_embedding) < 1:
            raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
        if int(max_Za) < 1:
            raise ValueError(f"max_Za must be >= 1, got {max_Za}")
        if float(cutoff_sr) <= 0.0:
            raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
        if int(lmax) < 0 or int(lmax) > 6:
            raise ValueError(f"lmax must lie in 0..6, got {lmax}")
        if int(nwave) < 1:
            raise ValueError(f"nwave must be >= 1, got {nwave}")
        if int(num_iterations) < 0:
            raise ValueError(f"num_iterations must be >= 0, got {num_iterations}")
        if int(nblock) < 0:
            raise ValueError(f"nblock must be >= 0, got {nblock}")
        orbit = default_n_orbit(int(nwave), int(lmax)) if n_orbit is None else int(n_orbit)
        if orbit < 1:
            raise ValueError(f"n_orbit must be >= 1, got {orbit}")
        numbers: Optional[List[int]]
        if elemental:
            if not atomic_numbers:
                raise ValueError(
                    "elemental=True requires atomic_numbers, the species that own a network"
                )
            numbers = [int(number) for number in atomic_numbers]
            if any(number < 0 or number > int(max_Za) for number in numbers):
                raise ValueError(
                    f"atomic_numbers must lie in 0..max_Za ({max_Za}), got {numbers}"
                )
        else:
            numbers = None
        super().__init__(
            input_fields={
                "atom_embedding",
                "Za",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "Dij_sr",
            },
            output_fields={"atom_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.max_Za = int(max_Za)
        self.cutoff_sr = float(cutoff_sr)
        self.lmax = int(lmax)
        self.nwave = int(nwave)
        self.n_orbit = orbit
        self.num_iterations = int(num_iterations)
        self.nblock = int(nblock)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.activation = resolve_reann_activation(activation)
        self.layer_norm = bool(layer_norm)
        self.dropout = float(dropout)
        self.elemental = bool(elemental)
        self.atomic_numbers = None if numbers is None else tuple(numbers)
        self.coeff_init = float(coeff_init)
        self.dim_feature_out = self.n_orbit
        self.feature_irreps = f"{self.n_orbit}x0e"
        self.register_buffer(
            "l_of_channel",
            angular_momentum_index(self.lmax),
            persistent=False,
        )
        self.rs = nn.Parameter(torch.rand(self.max_Za + 1, self.nwave) * self.cutoff_sr)
        self.inta = nn.Parameter(-(torch.rand(self.max_Za + 1, self.nwave) + 0.2))
        hyper = torch.empty(self.nwave, self.n_orbit)
        nn.init.xavier_uniform_(hyper)
        self.hyper = nn.Parameter(hyper.unsqueeze(0).repeat(self.lmax + 1, 1, 1))
        self.coeff = nn.Linear(self.dim_embedding, self.nwave)
        nn.init.zeros_(self.coeff.weight)
        nn.init.constant_(self.coeff.bias, self.coeff_init)
        self.updates = nn.ModuleList(
            [
                SpeciesMLP(self.nwave, self.atomic_numbers, self._new_update)
                for _ in range(self.num_iterations)
            ]
        )

    def _new_update(self):
        from .network import REANNMLP

        return REANNMLP(
            n_in=self.n_orbit,
            n_out=self.nwave,
            hidden_dims=self.hidden_dims,
            nblock=self.nblock,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )

    def build(self, built_layers: List[nn.Module]) -> None:
        self.adopt_radius_stack(built_layers)

    def _density(self, coeff: Tensor, species: Tensor, idx_i, idx_j, vij, distance) -> Tensor:
        return embedded_atom_density(
            coeff,
            idx_i,
            idx_j,
            vij,
            distance,
            species,
            self.rs,
            self.inta,
            self.hyper,
            self.l_of_channel,
            self.cutoff_sr,
        )

    def get_output(
        self,
        atom_embedding: Tensor,
        Za: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        Dij_sr: Tensor,
    ) -> Dict[str, Tensor]:
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.dim_embedding:
            raise ValueError(
                f"atom_embedding must have shape [N, {self.dim_embedding}], "
                f"got {tuple(atom_embedding.shape)}"
            )
        species = Za.reshape(-1).long()
        if species.shape[0] != atom_embedding.shape[0]:
            raise ValueError(
                f"Za has {species.shape[0]} atoms but atom_embedding has {atom_embedding.shape[0]}"
            )
        if species.numel() and (int(species.min()) < 0 or int(species.max()) > self.max_Za):
            raise ValueError(
                f"Za must lie in 0..{self.max_Za}, "
                f"got {int(species.min())}..{int(species.max())}"
            )
        coeff = self.coeff(atom_embedding)
        density = self._density(coeff, species, idx_i_sr, idx_j_sr, vij_sr, Dij_sr)
        for update in self.updates:
            coeff = coeff + update(density, species)
            density = self._density(coeff, species, idx_i_sr, idx_j_sr, vij_sr, Dij_sr)
        return {"atom_feature": density}
