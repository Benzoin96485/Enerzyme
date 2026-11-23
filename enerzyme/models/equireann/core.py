"""EquiREANN Core: REANN densities with an equivariant orbital message.

Wu, Xia, Zhang, and Jiang, J. Phys. Chem. A 128, 11061 (2024). The iteration
follows https://github.com/bjiangch/EquiREANN ``reann/src/density.py``. That
package is not imported, and its checkpoints do not load.

The Core reads ``atom_embedding`` and the short-range geometry. It emits the
final invariant density ``atom_feature`` ``[N, n_orbit]``. Species embedding,
the energy map, charges, dipoles, and polarizabilities stay in the layer stack.
``num_iterations = 0`` is the embedded-atom density, the same contraction as
REANN. Each later iteration adds a residual coefficient update and passes the
neighbor's aggregated Cartesian orbital back onto the edge.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from ..reann.core import REANNCore
from ..reann.density import (
    aggregate_weighted_orbitals,
    invariant_orbital_density,
    pair_gaussian_orbitals,
)
from .message import add_neighbor_orbital


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

_READOUT = {
    "output_fields": ["Ea"],
    "head_type": "mlp",
    "hidden_dims": [32, 32],
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
            "nwave": 20,
            "num_iterations": 2,
            "nblock": 1,
            "hidden_dims": [32, 32],
            "activation": "tanh_like",
            "layer_norm": False,
            "dropout": 0.0,
            "elemental": False,
            "coeff_init": 0.1,
        },
    },
    {"name": "SimpleReadout", "params": dict(_READOUT)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

# Cumulene section of the paper: Nφ = 8, L = 2, T = 2, 16×16 networks, rc = 6 Å.
# Eq. 1 uses Nφ for the contracted width. The NequIP timing paragraph also calls
# 8 the basis count, so nwave is set to 8 here. para/input_density instead uses
# nwave 20 and the default orbit formula; that is the checked-in example.
PUBLISHED_CUMULENE_CORE_PARAMS = {
    "lmax": 2,
    "nwave": 8,
    "n_orbit": 8,
    "num_iterations": 2,
    "nblock": 1,
    "hidden_dims": [16, 16],
    "activation": "tanh_like",
    "layer_norm": False,
    "dropout": 0.0,
    "elemental": False,
}
PUBLISHED_CUMULENE_CUTOFF = 6.0


class EquiREANNCore(REANNCore):
    """REANN coefficient updates plus the equivariant orbital passed in Eq. 5."""

    def __str__(self) -> str:
        return (
            "EquiREANN (Wu, Xia, Zhang, and Jiang, J. Phys. Chem. A 2024)\n"
            f"lmax={self.lmax}  nwave={self.nwave}  n_orbit={self.n_orbit}  "
            f"iterations={self.num_iterations}  elemental={self.elemental}"
        )

    def __init__(
        self,
        dim_embedding: int = 32,
        max_Za: int = 94,
        cutoff_sr: float = 6.0,
        lmax: int = 2,
        nwave: int = 20,
        n_orbit: Optional[int] = None,
        num_iterations: int = 2,
        nblock: int = 1,
        hidden_dims: Sequence[int] = (32, 32),
        activation: str = "tanh_like",
        layer_norm: bool = False,
        dropout: float = 0.0,
        elemental: bool = False,
        atomic_numbers: Optional[Sequence[int]] = None,
        coeff_init: float = 0.1,
    ) -> None:
        super().__init__(
            dim_embedding=dim_embedding,
            max_Za=max_Za,
            cutoff_sr=cutoff_sr,
            lmax=lmax,
            nwave=nwave,
            n_orbit=n_orbit,
            num_iterations=num_iterations,
            nblock=nblock,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
            elemental=elemental,
            atomic_numbers=atomic_numbers,
            coeff_init=coeff_init,
        )
        base = torch.empty(self.nwave, self.n_orbit)
        nn.init.xavier_uniform_(base)
        self.hyper = nn.Parameter(
            base.view(1, 1, self.nwave, self.n_orbit).repeat(
                self.num_iterations + 1, self.lmax + 1, 1, 1
            )
        )

    def _contract(
        self,
        pair: Tensor,
        coeff: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        hyper: Tensor,
    ) -> tuple[Tensor, Tensor]:
        psi = aggregate_weighted_orbitals(pair, coeff, idx_i, idx_j, coeff.shape[0])
        density = invariant_orbital_density(psi, hyper, self.l_of_channel)
        return density, psi

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
        n_atoms = atom_embedding.shape[0]
        if idx_i_sr.numel() == 0:
            density = atom_embedding.new_zeros((n_atoms, self.n_orbit))
            for update in self.updates:
                coeff = coeff + update(density, species)
            return {"atom_feature": density}
        pair, window = pair_gaussian_orbitals(
            idx_i_sr,
            idx_j_sr,
            vij_sr,
            Dij_sr,
            species,
            self.rs,
            self.inta,
            self.lmax,
            self.cutoff_sr,
        )
        density, psi = self._contract(pair, coeff, idx_i_sr, idx_j_sr, self.hyper[0])
        for step, update in enumerate(self.updates):
            coeff = coeff + update(density, species)
            pair = add_neighbor_orbital(pair, psi, idx_j_sr, window)
            density, psi = self._contract(pair, coeff, idx_i_sr, idx_j_sr, self.hyper[step + 1])
        return {"atom_feature": density}
