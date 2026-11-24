"""Initial orbital density and recursive coefficient/message blocks for both REANN models."""

from __future__ import annotations

import torch
from torch import nn

from ..mp.block import BlockState
from ..mp.core import Embed
from ..equireann.message import add_neighbor_orbital
from .density import (
    aggregate_weighted_orbitals,
    angular_momentum_index,
    invariant_orbital_density,
    pair_gaussian_orbitals,
    validate_species,
)
from .network import SpeciesMLP


class OrbitalEmbed(Embed):
    input_fields = ("atom_embedding", "Za")

    def __init__(self, dim_embedding, max_Za, cutoff_sr, lmax, nwave, n_orbit, coeff_init):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.max_Za = max_Za
        self.lmax = lmax
        self.nwave = nwave
        self.n_orbit = n_orbit
        self.cutoff_sr = cutoff_sr
        self.coeff_init = coeff_init
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

    def enable_orbital_messages(self, iterations):
        """EquiREANN uses a separately learned contraction at every iteration."""
        base = torch.empty(self.nwave, self.n_orbit)
        nn.init.xavier_uniform_(base)
        self.hyper = nn.Parameter(
            base.view(1, 1, self.nwave, self.n_orbit).repeat(
                iterations + 1, self.lmax + 1, 1, 1
            )
        )

    def forward(self, inputs, ctx):
        atom_embedding, Za = inputs["atom_embedding"], inputs["Za"]
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
        validate_species(species, self.max_Za, "Za")
        coeff = self.coeff(atom_embedding)
        if ctx.receiver.numel() == 0:
            return BlockState(scalar=atom_embedding.new_zeros(ctx.num_nodes, self.n_orbit), equi=(coeff, species, None))
        pair, window = pair_gaussian_orbitals(
            ctx.receiver, ctx.sender, ctx.vec, ctx.dist, species, self.rs, self.inta, self.lmax, self.cutoff_sr,
            species_checked=True,
        )
        psi = aggregate_weighted_orbitals(pair, coeff, ctx.receiver, ctx.sender, ctx.num_nodes)
        first = self.hyper[0] if self.hyper.ndim == 4 else self.hyper
        density = invariant_orbital_density(psi, first, self.l_of_channel, ordered_shells=True)
        return BlockState(scalar=density, equi=(coeff, species, psi), edge=(pair, window, self.hyper, self.l_of_channel))


class OrbitalUpdate(SpeciesMLP):
    def __init__(self, nwave, atomic_numbers, network, orbital_step):
        super().__init__(nwave, atomic_numbers, network)
        self.orbital_step = orbital_step

    def forward(self, state, ctx):
        coeff, species, psi = state.equi
        coeff = coeff + super().forward(state.scalar, species)
        if state.edge is None:
            return BlockState(scalar=state.scalar, equi=(coeff, species, None))
        pair, window, hyper, shell = state.edge
        if self.orbital_step is None:
            contraction = hyper
        else:
            pair = add_neighbor_orbital(pair, psi, ctx.sender, window)
            contraction = hyper[self.orbital_step]
        psi = aggregate_weighted_orbitals(pair, coeff, ctx.receiver, ctx.sender, ctx.num_nodes)
        density = invariant_orbital_density(psi, contraction, shell, ordered_shells=True)
        return BlockState(scalar=density, equi=(coeff, species, psi), edge=(pair, window, hyper, shell))


def build_orbital_stack(core, built_layers) -> None:
    from ..layers.geometry import RangeSeparationLayer

    for candidate in built_layers:
        if not isinstance(candidate, RangeSeparationLayer):
            continue
        if abs(float(candidate.cutoff_sr) - core.cutoff_sr) > 1e-4:
            raise ValueError(
                "RangeSeparation cutoff_sr "
                f"({candidate.cutoff_sr}) must equal REANN cutoff_sr "
                f"({core.cutoff_sr})"
            )
    core.adopt_radius_stack(built_layers)
