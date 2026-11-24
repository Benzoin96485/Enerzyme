"""Initial orbital density and recursive coefficient/message blocks for both REANN models."""

from __future__ import annotations

from itertools import combinations_with_replacement

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
        from ..mp.equivariant.body_order import multinomial

        # Products differing only by an axis permutation transform together and
        # all subsequent orbital messages are linear in those products. Keep
        # one representative, scaled by sqrt(multiplicity) before the final norm.
        representatives, multiplicities, shells = [], [], []
        for degree in range(self.lmax + 1):
            offset = (3 ** degree - 1) // 2
            for axes in combinations_with_replacement(range(3), degree):
                index = 0
                for axis in axes:
                    index = 3 * index + axis
                representatives.append(offset + index)
                multiplicities.append(multinomial(axes.count(0), axes.count(1), axes.count(2)))
                shells.append(degree)
        self.register_buffer("representatives", torch.tensor(representatives), persistent=False)
        self.register_buffer("multiplicities", torch.tensor(multiplicities), persistent=False)
        self.register_buffer("packed_shells", torch.tensor(shells), persistent=False)
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
        pass_orbitals = self.hyper.ndim == 4
        # CPU Inductor needs the ordered shells for correct coordinate
        # derivatives. CUDA force inference can share permutation-equivalent
        # channels without saving the repeated edge orbitals for backward.
        packed = pass_orbitals and (
            not torch.compiler.is_compiling()
            or (ctx.vec.is_cuda and not self.training and not torch._C._are_functorch_transforms_active())
        )
        pair, window = pair_gaussian_orbitals(
            ctx.receiver, ctx.sender, ctx.vec, ctx.dist, species, self.rs, self.inta, self.lmax, self.cutoff_sr,
            species_checked=True,
            symmetric_channels=(self.representatives, self.multiplicities) if packed else None,
        )
        psi = aggregate_weighted_orbitals(pair, coeff, ctx.receiver, ctx.sender, ctx.num_nodes)
        shell = self.packed_shells if packed else self.l_of_channel
        first = self.hyper[0] if pass_orbitals else self.hyper
        density = _project_orbitals(psi, first, shell, packed)
        return BlockState(scalar=density, equi=(coeff, species, psi), edge=(pair, window, self.hyper, shell, packed))


def _project_orbitals(psi, hyper, shell, packed):
    if packed:
        projection = torch.bmm(psi.transpose(0, 1), hyper.to(psi)[shell])
        return projection.square().sum(0)
    return invariant_orbital_density(psi, hyper, shell, ordered_shells=True)


class OrbitalUpdate(SpeciesMLP):
    def __init__(self, nwave, atomic_numbers, network, orbital_step):
        super().__init__(nwave, atomic_numbers, network)
        self.orbital_step = orbital_step

    def forward(self, state, ctx):
        coeff, species, psi = state.equi
        coeff = coeff + super().forward(state.scalar, species)
        if state.edge is None:
            return BlockState(scalar=state.scalar, equi=(coeff, species, None))
        pair, window, hyper, shell, packed = state.edge
        if self.orbital_step is None:
            contraction = hyper
        else:
            pair = add_neighbor_orbital(pair, psi, ctx.sender, window)
            contraction = hyper[self.orbital_step]
        psi = aggregate_weighted_orbitals(pair, coeff, ctx.receiver, ctx.sender, ctx.num_nodes)
        density = _project_orbitals(psi, contraction, shell, packed)
        return BlockState(scalar=density, equi=(coeff, species, psi), edge=(pair, window, hyper, shell, packed))


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
