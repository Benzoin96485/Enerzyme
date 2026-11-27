"""EquiREANN orbital message: formula, rotation, and the cumulene dihedral."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose

from enerzyme.models.equireann.message import add_neighbor_orbital
from enerzyme.models.reann.density import (
    aggregate_weighted_orbitals,
    angular_momentum_index,
    cartesian_angular,
    embedded_atom_density,
    invariant_orbital_density,
    pair_gaussian_orbitals,
    squared_cosine_window,
)


def _rotation(seed: int = 0, improper: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    if improper:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_packed_orbitals_preserve_recursive_density_and_force_loss_gradients():
    from enerzyme.models.reann.blocks import OrbitalEmbed, _project_orbitals

    torch.manual_seed(560)
    embed = OrbitalEmbed(4, 3, 5., 3, 2, 5, 0.1)
    embed.enable_orbital_messages(2)
    embed.double()
    receiver = torch.tensor([0, 0, 1, 1, 2, 2])
    sender = torch.tensor([1, 2, 0, 2, 0, 1])
    species = torch.tensor([1, 3, 2])
    vec = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
    coeff = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)

    def run(packed):
        shell = embed.packed_shells if packed else embed.l_of_channel
        layout = (embed.representatives, embed.multiplicities) if packed else None
        pair, cutoff = pair_gaussian_orbitals(receiver, sender, vec, vec.norm(dim=1), species,
                                              embed.rs, embed.inta, 3, 5., symmetric_channels=layout)
        for i in range(3):
            psi = aggregate_weighted_orbitals(pair, coeff, receiver, sender, 3)
            density = _project_orbitals(psi, embed.hyper[i], shell, packed)
            if i < 2:
                pair = add_neighbor_orbital(pair, psi, sender, cutoff)
        force, = torch.autograd.grad(density.sum(), vec, create_graph=True)
        gradient = torch.autograd.grad(force.square().sum(), (coeff, embed.rs, embed.inta, embed.hyper))
        return (density, force, *gradient)

    assert embed.representatives.numel() == 20
    assert embed.multiplicities.sum().item() == 40
    for a, b in zip(run(True), run(False)):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_neighbor_orbital_is_windowed_and_indexed_by_the_sender():
    pair = torch.ones(2, 4, 1, dtype=torch.float64)
    psi = torch.tensor(
        [[[2.0], [0.0], [0.0], [0.0]], [[5.0], [0.0], [0.0], [0.0]]],
        dtype=torch.float64,
    )
    window = torch.tensor([0.25, 0.5], dtype=torch.float64)
    idx_j = torch.tensor([1, 0])
    updated = add_neighbor_orbital(pair, psi, idx_j, window)
    expected = pair.clone()
    expected[0] = expected[0] + 0.25 * psi[1]
    expected[1] = expected[1] + 0.5 * psi[0]
    assert_allclose(updated.numpy(), expected.numpy())


def test_vector_channel_tracks_a_rotation_of_the_displacement():
    cutoff = 4.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, -0.2, 0.5], [0.1, 0.3, -0.2]], dtype=torch.float64)
    species = torch.tensor([1, 6, 8])
    idx_i = torch.tensor([0, 0])
    idx_j = torch.tensor([1, 2])
    coeff = torch.tensor([[0.0], [1.5], [0.7]], dtype=torch.float64)
    rs = torch.zeros(9, 1, dtype=torch.float64)
    inta = -torch.ones(9, 1, dtype=torch.float64)
    vij = pos[idx_j] - pos[idx_i]
    distance = torch.linalg.vector_norm(vij, dim=-1)
    pair, window = pair_gaussian_orbitals(
        idx_i, idx_j, vij, distance, species, rs, inta, 1, cutoff
    )
    psi = aggregate_weighted_orbitals(pair, coeff, idx_i, idx_j, pos.shape[0])
    radial = torch.exp(-(distance.unsqueeze(-1) - 0.0).square())
    geometric = (window.unsqueeze(-1) * vij).unsqueeze(-1) * radial.unsqueeze(1)
    weighted = geometric * coeff[idx_j].unsqueeze(1)
    assert_allclose(pair[:, 1:4].numpy(), geometric.numpy(), atol=1e-12)
    assert_allclose(psi[0, 1:4, 0].numpy(), weighted[:, :, 0].sum(0).numpy(), atol=1e-12)
    assert_allclose(psi[1].numpy(), torch.zeros_like(psi[1]).numpy(), atol=1e-12)

    rotation = _rotation(3)
    rotated = pos @ rotation
    vij_r = rotated[idx_j] - rotated[idx_i]
    pair_r, _ = pair_gaussian_orbitals(
        idx_i, idx_j, vij_r, torch.linalg.vector_norm(vij_r, dim=-1), species, rs, inta, 1, cutoff
    )
    psi_r = aggregate_weighted_orbitals(pair_r, coeff, idx_i, idx_j, pos.shape[0])
    assert_allclose(psi_r[0, 1:4, 0].numpy(), (psi[0, 1:4, 0] @ rotation).numpy(), atol=1e-10)
    improper = _rotation(5, improper=True)
    vij_i = (pos @ improper)[idx_j] - (pos @ improper)[idx_i]
    pair_i, _ = pair_gaussian_orbitals(
        idx_i, idx_j, vij_i, torch.linalg.vector_norm(vij_i, dim=-1), species, rs, inta, 1, cutoff
    )
    psi_i = aggregate_weighted_orbitals(pair_i, coeff, idx_i, idx_j, pos.shape[0])
    assert_allclose(psi_i[0, 1:4, 0].numpy(), (psi[0, 1:4, 0] @ improper).numpy(), atol=1e-10)
    shell = angular_momentum_index(1)
    hyper = torch.ones(2, 1, 1, dtype=torch.float64)
    assert_allclose(
        invariant_orbital_density(psi, hyper, shell).numpy(),
        invariant_orbital_density(psi_r, hyper, shell).numpy(),
        atol=1e-10,
    )


def test_one_message_matches_the_closed_form_and_reann_at_zero_iterations():
    cutoff = 3.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    species = torch.tensor([6, 1])
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    vij = pos[idx_j] - pos[idx_i]
    distance = torch.linalg.vector_norm(vij, dim=-1)
    coeff = torch.ones(2, 1, dtype=torch.float64)
    rs = torch.zeros(9, 1, dtype=torch.float64)
    inta = -torch.ones(9, 1, dtype=torch.float64)
    hyper = torch.ones(1, 1, 1, dtype=torch.float64)
    shell = angular_momentum_index(0)
    window = squared_cosine_window(distance, cutoff)
    phi = window * torch.exp(-distance.square())
    # l = 0, one radial function, unit coefficients and contraction.
    base = phi.square()
    density0 = embedded_atom_density(
        coeff, idx_i, idx_j, vij, distance, species, rs, inta, hyper, shell, cutoff
    )
    assert_allclose(density0[:, 0].numpy(), base.numpy(), atol=1e-12)
    passed = phi + phi * window
    assert_allclose(passed.numpy(), (phi * (1.0 + window)).numpy(), atol=1e-12)
    pair, edge_window = pair_gaussian_orbitals(
        idx_i, idx_j, vij, distance, species, rs, inta, 0, cutoff
    )
    psi = aggregate_weighted_orbitals(pair, coeff, idx_i, idx_j, 2)
    stepped = add_neighbor_orbital(pair, psi, idx_j, edge_window)
    again = aggregate_weighted_orbitals(stepped, coeff, idx_i, idx_j, 2)
    density1 = invariant_orbital_density(again, hyper, shell)
    assert_allclose(density1[:, 0].numpy(), passed.square().numpy(), atol=1e-12)
    # The geometric channel order used above is the released product at l = 0.
    angular = cartesian_angular(vij, window, 0)
    assert_allclose(angular.numpy(), window.view(1, -1).numpy(), atol=1e-12)
    assert math.isclose(float(window[0]), (0.5 * math.cos(math.pi / cutoff) + 0.5) ** 2)


def _ethene(dihedral_deg: float) -> torch.Tensor:
    """C2H4 with unequal methylene bonds on the second carbon."""
    cc = 1.34
    c1 = torch.zeros(3, dtype=torch.float64)
    c2 = torch.tensor([cc, 0.0, 0.0], dtype=torch.float64)
    angle = math.radians(120.0)
    ch = 1.09
    h1 = torch.tensor([ch * math.cos(angle), ch * math.sin(angle), 0.0], dtype=torch.float64)
    h2 = torch.tensor([ch * math.cos(angle), -ch * math.sin(angle), 0.0], dtype=torch.float64)
    phi = math.radians(dihedral_deg)
    # 120 degrees from the C2 -> C1 direction (-x), then rolled around the bond.
    bonds = (1.05, 1.20)
    hydrogens = []
    for sign, length in ((1.0, bonds[0]), (-1.0, bonds[1])):
        local = torch.tensor(
            [length * math.cos(math.radians(60.0)), sign * length * math.sin(math.radians(60.0)), 0.0],
            dtype=torch.float64,
        )
        rolled = torch.tensor(
            [local[0], local[1] * math.cos(phi), local[1] * math.sin(phi)],
            dtype=torch.float64,
        )
        hydrogens.append(c2 + rolled)
    return torch.stack([c1, c2, h1, h2, *hydrogens])


def _neighbor_counts(pos: torch.Tensor, cutoff: float) -> list[int]:
    distance = torch.linalg.vector_norm(pos.unsqueeze(0) - pos.unsqueeze(1), dim=-1)
    return [int(((distance[i] < cutoff) & (distance[i] > 0)).sum()) for i in range(pos.shape[0])]


def test_a_terminal_dihedral_changes_equireann_and_not_reann():
    from enerzyme.models.presets.cartesian_tensor import equireann
    from enerzyme.models.presets.cartesian_tensor import reann

    cutoff = 1.8
    pos_flat = _ethene(0.0)
    pos_twist = _ethene(90.0)
    counts = _neighbor_counts(pos_flat, cutoff)
    assert counts == [3, 3, 1, 1, 1, 1]
    assert _neighbor_counts(pos_twist, cutoff) == counts
    # The twist must not put a hydrogen into the other carbon's cutoff.
    cross = torch.linalg.vector_norm(pos_twist[0] - pos_twist[4])
    assert float(cross) > cutoff

    # l = 1 cancels on a symmetric methylene: the transverse vector is perpendicular
    # to the receiver's bond axis, so its square does not see the roll. l = 2 keeps it.
    kwargs = dict(
        dim_embedding=4,
        max_Za=10,
        cutoff_sr=cutoff,
        lmax=2,
        nwave=1,
        n_orbit=1,
        num_iterations=1,
        nblock=1,
        hidden_dims=(4,),
        activation="tanh_like",
        layer_norm=False,
        coeff_init=1.0,
    )
    equi = equireann(**kwargs).double()
    plain = reann(**kwargs).double()
    with torch.no_grad():
        equi.embed.rs.zero_()
        equi.embed.inta.fill_(-1.0)
        equi.embed.hyper.fill_(1.0)
        equi.embed.coeff.weight.zero_()
        equi.embed.coeff.bias.fill_(1.0)
        plain.embed.rs.copy_(equi.embed.rs)
        plain.embed.inta.copy_(equi.embed.inta)
        plain.embed.hyper.copy_(equi.embed.hyper[0])
        plain.embed.coeff.load_state_dict(equi.embed.coeff.state_dict())
        plain.blocks.load_state_dict(equi.blocks.state_dict())
    species = torch.tensor([6, 6, 1, 1, 1, 1])
    embedding = torch.zeros(6, 4, dtype=torch.float64)

    def run(core, pos):
        idx_i, idx_j = [], []
        for i in range(pos.shape[0]):
            for j in range(pos.shape[0]):
                if i != j and float(torch.linalg.vector_norm(pos[j] - pos[i])) < cutoff:
                    idx_i.append(i)
                    idx_j.append(j)
        idx_i = torch.tensor(idx_i)
        idx_j = torch.tensor(idx_j)
        vij = pos[idx_j] - pos[idx_i]
        return core.get_output(
            atom_embedding=embedding,
            Za=species,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=vij,
            Dij_sr=torch.linalg.vector_norm(vij, dim=-1),
        )["atom_feature"]

    reann_flat = run(plain, pos_flat)
    reann_twist = run(plain, pos_twist)
    assert_allclose(reann_flat.detach().numpy(), reann_twist.detach().numpy(), atol=1e-10)
    equi_flat = run(equi, pos_flat)
    equi_twist = run(equi, pos_twist)
    assert not torch.allclose(equi_flat, equi_twist, atol=1e-6)
    # The first carbon never sees the far hydrogens directly. Its density still moves.
    assert not torch.allclose(equi_flat[0], equi_twist[0], atol=1e-6)
    quiet = equireann(**{**kwargs, "num_iterations": 0}).double()
    with torch.no_grad():
        quiet.embed.rs.copy_(equi.embed.rs)
        quiet.embed.inta.copy_(equi.embed.inta)
        quiet.embed.hyper.copy_(equi.embed.hyper[:1])
        quiet.embed.coeff.load_state_dict(equi.embed.coeff.state_dict())
    assert_allclose(
        reann_flat.detach().numpy(),
        run(quiet, pos_flat).detach().numpy(),
        atol=1e-10,
    )
