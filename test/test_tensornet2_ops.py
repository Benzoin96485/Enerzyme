"""TensorNet2 charge equilibration, Coulomb, and the fixed ZBL wall."""

from __future__ import annotations

import math

import torch


def test_neutral_charge_equilibration_channels_and_batches():
    from enerzyme.models.layers.electrostatics import neutral_charge_equilibration

    charges = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    )
    weights = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
    )
    batch = torch.tensor([0, 0, 0, 1, 1])
    total = torch.tensor([3.0, 3.0, 3.0, -1.0, -1.0])
    updated = neutral_charge_equilibration(charges, weights, total, batch, eps=0.0)
    assert torch.allclose(updated[:3].sum(0), torch.tensor([3.0, 3.0]))
    assert torch.allclose(updated[3:].sum(0), torch.tensor([-1.0, -1.0]))
    # Channel 0 of the first molecule splits the residual of 2 equally.
    assert torch.allclose(updated[0, 0], torch.tensor(1.0 + 2.0 / 3.0))
    # Channel 1 puts the whole residual on the only nonzero weight.
    assert torch.allclose(updated[:3, 1], torch.tensor([2.0, 0.0, 1.0]))


def test_zero_weights_shift_uniformly_and_inconsistent_charge_fails():
    from enerzyme.models.layers.electrostatics import neutral_charge_equilibration

    charges = torch.tensor([[0.0, 4.0], [0.0, -1.0]])
    weights = torch.tensor([[0.0, 1.0], [0.0, 3.0]])
    updated = neutral_charge_equilibration(charges, weights, torch.tensor(2.0), eps=0.0)
    assert torch.allclose(updated[:, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(updated[:, 1], torch.tensor([3.75, -1.75]))

    isolated = neutral_charge_equilibration(
        torch.zeros(3, 1),
        torch.tensor([[1.0], [1.0], [0.0]]),
        torch.tensor([2.0, 2.0, 5.0]),
        torch.tensor([0, 0, 1]),
        eps=0.0,
    )
    assert torch.allclose(isolated.view(-1), torch.tensor([1.0, 1.0, 5.0]))

    try:
        neutral_charge_equilibration(
            torch.zeros(2, 1),
            torch.ones(2, 1),
            torch.tensor([1.0, 2.0]),
            torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "molecular total" in str(exc)
    else:
        raise AssertionError("expected inconsistent per-atom Q to fail")


def test_aimnet_damping_endpoints_and_coulomb_weights():
    from enerzyme.models.layers.electrostatics import (
        aimnet_charge_damping,
        expand_coulomb_weights,
    )

    distance = torch.tensor([0.0, 2.0, 10.0])
    damping = aimnet_charge_damping(distance, 4.6)
    assert torch.allclose(damping[0], torch.tensor(0.0), atol=1e-6)
    assert 0.0 < float(damping[1]) < 1.0
    assert torch.allclose(damping[2], torch.tensor(1.0), atol=1e-5)
    reduced = 2.0 / 4.6
    screen = math.exp(-1.0 / (1.0 - reduced**2)) / math.exp(-1.0)
    assert torch.allclose(damping[1], torch.tensor(1.0 - screen), atol=1e-6)

    weights = expand_coulomb_weights([1.0, 3.0], 4, torch.zeros(1))
    assert torch.allclose(weights, torch.tensor([0.125, 0.125, 0.375, 0.375]))
    try:
        expand_coulomb_weights([1.0, 2.0, 3.0], 4, torch.zeros(1))
    except ValueError as exc:
        assert "divide" in str(exc)
    else:
        raise AssertionError("expected an incompatible weight length to fail")


def test_weighted_coulomb_matches_pair_formula_and_directed_list():
    from enerzyme.models.layers.electrostatics import (
        WeightedCoulombEnergyLayer,
        aimnet_charge_damping,
        coulomb_kernel,
    )

    bohr = 0.5291772105638411
    layer = WeightedCoulombEnergyLayer(
        layer_weights=[1.0, 3.0],
        damping_cutoff=4.6,
        Bohr_in_R=bohr,
        Hartree_in_E=1.0,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    charges = torch.tensor([[0.4, -0.2], [-0.4, 0.5]])
    atomic = layer.get_E_coul_a(positions, charges, batch_seg=torch.zeros(2, dtype=torch.long))
    damping = aimnet_charge_damping(torch.tensor(2.0), 4.6)
    kernel = coulomb_kernel(torch.tensor(2.0))
    pair = 0.5 * bohr * damping * kernel * charges[0] * charges[1]
    expected_edge = (pair * torch.tensor([0.25, 0.75])).sum()
    assert torch.allclose(atomic.sum(), 2 * expected_edge, atol=1e-6)
    assert torch.allclose(atomic[0], expected_edge, atol=1e-6)

    directed = WeightedCoulombEnergyLayer(
        layer_weights=[1.0, 3.0],
        pairs="directed",
        Bohr_in_R=bohr,
        Hartree_in_E=1.0,
    )
    both = directed.get_E_coul_a(
        positions,
        charges,
        idx_i=torch.tensor([0, 1]),
        idx_j=torch.tensor([1, 0]),
        Dij_lr=torch.tensor([2.0, 2.0]),
    )
    assert torch.allclose(both.sum(), atomic.sum(), atol=1e-6)

    # A second molecule is not coupled by the vacuum pair list.
    separated = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 30.0, 0.0], [2.0, 30.0, 0.0]]
    )
    copied = charges.repeat(2, 1)
    batch = torch.tensor([0, 0, 1, 1])
    joined = layer.get_E_coul_a(separated, copied, batch_seg=batch)
    assert torch.allclose(joined[:2], atomic, atol=1e-6)
    assert torch.allclose(joined[2:], atomic, atol=1e-6)


def test_reaction_field_kernel_and_cutoff():
    from enerzyme.models.layers.electrostatics import WeightedCoulombEnergyLayer, coulomb_kernel

    cutoff = 5.0
    epsilon = 78.3
    distance = torch.tensor(2.0)
    kernel = coulomb_kernel(distance, cutoff, epsilon, reaction_field=True)
    k_rf = (1.0 / cutoff**3) * (epsilon - 1.0) / (2.0 * epsilon + 1.0)
    c_rf = (1.0 / cutoff) * (3.0 * epsilon) / (2.0 * epsilon + 1.0)
    assert torch.allclose(kernel, 1.0 / distance + k_rf * distance**2 - c_rf)

    layer = WeightedCoulombEnergyLayer(
        cutoff=cutoff,
        reaction_field=True,
        epsilon_solvent=epsilon,
        layer_weights=[1.0],
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    charges = torch.ones(3, 1)
    atomic = layer.get_E_coul_a(positions, charges)
    # The 9 Å pair is outside the cutoff. Only the 2 Å pair remains.
    bare = WeightedCoulombEnergyLayer(layer_weights=[1.0], reaction_field=False)
    near = bare.get_E_coul_a(positions[:2], charges[:2])
    assert atomic[2].abs() < 1e-8
    assert not torch.allclose(atomic[:2].sum(), near.sum(), atol=1e-6)
    try:
        WeightedCoulombEnergyLayer(reaction_field=True)
    except ValueError as exc:
        assert "cutoff" in str(exc)
    else:
        raise AssertionError("expected reaction_field without cutoff to fail")


def test_charge_conditioned_interaction_uses_both_endpoints():
    from enerzyme.models.tensornet.interaction import (
        TensorNet2Interaction,
        scalar_edge_gates,
        tensor_message_update,
    )

    torch.manual_seed(0)
    n, features, num_rbf, q_dim = 4, 3, 2, 2
    interaction = TensorNet2Interaction(features, num_rbf, q_dim, "O(3)")
    assert interaction.linears_scalar[0].in_features == num_rbf + 2 * q_dim
    features_t = torch.randn(n, 3, 3, features)
    charges = torch.randn(n, q_dim)
    idx_i = torch.tensor([0, 1, 2, 3])
    idx_j = torch.tensor([1, 2, 3, 0])
    rbf = torch.rand(4, num_rbf)
    window = torch.rand(4)
    updated = interaction(features_t, charges, idx_i, idx_j, rbf, window)
    edge = scalar_edge_gates(
        torch.cat((rbf, charges[idx_i], charges[idx_j]), dim=-1),
        interaction.linears_scalar,
        window,
        features,
    )
    reference = tensor_message_update(
        features_t,
        edge,
        idx_i,
        idx_j,
        interaction.linears_tensor,
        "O(3)",
        torch.ones(n),
    )
    assert torch.allclose(updated, reference, atol=1e-6)
    swapped = charges.clone()
    swapped[0] = charges[1]
    swapped[1] = charges[0]
    assert not torch.allclose(
        updated,
        interaction(features_t, swapped, idx_i, idx_j, rbf, window),
        atol=1e-5,
    )


def test_standard_zbl_matches_fixed_pair_formula():
    from ase.data import covalent_radii

    from enerzyme.models.mp.basis.envelope import polynomial_envelope
    from enerzyme.models.layers.screened_zbl import StandardZBLRepulsionEnergyLayer

    bohr = 0.5291772105638411
    hartree = 2.0
    layer = StandardZBLRepulsionEnergyLayer(
        max_Za=8, Bohr_in_R=bohr, Hartree_in_E=hartree, envelope_p=6
    )
    distance = torch.tensor([0.4, 0.4])
    species = torch.tensor([1, 8])
    atomic = layer.get_E_zbl_a(
        species,
        distance,
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )
    screening = 0.4543 * bohr / (1.0**0.3 + 8.0**0.3)
    reduced = 0.4 / screening
    phi = (
        0.1818 * math.exp(-3.2 * reduced)
        + 0.5099 * math.exp(-0.9423 * reduced)
        + 0.2802 * math.exp(-0.4029 * reduced)
        + 0.02817 * math.exp(-0.2016 * reduced)
    )
    radius_sum = float(covalent_radii[1] + covalent_radii[8])
    envelope = polynomial_envelope(torch.tensor(0.4), radius_sum, 6)
    full_pair = bohr * hartree * 1.0 * 8.0 / 0.4 * phi * float(envelope)
    assert torch.allclose(atomic.sum(), torch.tensor(full_pair), atol=1e-5, rtol=1e-5)
    assert torch.allclose(atomic[0], atomic[1], atol=1e-6)
