"""Correctness tests for EwaldElectrostaticEnergyLayer (periodic-boundary electrostatics).

Real space + reciprocal space + self-energy follows SpookyNet's point-charge Ewald
sum (github.com/OUnke/SpookyNet), generalized from an orthorhombic to an arbitrary
triclinic cell using LES's reciprocal-lattice-vector construction
(github.com/ChengUCB/les). Like every standard Ewald implementation used in MD/ML
force fields (SpookyNet, LAMMPS, GROMACS, LES, ...), this layer computes the
"tin-foil" (conducting) boundary-condition energy, i.e. it does *not* include the
shape-dependent dipole "surface term" ``(2*pi/(3V))*|sum_i q_i r_i|^2`` that a naive
growing-cubic-shell direct sum converges to for systems with a net dipole moment
(de Leeuw, Perram & Smith, Proc. R. Soc. Lond. A 1980, 373, 27-56). The Madelung-
constant test below sidesteps this ambiguity entirely (a rocksalt lattice has zero
net dipole per periodic cell by symmetry); the dipolar-system test explicitly adds
the surface-term correction before comparing to a brute-force replica sum.
"""
import math
import numpy as np
import pytest
import torch

from enerzyme.data.neighbor_list import cutoff_neighbor_list
from enerzyme.models.layers.electrostatics import EwaldElectrostaticEnergyLayer

# positions/charges/cell/offsets are plain float64 numpy arrays throughout this
# file; torch.tensor(...) below preserves that dtype without needing a global
# torch.set_default_dtype (which would leak into other test files in the same
# pytest session).


def _build_inputs(positions, charges, cell, cutoff_real, requires_grad=False, device="cpu"):
    idx_i, idx_j, offsets = cutoff_neighbor_list(
        positions, cutoff_real, cell=cell, pbc=(True, True, True)
    )
    Ra = torch.tensor(positions, dtype=torch.float64, requires_grad=requires_grad, device=device)
    Qa = torch.tensor(charges, dtype=torch.float64, device=device)
    idx_i_t = torch.tensor(idx_i, dtype=torch.long, device=device)
    idx_j_t = torch.tensor(idx_j, dtype=torch.long, device=device)
    offsets_t = torch.tensor(offsets, dtype=torch.float64, device=device)
    Dij = torch.norm(Ra[idx_j_t] + offsets_t - Ra[idx_i_t], dim=-1)
    cell_t = torch.tensor(cell, dtype=torch.float64, device=device).unsqueeze(0)
    return Ra, Qa, idx_i_t, idx_j_t, Dij, cell_t


def _rocksalt(n=2, d=1.0):
    positions, charges = [], []
    for i in range(2 * n):
        for j in range(2 * n):
            for k in range(2 * n):
                positions.append([i * d, j * d, k * d])
                charges.append(1.0 if (i + j + k) % 2 == 0 else -1.0)
    return np.array(positions), np.array(charges), np.eye(3) * (2 * n * d)


def test_madelung_constant_rocksalt():
    # NaCl (rocksalt) Madelung constant: E_per_ion_pair = -alpha_M * ke*q^2/d,
    # alpha_M = 1.7475645946 (well-established literature value). ke=q=d=1 here.
    positions, charges, cell = _rocksalt(n=2, d=1.0)
    cutoff_real = 6.0
    Ra, Qa, idx_i, idx_j, Dij, cell_t = _build_inputs(positions, charges, cell, cutoff_real)
    batch_seg = torch.zeros(len(positions), dtype=torch.long)

    layer = EwaldElectrostaticEnergyLayer(
        cutoff_real=cutoff_real, k_cutoff=6.0, Bohr_in_R=1.0, Hartree_in_E=1.0
    )
    E_ele_a = layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=cell_t, batch_seg=batch_seg)
    E_per_ion_pair = 2.0 * (E_ele_a.sum() / len(positions)).item()
    assert math.isclose(E_per_ion_pair, -1.7475645946, rel_tol=0, abs_tol=1e-4)


def test_dipolar_system_matches_bruteforce_plus_surface_term():
    rng = np.random.default_rng(0)
    N, L = 6, 6.0
    positions = rng.uniform(0, L, size=(N, 3))
    charges = rng.uniform(-1, 1, size=N)
    charges -= charges.mean()  # exactly neutral
    cell = np.eye(3) * L
    cutoff_real = 5.9

    Ra, Qa, idx_i, idx_j, Dij, cell_t = _build_inputs(positions, charges, cell, cutoff_real)
    batch_seg = torch.zeros(N, dtype=torch.long)
    layer = EwaldElectrostaticEnergyLayer(
        cutoff_real=cutoff_real, k_cutoff=8.0, Bohr_in_R=1.0, Hartree_in_E=1.0
    )
    E_tinfoil = layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=cell_t, batch_seg=batch_seg).sum().item()

    # Vacuum-boundary surface-term correction (de Leeuw-Perram-Smith).
    M = (charges[:, None] * positions).sum(axis=0)
    V = L ** 3
    E_surface = (2.0 * math.pi / (3.0 * V)) * float(np.sum(M ** 2))

    # Reference: direct sum over growing cubic replicas of the same cell (vectorized).
    nrep = 10
    rng_idx = np.arange(-nrep, nrep + 1)
    shifts = np.stack(np.meshgrid(rng_idx, rng_idx, rng_idx, indexing="ij"), axis=-1).reshape(-1, 3) * L
    diff = positions[None, :, None, :] - positions[None, None, :, :] - shifts[:, None, None, :]
    r = np.linalg.norm(diff, axis=-1)
    inv_r = np.where(r > 1e-9, 1.0 / np.where(r > 1e-9, r, 1.0), 0.0)
    qq = charges[:, None] * charges[None, :]
    E_bruteforce = 0.5 * float(np.sum(qq[None, :, :] * inv_r))

    assert math.isclose(E_tinfoil + E_surface, E_bruteforce, rel_tol=0, abs_tol=2e-3)


def test_force_matches_finite_difference():
    rng = np.random.default_rng(1)
    N, L = 5, 5.0
    positions = rng.uniform(0, L, size=(N, 3))
    charges = rng.uniform(-1, 1, size=N)
    charges -= charges.mean()
    cell = np.eye(3) * L
    cutoff_real = 4.9

    Ra, Qa, idx_i, idx_j, Dij, cell_t = _build_inputs(
        positions, charges, cell, cutoff_real, requires_grad=True
    )
    batch_seg = torch.zeros(N, dtype=torch.long)
    layer = EwaldElectrostaticEnergyLayer(
        cutoff_real=cutoff_real, k_cutoff=8.0, Bohr_in_R=1.0, Hartree_in_E=1.0
    )
    E = layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=cell_t, batch_seg=batch_seg).sum()
    E.backward()
    grad_analytic = Ra.grad.detach().numpy().copy()

    def energy_at(pos):
        Ra_, Qa_, idx_i_, idx_j_, Dij_, cell_ = _build_inputs(pos, charges, cell, cutoff_real)
        return layer.get_E_ele_a(Ra_, Qa_, idx_i_, idx_j_, Dij_, cell=cell_, batch_seg=batch_seg).sum().item()

    eps = 1e-5
    grad_fd = np.zeros_like(positions)
    for i in range(N):
        for d in range(3):
            pp, pm = positions.copy(), positions.copy()
            pp[i, d] += eps
            pm[i, d] -= eps
            grad_fd[i, d] = (energy_at(pp) - energy_at(pm)) / (2 * eps)

    assert np.max(np.abs(grad_analytic - grad_fd)) < 1e-6


def test_batching_is_additive_over_independent_graphs():
    rng = np.random.default_rng(2)

    def make_system(n, L, seed):
        r = np.random.default_rng(seed)
        pos = r.uniform(0, L, size=(n, 3))
        q = r.uniform(-1, 1, size=n)
        q -= q.mean()
        return pos, q, np.eye(3) * L

    cutoff_real = 3.9
    layer = EwaldElectrostaticEnergyLayer(
        cutoff_real=cutoff_real, k_cutoff=8.0, Bohr_in_R=1.0, Hartree_in_E=1.0
    )

    pos1, q1, cell1 = make_system(4, 4.0, 10)
    pos2, q2, cell2 = make_system(5, 5.0, 11)

    def single_energy(pos, q, cell):
        Ra, Qa, idx_i, idx_j, Dij, cell_t = _build_inputs(pos, q, cell, cutoff_real)
        batch_seg = torch.zeros(len(pos), dtype=torch.long)
        return layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=cell_t, batch_seg=batch_seg).sum().item()

    E1 = single_energy(pos1, q1, cell1)
    E2 = single_energy(pos2, q2, cell2)

    # Batch the two independent, differently-celled graphs together.
    idx_i1, idx_j1, off1 = cutoff_neighbor_list(pos1, cutoff_real, cell=cell1, pbc=(True, True, True))
    idx_i2, idx_j2, off2 = cutoff_neighbor_list(pos2, cutoff_real, cell=cell2, pbc=(True, True, True))
    n1 = len(pos1)
    Ra = torch.tensor(np.concatenate([pos1, pos2], axis=0), dtype=torch.float64)
    Qa = torch.tensor(np.concatenate([q1, q2], axis=0), dtype=torch.float64)
    idx_i = torch.tensor(np.concatenate([idx_i1, idx_i2 + n1]), dtype=torch.long)
    idx_j = torch.tensor(np.concatenate([idx_j1, idx_j2 + n1]), dtype=torch.long)
    offsets = torch.tensor(np.concatenate([off1, off2], axis=0), dtype=torch.float64)
    Dij = torch.norm(Ra[idx_j] + offsets - Ra[idx_i], dim=-1)
    cell_t = torch.tensor(np.stack([cell1, cell2], axis=0), dtype=torch.float64)
    batch_seg = torch.cat([torch.zeros(n1, dtype=torch.long), torch.ones(len(pos2), dtype=torch.long)])

    E_batched = layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=cell_t, batch_seg=batch_seg)
    assert math.isclose(E_batched[batch_seg == 0].sum().item(), E1, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(E_batched[batch_seg == 1].sum().item(), E2, rel_tol=1e-9, abs_tol=1e-9)


def test_requires_cell():
    positions = np.random.default_rng(0).uniform(0, 3, size=(4, 3))
    charges = np.array([1.0, -1.0, 1.0, -1.0])
    Ra = torch.tensor(positions, dtype=torch.float64)
    Qa = torch.tensor(charges, dtype=torch.float64)
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    Dij = torch.norm(Ra[idx_j] - Ra[idx_i], dim=-1)
    layer = EwaldElectrostaticEnergyLayer(cutoff_real=3.0, k_cutoff=5.0)
    try:
        layer.get_E_ele_a(Ra, Qa, idx_i, idx_j, Dij, cell=None)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when cell is None")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"))])
def test_skewed_and_left_handed_representations_of_the_same_lattice(device):
    # An integer shear with determinant ±1 preserves all translations.
    positions = np.array([[.1, .3, .4], [1.25, 1.08, .9]])
    charges = np.array([1., -1.])
    cell = np.diag([3.2, 3.7, 4.1])
    layer = EwaldElectrostaticEnergyLayer(4.5, 7.1, Bohr_in_R=1.)
    positions_t, charges_t, i, j, distances, _ = _build_inputs(
        positions, charges, cell, 4.5, requires_grad=True, device=device)
    energies_forces = []
    for transform in (np.eye(3), np.array([[1, 4, 0], [0, 1, 0], [0, 0, -1]])):
        lattice = torch.tensor(transform @ cell, dtype=positions_t.dtype)
        # A constant input lattice can stay on the host while positions live on CUDA.
        energy = layer.get_E_ele_a(positions_t, charges_t, i, j, distances, cell=lattice)
        force = -torch.autograd.grad(energy.sum(), positions_t, retain_graph=True)[0]
        energies_forces.append((energy, force))
    torch.testing.assert_close(*energies_forces, atol=1e-12, rtol=1e-12)


def test_charged_periodic_cell_split_invariance_and_ensemble_channels():
    positions = np.array([[.15, .2, .45]])
    cell = np.eye(3) * 3.0
    args = _build_inputs(positions, np.ones(1), cell, cutoff_real=8.)
    # A cubic array of point charges in a uniform neutralizing background.
    # Changing the Ewald split or the charge channel must preserve its Madelung energy.
    charges = torch.tensor([[1., -2.]], dtype=torch.float64, requires_grad=True)
    for alpha in (.75, 1.3):
        layer = EwaldElectrostaticEnergyLayer(8., 13., alpha=alpha, Bohr_in_R=1.)
        scale = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        energy = layer.get_E_ele_a(args[0] * scale, charges, *args[2:4], args[4] * scale, cell=args[-1] * scale)
        expected = -(2.837297479480619 / (2 * 3.)) * charges.square()
        torch.testing.assert_close(energy, expected, atol=3e-11, rtol=3e-11)
        grad, strain = torch.autograd.grad(energy.sum(), (charges, scale))
        torch.testing.assert_close(strain, -expected.sum(), atol=2e-9, rtol=2e-9)
        torch.testing.assert_close(grad, 2 * expected / charges, atol=1e-10, rtol=1e-10)


def test_partial_periodicity_and_singular_cell_are_rejected():
    layer = EwaldElectrostaticEnergyLayer(4., 6.)
    args = _build_inputs(np.array([[.1, .2, .3]]), np.ones(1), np.eye(3) * 4, 4.)
    with pytest.raises(ValueError, match="all three axes"):
        layer.get_E_ele_a(*args[:5], cell=args[-1], pbc=torch.tensor([True, True, False]))
    with pytest.raises(ValueError, match="nonsingular"):
        layer.get_E_ele_a(*args[:5], cell=torch.zeros_like(args[-1]))


def test_external_double_charges_retain_precision_with_single_positions():
    layer = EwaldElectrostaticEnergyLayer(3., 6.)
    positions, charges, i, j, distance, cell = _build_inputs(
        np.array([[.3, .5, .7], [1.1, 1.2, .8]]), np.array([1., -1.]), np.eye(3) * 4, 3.)
    positions, distance, cell = positions.float(), distance.float(), cell.float()
    scale = 1 + 2 ** -30
    reference = layer.get_E_ele_a(positions, charges, i, j, distance, cell=cell)
    actual = layer.get_E_ele_a(positions, charges * scale, i, j, distance, cell=cell)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, reference * scale ** 2, rtol=2e-14, atol=2e-14)
