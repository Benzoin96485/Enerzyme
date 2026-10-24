"""Correctness tests for the O(N) cutoff/PBC neighbor list builder.

``cutoff_neighbor_list`` (enerzyme/data/neighbor_list.py) is the datahub-level
analogue of MACE's ``mace.data.neighborhood.get_neighborhood`` and fairchem's
``radius_graph_pbc``/``get_pbc_distances``: a linked-cell-list search (O(N), not
``full_neighbor_list``'s O(N^2)) that also handles periodic boundary conditions
via cartesian ``offsets`` (the same additive convention as
``enerzyme.models.layers.geometry.DistanceLayer``).
"""
import numpy as np
import pytest

from enerzyme.data.neighbor_list import cutoff_neighbor_list, full_neighbor_list


def _brute_force_periodic_pairs(positions, cutoff, cell, pbc, nrep=3):
    """Reference triple-loop periodic-image neighbor search (O(N^2 * images))."""
    N = len(positions)
    ranges = [range(-nrep, nrep + 1) if p else range(0, 1) for p in pbc]
    pairs = []
    for a in ranges[0]:
        for b in ranges[1]:
            for c in ranges[2]:
                shift = np.array([a, b, c], dtype=float) @ cell
                for i in range(N):
                    for j in range(N):
                        if a == 0 and b == 0 and c == 0 and i == j:
                            continue
                        d = np.linalg.norm(positions[j] + shift - positions[i])
                        if d <= cutoff:
                            pairs.append((i, j, tuple(shift.round(6))))
    return set(pairs)


def _pairs_with_shift(idx_i, idx_j, offsets):
    return set(
        (int(i), int(j), tuple(np.round(o, 6)))
        for i, j, o in zip(idx_i, idx_j, offsets)
    )


def test_open_boundary_matches_full_neighbor_list_filtered_by_distance():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-2, 2, size=(10, 3))
    cutoff = 2.2
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=cutoff)
    assert offsets.shape == (len(idx_i), 3)
    d = np.linalg.norm(pos[idx_j] + offsets - pos[idx_i], axis=-1)
    assert np.all(d <= cutoff + 1e-8)

    fi, fj = full_neighbor_list(len(pos))
    fd = np.linalg.norm(pos[fj] - pos[fi], axis=-1)
    ref_pairs = set(zip(fi[fd <= cutoff].tolist(), fj[fd <= cutoff].tolist()))
    got_pairs = set(zip(idx_i.tolist(), idx_j.tolist()))
    assert got_pairs == ref_pairs
    assert np.all(offsets == 0)  # no cell/pbc given -> no periodic shifts


def test_open_boundary_offsets_are_zero_and_no_self_edges():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=5.0)
    assert not np.any(idx_i == idx_j)
    assert np.all(offsets == 0)


def test_simple_cubic_pbc_known_coordination_number():
    # Single atom on a simple cubic lattice (a=2): first shell has 6 neighbors
    # (images at +-a along each axis); first+second shell has 6+12=18.
    pos = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3) * 2.0
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=2.1, cell=cell, pbc=(True, True, True))
    assert len(idx_i) == 6
    d = np.linalg.norm(offsets, axis=-1)  # Ra is at the origin
    assert np.allclose(np.sort(d), 2.0)

    idx_i2, idx_j2, offsets2 = cutoff_neighbor_list(pos, cutoff=2.9, cell=cell, pbc=(True, True, True))
    assert len(idx_i2) == 18


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
def test_pbc_matches_bruteforce_periodic_image_search(pbc):
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.2, 1.8, size=(5, 3))
    cell = np.array([[2.0, 0.0, 0.0], [0.3, 2.0, 0.0], [0.1, 0.2, 2.2]])
    cutoff = 1.6
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=cutoff, cell=cell, pbc=pbc)
    got = _pairs_with_shift(idx_i, idx_j, offsets)
    ref = _brute_force_periodic_pairs(pos, cutoff, cell, pbc, nrep=4)
    assert got == ref, (got - ref, ref - got)


def test_offsets_round_trip_to_distances():
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 3, size=(6, 3))
    cell = np.eye(3) * 3.0
    cutoff = 1.4
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=cutoff, cell=cell, pbc=(True, True, True))
    d = np.linalg.norm(pos[idx_j] + offsets - pos[idx_i], axis=-1)
    assert np.all(d <= cutoff + 1e-8)
    assert np.all(d > 1e-8)


def test_matscipy_and_ase_backends_agree():
    pytest.importorskip("matscipy")
    import builtins

    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 4, size=(9, 3))
    cell = np.array([[3.0, 0.0, 0.0], [0.4, 3.0, 0.0], [0.2, 0.1, 3.0]])
    cutoff = 1.8

    idx_i_ms, idx_j_ms, off_ms = cutoff_neighbor_list(pos, cutoff, cell=cell, pbc=(True, True, True))

    real_import = builtins.__import__

    def blocked_import(name, *a, **k):
        if name == "matscipy" or name.startswith("matscipy."):
            raise ImportError("blocked for backend-agreement test")
        return real_import(name, *a, **k)

    builtins.__import__ = blocked_import
    try:
        idx_i_ase, idx_j_ase, off_ase = cutoff_neighbor_list(
            pos, cutoff, cell=cell, pbc=(True, True, True)
        )
    finally:
        builtins.__import__ = real_import

    assert _pairs_with_shift(idx_i_ms, idx_j_ms, off_ms) == _pairs_with_shift(
        idx_i_ase, idx_j_ase, off_ase
    )


def test_empty_system():
    pos = np.zeros((0, 3))
    idx_i, idx_j, offsets = cutoff_neighbor_list(pos, cutoff=1.0)
    assert len(idx_i) == 0 and len(idx_j) == 0
    assert offsets.shape == (0, 3)
