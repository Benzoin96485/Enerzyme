from typing import Optional, Sequence, Tuple, Union
import numpy as np


def full_neighbor_list(N):
    if N == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    idx = np.indices((N, N))
    idx_i = np.concatenate(idx[0][:, :N-1])
    idx_j = []
    for i in range(N):
        idx_int = idx[1][i]
        idx_int = idx_int[idx_int != i]
        idx_j.append(idx_int)
    idx_j = np.concatenate(idx_j)
    return idx_i, idx_j


def _backend_neighbor_list(
    positions: np.ndarray, cell: np.ndarray, pbc: Sequence[bool], cutoff: float
):
    """``(idx_i, idx_j, unit_shifts)`` via a linked-cell-list search, O(N) on average.

    Prefers ``matscipy.neighbours.neighbour_list`` (C-optimized cell list, the backend
    MACE (github.com/ACEsuit/mace) uses). Falls back to the pure-Python cell-list
    implementation ``ase.neighborlist.primitive_neighbor_list`` when matscipy is not
    installed -- ASE is already a hard Enerzyme dependency, so this fallback needs no
    new package. Both are linked-cell algorithms: expected O(N) time (strictly better
    than the O(N log N) requirement) when the cutoff is small relative to the cell,
    not the O(N^2) brute-force scan that ``full_neighbor_list`` performs.
    """
    try:
        from matscipy.neighbours import neighbour_list as _neighbour_list
        idx_i, idx_j, unit_shifts = _neighbour_list(
            quantities="ijS", pbc=pbc, cell=cell, positions=positions, cutoff=cutoff
        )
    except ImportError:
        from ase.neighborlist import primitive_neighbor_list as _neighbour_list
        idx_i, idx_j, unit_shifts = _neighbour_list(
            "ijS", pbc=pbc, cell=cell, positions=positions, cutoff=cutoff,
            self_interaction=False,
        )
    return (
        np.asarray(idx_i, dtype=np.int64),
        np.asarray(idx_j, dtype=np.int64),
        np.asarray(unit_shifts, dtype=np.float64).reshape(-1, 3),
    )


def cutoff_neighbor_list(
    positions: np.ndarray,
    cutoff: float,
    cell: Optional[np.ndarray] = None,
    pbc: Optional[Union[bool, Sequence[bool]]] = None,
    self_interaction: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-range, cutoff-limited, periodic-boundary-aware neighbor list.

    Builds every pair ``(i, j)`` (including periodic images of ``j``) with
    ``|Ra[j] + offsets - Ra[i]| <= cutoff``, via a linked-cell-list search (see
    :func:`_backend_neighbor_list`) instead of ``full_neighbor_list``'s O(N^2)
    all-pairs scan. This is the datahub-level analogue of MACE's
    ``mace.data.neighborhood.get_neighborhood`` (github.com/ACEsuit/mace) and
    fairchem's ``radius_graph_pbc``/``get_pbc_distances``
    (github.com/facebookresearch/fairchem): non-periodic axes are padded with an
    oversized cell so they never spuriously wrap, and the returned ``offsets`` use
    the same additive convention as Enerzyme's own
    :class:`enerzyme.models.layers.geometry.DistanceLayer` (``Rj + offsets``).

    Parameters
    ----------
    positions: ``(N, 3)`` Cartesian atomic positions.

    cutoff: neighbor search radius.

    cell: ``(3, 3)`` lattice vectors (rows). Each periodic axis needs an
        independent, nonzero vector. Open axes can be zero.

    pbc: one boolean for all axes or one boolean per cell axis. Defaults to
        ``(False, False, False)`` (open boundary; ``cutoff_neighbor_list`` is then
        just an O(N) short-range radius graph for an ordinary finite molecule).

    self_interaction: include the zero-shift self-edge of each atom. Bonds
        to an atom's nonzero periodic images are kept in both modes.

    Returns
    -------
    idx_i, idx_j: ``int64`` arrays of pair indices, shape ``(N_pair,)``.

    offsets: ``float64`` array of Cartesian shift vectors, shape ``(N_pair, 3)``,
        such that ``positions[idx_j] + offsets - positions[idx_i]`` are the true
        (possibly periodic-image) displacement vectors -- i.e. exactly the
        ``offsets`` input expected by :class:`DistanceLayer`.
    """
    from ase.geometry import complete_cell

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape == (0,):
        positions = positions.reshape(0, 3)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("positions must be a finite array of shape (N, 3)")
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")

    periodic = np.asarray(False if pbc is None else pbc, dtype=bool)
    if periodic.ndim == 0:
        periodic = np.full(3, periodic, dtype=bool)
    if periodic.shape != (3,):
        raise ValueError("pbc must be a scalar or three flags, one per lattice vector")
    lattice = np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("cell must be a finite array of shape (3, 3)")
    if periodic.any() and np.linalg.matrix_rank(lattice[periodic]) != int(periodic.sum()):
        raise ValueError("periodic axes require linearly independent, nonzero cell vectors")

    if len(positions) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty((0, 3))

    search_positions = positions - positions[0]
    search_cell = lattice.copy()
    if not periodic.all():
        # Extend the periodic subspace with orthonormal directions. Cartesian
        # axis vectors can be parallel to a rotated wire or slab.
        search_cell[~periodic] = 0
        directions = complete_cell(search_cell)[~periodic]
        heights = search_positions @ directions.T
        search_positions -= (heights.min(axis=0) - cutoff) @ directions
        search_cell[~periodic] = (np.ptp(heights, axis=0) + 2 * cutoff)[:, None] * directions

    # Both search backends use a strict radius. Include the boundary before
    # applying the advertised Cartesian distance criterion on the original cell.
    idx_i, idx_j, unit_shifts = _backend_neighbor_list(
        search_positions, search_cell, periodic, np.nextafter(cutoff, np.inf)
    )
    offsets = unit_shifts @ lattice
    distance = np.linalg.norm((positions[idx_j] - positions[idx_i]) + offsets, axis=1)
    keep = distance <= cutoff
    idx_i, idx_j, offsets = idx_i[keep], idx_j[keep], offsets[keep]
    if self_interaction:
        atoms = np.arange(len(positions), dtype=np.int64)
        idx_i = np.concatenate((idx_i, atoms))
        idx_j = np.concatenate((idx_j, atoms))
        offsets = np.concatenate((offsets, np.zeros((len(atoms), 3))))
    return idx_i, idx_j, offsets
