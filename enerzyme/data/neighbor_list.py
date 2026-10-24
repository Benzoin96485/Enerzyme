from typing import Optional, Sequence, Tuple
import numpy as np


def full_neighbor_list(N):
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
    positions: np.ndarray, cell: np.ndarray, pbc: Tuple[bool, bool, bool], cutoff: float
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
            self_interaction=True,
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
    pbc: Optional[Sequence[bool]] = None,
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

    Params
    -----
    positions: ``(N, 3)`` Cartesian atomic positions.

    cutoff: neighbor search radius.

    cell: ``(3, 3)`` lattice vectors (rows). Ignored along non-periodic axes.
        Defaults to the identity (irrelevant when ``pbc`` is all ``False``).

    pbc: length-3 sequence of bools, one per cell axis. Defaults to
        ``(False, False, False)`` (open boundary; ``cutoff_neighbor_list`` is then
        just an O(N) short-range radius graph for an ordinary finite molecule).

    self_interaction: keep an atom's bond to its own periodic image (``i == j``
        with a non-zero shift). Trivial self-edges (``i == j`` and zero shift) are
        always dropped.

    Returns
    -----
    idx_i, idx_j: ``int64`` arrays of pair indices, shape ``(N_pair,)``.

    offsets: ``float64`` array of Cartesian shift vectors, shape ``(N_pair, 3)``,
        such that ``positions[idx_j] + offsets - positions[idx_i]`` are the true
        (possibly periodic-image) displacement vectors -- i.e. exactly the
        ``offsets`` input expected by :class:`DistanceLayer`.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, np.zeros((0, 3), dtype=np.float64)

    if pbc is None:
        pbc = (False, False, False)
    pbc = tuple(bool(p) for p in np.asarray(pbc).reshape(-1)[:3])
    if len(pbc) < 3:
        pbc = pbc + (False,) * (3 - len(pbc))

    if cell is None:
        cell = np.eye(3, dtype=np.float64)
    cell = np.array(cell, dtype=np.float64).reshape(3, 3)

    # Pad non-periodic axes with an oversized cell so they never wrap into a
    # spurious periodic image, mirroring MACE's get_neighborhood.
    if not all(pbc):
        max_extent = float(np.max(np.abs(positions))) + 1.0 if positions.size else 1.0
        pad = max(5.0 * cutoff * max_extent, 5.0 * cutoff)
        identity = np.eye(3, dtype=np.float64)
        for axis in range(3):
            if not pbc[axis]:
                cell[axis, :] = pad * identity[axis, :]

    idx_i, idx_j, unit_shifts = _backend_neighbor_list(positions, cell, pbc, cutoff)

    if not self_interaction:
        trivial_self_edge = (idx_i == idx_j) & np.all(unit_shifts == 0, axis=-1)
        keep = ~trivial_self_edge
        idx_i, idx_j, unit_shifts = idx_i[keep], idx_j[keep], unit_shifts[keep]

    offsets = unit_shifts @ cell
    return idx_i, idx_j, offsets

