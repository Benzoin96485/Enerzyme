import operator

import numpy as np


def full_neighbor_list(N):
    """All directed off-diagonal pairs in receiver-major, then sender-major order."""
    N = operator.index(N)
    if N < 0:
        raise ValueError("Atom count must be non-negative")
    atoms = np.arange(N, dtype=np.int64)
    if N == 0:
        return atoms, atoms.copy()
    receivers = np.repeat(atoms, N - 1)
    columns = atoms[:-1]
    # Skip the diagonal in each row without constructing two square index grids.
    senders = columns + (columns >= atoms[:, None])
    return receivers, senders.ravel()
