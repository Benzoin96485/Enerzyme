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

