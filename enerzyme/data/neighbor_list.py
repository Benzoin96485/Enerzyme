import numpy as np

def full_neighbor_list(Ns):
    idx_is = []
    idx_js = []
    for N in Ns:
        idx = np.indices((N, N))
        idx_is.append(np.concatenate(idx[0][:, :N-1]))
        idx_j = []
        for i in range(N):
            idx_int = idx[1][i]
            idx_int = idx_int[idx_int != i]
            idx_j.append(idx_int)
        idx_js.append(np.concatenate(idx_j))
    return idx_is, idx_js

