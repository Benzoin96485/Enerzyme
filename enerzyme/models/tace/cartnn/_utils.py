import torch

def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T
