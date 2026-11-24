from typing import Union, Optional
import numpy as np
import torch
from torch import Tensor


def softplus_inverse(x: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus). 
    
    The indirect implementation is for numerical stability.
    """
    if not isinstance(x, Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def gather_nd(params: Tensor, indices: Tensor) -> Tensor:
    """Index the leading axes of a tensor with coordinate tuples on its device.

    The output shape is `indices.shape[:-1] + params.shape[indices.shape[-1]:]`.
    """
    width = indices.shape[-1]
    if width > params.ndim:
        raise ValueError(f"index width {width} exceeds the rank of params ({params.ndim})")
    if width == 0:
        return params.expand(tuple(indices.shape[:-1]) + tuple(params.shape)).contiguous()
    return params[indices.unbind(dim=-1)].contiguous()
