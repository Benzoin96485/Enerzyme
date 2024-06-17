import math
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F_

LOG2 = math.log(2.0)

def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    return F_.softplus(x) - LOG2


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus). 
    
    The indirect implementation is for numerical stability.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    implemented at https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/37
    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


def smooth_cutoff_function(x: torch.Tensor, cutoff: float, flavor: Literal["bump", "poly"]="bump") -> torch.Tensor:
    """
    Cutoff function that smoothly goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff. 
    For x >= cutoff, f(x) = 0. 

    Params:
    -----

    x: Only positive x should be used as input.

    cutoff: A positive cutoff radius.

    flavor:
        - "bump" flavor has infinitely many smooth derivatives. 
        - "poly" flavor has sufficiently many smooth derivatives [1]. 
    
    References:
    -----
    [1] Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003.
    """
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
    if flavor == "bump":
        return torch.where(
            x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
        )
    elif flavor == "poly":
        x_ = x_ / cutoff
        x3 = x_ ** 3
        x4 = x3 * x_
        x5 = x4 * x_
        return torch.where(x_ < 1, 1 - 6 * x5 + 15 * x4 - 10 * x3, zeros)