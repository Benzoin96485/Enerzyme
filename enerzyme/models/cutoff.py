import torch
from torch import Tensor


def polynomial_cutoff(x: Tensor, cutoff: float) -> Tensor:
    """
    Polynomial cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff with sufficiently many smooth derivatives [1]. 
    For x >= cutoff, f(x) = 0. 

    Params:
    -----

    x: Only positive x should be used as input.

    cutoff: A positive cutoff radius.
    
    References:
    -----
    [1] Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003.
    """
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
    x_ = x_ / cutoff
    x3 = x_ ** 3
    x4 = x3 * x_
    x5 = x4 * x_
    return torch.where(x < cutoff, 1 - 6 * x5 + 15 * x4 - 10 * x3, zeros)


def bump_cutoff(x: Tensor, cutoff: float) -> Tensor:
    """
    Smooth cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff [1]. 
    For x >= cutoff, f(x) = 0. 

    Params:
    -----

    x: Only positive x should be used as input.

    cutoff: A positive cutoff radius.
    
    References:
    -----
    [1] Nat. Commun., 2021, 12, 7273.
    """
    zeros = torch.zeros_like(x)
    x_ = torch.where(x < cutoff, x, zeros)  # prevent nan in backprop
    return torch.where(
        x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
    )


CUTOFF_REGISTER = {
    "polynomial": polynomial_cutoff,
    "bump": bump_cutoff
}