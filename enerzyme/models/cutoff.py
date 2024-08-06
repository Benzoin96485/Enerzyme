from typing import Literal, Callable
import torch
from torch import Tensor


def scale(cutoff_fn: Callable[[Tensor], Tensor]):
    def scaled_transition_fn(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        x_ = (x - cuton) / (cutoff - cuton)
        x_ = torch.where(x_ > 0, torch.where(x_ < 1, x_, ones), zeros)
        kernel = cutoff_fn(x_)
        return torch.where(x_ > 0, torch.where(x_ < 1, kernel, zeros), ones)
    return scaled_transition_fn


@scale
def polynomial_transition(x_: Tensor) -> Tensor:
    """
    Polynomial cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = 1 with sufficiently many smooth derivatives [1]. 
    For x >= 0, f(x) = 0. 
    For x <= 1, f(x) = 1. 

    Params:
    -----

    x: Only 0<=x<=1 should be used as input.
    
    References:
    -----
    [1] Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003.
    """
    x3 = x_ ** 3
    x4 = x3 * x_
    x5 = x4 * x_
    return 1 - 6 * x5 + 15 * x4 - 10 * x3


@scale
def bump_transition(x_: Tensor) -> Tensor:
    """
    Smooth cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
    from x = 0 to x = cutoff [1]. 
    For x >= cutoff, f(x) = 0. 

    Params:
    -----

    x: Only 0<=x<=1 should be used as input.
    
    References:
    -----
    [1] Nat. Commun., 2021, 12, 7273.
    """
    return torch.exp(-(x_ ** 2) / ((1 - x_) * (1 + x_)))


@scale
def smooth_transition(x_: Tensor) -> Tensor:
    fp = torch.exp(-1 / x_)
    fm = torch.exp(-1 / (1 - x_))
    return fm / (fp + fm)
# def polynomial_cutoff(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
#     """
#     Polynomial cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
#     from x = cuton to x = cutoff with sufficiently many smooth derivatives [1]. 
#     For x >= cutoff, f(x) = 0. 
#     For x <= cutoff, f(x) = 1. 

#     Params:
#     -----

#     x: Only positive x should be used as input.

#     cutoff: A positive cutoff radius.
    
#     References:
#     -----
#     [1] Texturing & Modeling: A Procedural Approach; Morgan Kaufmann: 2003.
#     """
#     zeros = torch.zeros_like(x)
#     ones = torch.ones_like(x)  
#     x_ = (x_ - cuton) / (cutoff - cuton)
#     x_ = torch.where(x_ > cuton, torch.where(x_ < cutoff, x_, ones), zeros) # prevent nan in backprop
#     x3 = x_ ** 3
#     x4 = x3 * x_
#     x5 = x4 * x_
#     kernel = 1 - 6 * x5 + 15 * x4 - 10 * x3
#     return torch.where(x > cuton, torch.where(x < cutoff, kernel, zeros), ones)


# def bump_cutoff(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
#     """
#     Smooth cutoff function that goes from f(x) = 1 to f(x) = 0 in the interval
#     from x = 0 to x = cutoff [1]. 
#     For x >= cutoff, f(x) = 0. 

#     Params:
#     -----

#     x: Only positive x should be used as input.

#     cutoff: A positive cutoff radius.
    
#     References:
#     -----
#     [1] Nat. Commun., 2021, 12, 7273.
#     """
#     zeros = torch.zeros_like(x)
#     ones = torch.ones_like(x)  
#     x_ = (x_ - cuton) / (cutoff - cuton)
#     x_ = torch.where(x_ > cuton, torch.where(x_ < cutoff, x_, ones), zeros) # prevent nan in backprop
#     return torch.where(
#         x < cutoff, torch.exp(-(x_ ** 2) / ((cutoff - x_) * (cutoff + x_))), zeros
#     )


# def smooth_transition(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
#     pass


CUTOFF_REGISTER = {
    "polynomial": polynomial_transition,
    "bump": bump_transition,
    "smooth": smooth_transition
}
CUTOFF_KEY_TYPE = Literal["polynomial", "bump", "smooth"]