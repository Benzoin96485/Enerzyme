from typing import Literal, Callable
import torch
from torch import Tensor


def scale(cutoff_fn: Callable[[Tensor, Tensor, Tensor], Tensor]):
    def scaled_transition_fn(x: Tensor, cutoff: float, cuton: float=0) -> Tensor:
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        x_ = (x - cuton) / (cutoff - cuton)
        kernel = cutoff_fn(x_, zeros, ones)
        return torch.where(x_ > 0, torch.where(x_ < 1, kernel, zeros), ones)
    return scaled_transition_fn


@scale
def polynomial_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
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
def bump_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
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
    x_ = torch.where((0 < x_) & (x_ < 1), x_, zeros)
    x2 = x_ ** 2
    return torch.exp(-x2 / (1 - x2))


def _smooth_transition(x_: Tensor, ones: Tensor) -> Tensor:
    return torch.exp(-1 / torch.where(x_ > 0, x_, ones))


@scale
def smooth_transition(x_: Tensor, zeros: Tensor, ones: Tensor) -> Tensor:
    fp = _smooth_transition(x_, ones)
    fm = _smooth_transition(1 - x_, ones)
    return fm / (fp + fm)


CUTOFF_REGISTER = {
    "polynomial": polynomial_transition,
    "bump": bump_transition,
    "smooth": smooth_transition
}
CUTOFF_KEY_TYPE = Literal["polynomial", "bump", "smooth"]