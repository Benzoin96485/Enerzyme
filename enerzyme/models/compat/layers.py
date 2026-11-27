"""Layers that ``main`` YAML files name but the current layer registry does not.

:func:`enerzyme.models.ff.build_model` looks names up here after
:mod:`enerzyme.models.layers`, so ``main`` model sections build unchanged.
"""

from torch import Tensor

from ..layers import BaseFFLayer
from ..mp.basis.radial import gaussian_expansion, gaussian_grid


class GaussianSmearing(BaseFFLayer):
    """``main``'s SchNet smearing ``exp(coeff · (d - offset)²)`` without a cutoff envelope.

    ``coeff`` stays a Python float derived from the float32 ``offset`` grid,
    as on ``main``, so float64 runs reproduce ``main`` bit for bit.
    ``GaussianRBF`` with ``flavor: SchNet`` stores it as a float32 buffer.
    """

    def __init__(self, num_rbf: int, cutoff_sr: float, cuton: float = 0.0) -> None:
        super().__init__(input_fields={"Dij_sr"}, output_fields={"rbf"})
        offset, self.coeff = gaussian_grid(cuton, cutoff_sr, num_rbf, 1.0)
        self.register_buffer("offset", offset)

    def get_rbf(self, Dij_sr: Tensor) -> Tensor:
        return gaussian_expansion(Dij_sr, self.offset, self.coeff)


LEGACY_LAYERS = {"GaussianSmearing": GaussianSmearing}
