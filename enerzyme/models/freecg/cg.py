"""Group Clebsch–Gordan product on permutation-invariant abstract edges.

Shao et al. (arXiv:2407.02263) apply the tensor product after messages have
been summed onto each atom. The sum is invariant to a permutation of the
neighbors, so each group may own its own weights. The four paths that stay
are the parity-allowed couplings of ``1o`` and ``2e``:

* ``1o ⊗ 1o → 2e``
* ``1o ⊗ 2e → 1o``
* ``2e ⊗ 1o → 1o``
* ``2e ⊗ 2e → 2e``

That is fewer paths than an SO(3) product of the same two degrees, and each
output degree still sees both input degrees. Features use the polynomial
spherical-harmonic order from :func:`enerzyme.models.layers.polynomial_harmonics.polynomial_spherical_harmonics`
(``l = 1`` then ``l = 2``, Cartesian ``(x, y, z)``). That order matches the
e3nn real harmonics, up to one scale per degree, so
:class:`e3nn.o3.FullyConnectedTensorProduct` is equivariant on these tensors.

The channel roll follows the paper: each abstract-edge index increases by
``shuffle_scale * C / G`` and wraps. ``shuffle_scale = 1.5`` is the
"1.5-group shuffle" in Table 5. The released ``freecg_block.py`` rolls the
other way by ``1.5 * num_groups`` channels. Those two offsets agree only for
special widths; this module follows the paper.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import Tensor
from torch.nn import Module, Parameter, init

from ..layers.polynomial_harmonics import angular_dim


ANGULAR_DIM = angular_dim(2)
_L1 = 3
_L2 = 5


def shuffle_offset(channels: int, num_groups: int, shuffle_scale: float) -> int:
    """Channel shift ``round(shuffle_scale * C / G)``.

    The product must be an integer. A scale of 0 leaves the channels in place.
    """
    channels = int(channels)
    groups = int(num_groups)
    if groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if channels < 1 or channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be a positive multiple of num_groups ({groups})"
        )
    scale = float(shuffle_scale)
    if scale < 0:
        raise ValueError(f"shuffle_scale must be >= 0, got {shuffle_scale}")
    raw = scale * (channels // groups)
    offset = int(round(raw))
    if abs(raw - offset) > 1e-6:
        raise ValueError(
            "shuffle_scale * (channels / num_groups) must be an integer, "
            f"got {raw}"
        )
    return offset


def _pack_groups(features: Tensor, num_groups: int, width: int) -> Tensor:
    """``[N, 8, C]`` to ``[N, G, width*8]`` in e3nn ``(channel, m)`` order."""
    n_atoms = int(features.shape[0])
    grouped = features.reshape(n_atoms, ANGULAR_DIM, num_groups, width)
    grouped = grouped.permute(0, 2, 3, 1)
    degree_1 = grouped[..., :_L1].reshape(n_atoms, num_groups, width * _L1)
    degree_2 = grouped[..., _L1:].reshape(n_atoms, num_groups, width * _L2)
    return torch.cat((degree_1, degree_2), dim=-1)


def _unpack_groups(packed: Tensor, num_groups: int, width: int) -> Tensor:
    """Inverse of :func:`_pack_groups`."""
    n_atoms = int(packed.shape[0])
    degree_1 = packed[..., : width * _L1].reshape(n_atoms, num_groups, width, _L1)
    degree_2 = packed[..., width * _L1 :].reshape(n_atoms, num_groups, width, _L2)
    angular = torch.cat((degree_1, degree_2), dim=-1)
    return angular.permute(0, 3, 1, 2).reshape(n_atoms, ANGULAR_DIM, num_groups * width)


class GroupSparseCG(Module):
    """One grouped sparse tensor product, then the abstract-edge roll.

    Both inputs and the output are ``[N, 8, C]``. ``left`` is the first
    factor (the aggregated abstract edge in FreeCG) and ``right`` is the
    second (the gated linear of the node state). Weights have shape
    ``[num_groups, weight_numel]`` and are shared across atoms.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int,
        shuffle_scale: float = 1.5,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_groups = int(num_groups)
        self.channels = channels
        self.num_groups = num_groups
        self.shuffle_scale = float(shuffle_scale)
        self.offset = shuffle_offset(channels, num_groups, self.shuffle_scale)
        self.width = channels // num_groups
        irreps = f"{self.width}x1o + {self.width}x2e"
        self.tp = o3.FullyConnectedTensorProduct(
            irreps,
            irreps,
            irreps,
            shared_weights=False,
            internal_weights=False,
        )
        self.n_paths = len(self.tp.instructions)
        if self.n_paths != 4:
            raise RuntimeError(
                f"sparse 1o/2e product should have 4 paths, got {self.n_paths}"
            )
        self.tp_weight = Parameter(torch.empty(num_groups, self.tp.weight_numel))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.tp_weight)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        expected = (left.shape[0], ANGULAR_DIM, self.channels)
        if left.shape != expected or right.shape != expected:
            raise ValueError(
                f"group CG expects two tensors of shape {expected}, "
                f"got {tuple(left.shape)} and {tuple(right.shape)}"
            )
        packed_left = _pack_groups(left, self.num_groups, self.width)
        packed_right = _pack_groups(right, self.num_groups, self.width)
        mixed = self.tp(packed_left, packed_right, self.tp_weight)
        mixed = _unpack_groups(mixed, self.num_groups, self.width)
        if self.offset % self.channels == 0:
            return mixed
        return torch.roll(mixed, shifts=self.offset, dims=-1)
