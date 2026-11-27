"""Convert between e3nn's multiplicity-major blocks and spherical arrays.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

import torch
from e3nn import o3


class LayoutTransform(torch.nn.Module):
    """Convert flat e3nn features to [N, degree components, channels] and back."""

    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.flat_width = self.irreps.dim
        multiplicities = {mul for mul, _ in self.irreps}
        if len(multiplicities) != 1 or 0 in multiplicities:
            raise ValueError("a spherical array requires equal positive multiplicities")
        self.channels = multiplicities.pop()
        self.components = sum(ir.dim for _, ir in self.irreps)
        self.single_block = len(self.irreps) == 1
        self.same_order = self.channels == 1 or all(ir.dim == 1 for _, ir in self.irreps)
        # Each individual block is a transpose view. Joining several transposed
        # blocks with different degree widths requires a permutation of storage.
        perm = torch.arange(self.flat_width)
        perm = torch.cat([perm[sl].reshape(mul, ir.dim).T.reshape(-1)
                          for (mul, ir), sl in zip(self.irreps, self.irreps.slices())])
        self.register_buffer("to_sphere", perm, persistent=False)
        self.register_buffer("to_flat", perm.argsort(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.flat_width:
            raise ValueError(f"expected flat features [N, {self.flat_width}]")
        if self.single_block:
            return x.reshape(x.size(0), self.channels, self.components).transpose(1, 2)
        if not self.same_order:
            x = x.index_select(1, self.to_sphere)
        return x.reshape(x.size(0), self.components, self.channels)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.components or x.size(2) != self.channels:
            raise ValueError(f"expected spherical features [N, {self.components}, {self.channels}]")
        if self.single_block:
            return x.transpose(1, 2).reshape(x.size(0), self.flat_width)
        flat = x.reshape(x.size(0), self.flat_width)
        return flat if self.same_order else flat.index_select(1, self.to_flat)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps})"
