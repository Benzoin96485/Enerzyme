"""Species-pair Chebyshev radial basis used inside each CAMP layer.

``T_n`` is the Chebyshev polynomial of the first kind on ``x = r / cutoff``,
the MTP argument. GRACE evaluates the same polynomials on ``2r/cutoff - 1``.
The linear map is a separate weight for every species pair and every channel.
The degree-``p`` polynomial envelope is
:func:`enerzyme.models.cutoff.polynomial_envelope`, so the basis and its first
``p - 1`` derivatives vanish at the cutoff.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ..cutoff import polynomial_envelope


HiddenSpec = Union[int, Sequence[int], None]


def chebyshev_first(degree: int, x: Tensor) -> Tensor:
    """``T_0, ..., T_degree`` stacked on a leading axis. Shape ``(degree + 1, *x.shape)``."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    columns = [torch.ones_like(x)]
    if degree == 0:
        return columns[0].unsqueeze(0)
    columns.append(x)
    for _ in range(2, degree + 1):
        columns.append(2.0 * x * columns[-1] - columns[-2])
    return torch.stack(columns, dim=0)


def resolve_hidden_layers(spec: HiddenSpec, width: int) -> List[int]:
    """An int is that many hidden layers of size ``width``. A list is explicit."""
    if spec is None:
        spec = 2
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"hidden depth must be >= 0, got {spec}")
        return [int(width)] * int(spec)
    layers = [int(size) for size in spec]
    if any(size < 1 for size in layers):
        raise ValueError(f"hidden widths must be >= 1, got {layers}")
    return layers


def silu_mlp(in_features: int, hidden: Sequence[int], out_features: int) -> nn.Sequential:
    """Linear stack with SiLU between layers and a linear output."""
    sizes = [int(in_features), *[int(size) for size in hidden], int(out_features)]
    layers: List[nn.Module] = []
    for left, right in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(left, right))
        layers.append(nn.SiLU())
    if not layers:
        raise ValueError("MLP needs at least one linear map")
    layers.pop()
    return nn.Sequential(*layers)


class SpeciesPairChebyshev(nn.Module):
    """``B_u(r, zi, zj) = envelope(r) Σ_β c[zi, zj, u, β] T_β(r / cutoff)``.

    ``zi`` and ``zj`` are compact species indices, not atomic numbers. The
    caller owns that table so several layers can share one vocabulary.
    """

    def __init__(
        self,
        n_channels: int,
        n_species: int,
        degree: int,
        cutoff: float,
        envelope_p: int,
    ) -> None:
        super().__init__()
        if int(n_channels) < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if int(n_species) < 1:
            raise ValueError(f"n_species must be >= 1, got {n_species}")
        if int(degree) < 0:
            raise ValueError(f"degree must be >= 0, got {degree}")
        if float(cutoff) <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")
        if int(envelope_p) < 1:
            raise ValueError(f"envelope_p must be >= 1, got {envelope_p}")
        self.n_channels = int(n_channels)
        self.n_species = int(n_species)
        self.degree = int(degree)
        self.cutoff = float(cutoff)
        self.envelope_p = int(envelope_p)
        self.coeff = nn.Parameter(
            torch.empty(self.n_species, self.n_species, self.n_channels, self.degree + 1)
        )
        bound = (self.degree + 1) ** -0.5
        nn.init.uniform_(self.coeff, -bound, bound)

    def forward(self, distance: Tensor, zi: Tensor, zj: Tensor) -> Tensor:
        """Return ``[E, n_channels]``. Distances at or beyond the cutoff are 0."""
        if distance.shape != zi.shape or distance.shape != zj.shape:
            raise ValueError(
                "distance, zi, and zj must share a shape, "
                f"got {tuple(distance.shape)}, {tuple(zi.shape)}, {tuple(zj.shape)}"
            )
        basis = chebyshev_first(self.degree, distance / self.cutoff)
        window = polynomial_envelope(distance, self.cutoff, self.envelope_p).to(
            dtype=distance.dtype
        )
        # coeff rows: (E, C, degree + 1). basis: (degree + 1, E).
        pair = self.coeff[zi.long(), zj.long()]
        return torch.einsum("ecd,de->ec", pair, basis * window)


def species_lookup(max_Za: int, atomic_numbers: Optional[Sequence[int]]) -> tuple:
    """Map atomic numbers onto the rows of a species-pair table.

    Without ``atomic_numbers`` the table is indexed by ``Z`` itself and has
    ``max_Za + 1`` rows. With a vocabulary the table has one row per listed
    element, which is how the reference model stores ``num_atom_types``.
    """
    if int(max_Za) < 1:
        raise ValueError(f"max_Za must be >= 1, got {max_Za}")
    if atomic_numbers is None:
        return torch.arange(int(max_Za) + 1, dtype=torch.long), int(max_Za) + 1
    numbers = [int(z) for z in atomic_numbers]
    if len(set(numbers)) != len(numbers):
        raise ValueError("atomic_numbers contains duplicates")
    if any(z < 0 or z > int(max_Za) for z in numbers):
        raise ValueError(f"atomic_numbers must lie in 0..max_Za ({max_Za})")
    lookup = torch.full((int(max_Za) + 1,), -1, dtype=torch.long)
    for index, atomic_number in enumerate(numbers):
        lookup[atomic_number] = index
    return lookup, len(numbers)
