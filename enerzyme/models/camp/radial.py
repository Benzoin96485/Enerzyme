"""Species-pair Chebyshev radial basis used inside each CAMP layer.

``T_n`` is the Chebyshev polynomial of the first kind on ``x = r / cutoff``,
the MTP argument. GRACE evaluates the same polynomials on ``2r/cutoff - 1``.
The linear map is a separate weight for every species pair and every channel.
The degree-``p`` polynomial envelope is
:func:`enerzyme.models.mp.basis.envelope.polynomial_envelope`, so the basis and its first
``p - 1`` derivatives vanish at the cutoff.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor, nn

from ..mp.basis.radial import SpeciesPairChebyshev, chebyshev_first


HiddenSpec = Union[int, Sequence[int], None]


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
