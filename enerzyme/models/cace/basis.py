"""Cartesian monomials and polynomially independent CACE invariants.

The angular basis on a unit vector ``u = (x, y, z)`` is

``L_{lx,ly,lz}(u) = x^{lx} y^{ly} z^{lz}``, ``lx + ly + lz = l``.

Body order ``ν = 1`` keeps the rotationally invariant ``l = 0`` component.
Higher orders contract monomials that share a positive angular factor, which
is the Cartesian form of a connected Clebsch–Gordan invariant (Cheng, 2024,
Eqs. 7–10). Permutations of the same leg and products that factor into lower
body orders are left out, so each slot is one polynomial class.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


Index = Tuple[int, int, int]
Key = Tuple[int, ...]


def multi_indices(lmax: int) -> List[Index]:
    """Nonnegative ``(lx, ly, lz)`` with ``lx + ly + lz <= lmax``, grouped by ``l``."""
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    indices: List[Index] = []
    for total in range(lmax + 1):
        for lx in range(total + 1):
            for ly in range(total - lx + 1):
                indices.append((lx, ly, total - lx - ly))
    return indices


def multinomial(lx: int, ly: int, lz: int) -> int:
    """``(lx + ly + lz)! / (lx! ly! lz!)``."""
    total = lx + ly + lz
    return math.factorial(total) // (
        math.factorial(lx) * math.factorial(ly) * math.factorial(lz)
    )


def invariant_keys(lmax: int, max_nu: int) -> List[Key]:
    """One key per independent invariant, in the order the symmetrizer emits.

    ``(1, 0)`` is the scalar. ``(2, L)`` is the squared shell of weight ``L``.
    ``(3, l1, l2)`` has ``1 <= l1 <= l2`` and ``l1 + l2 <= lmax``.
    ``(4, l1, l2, dl)`` has ``1 <= l1 < l2``, ``dl >= 1``, and
    ``l2 + dl <= lmax``.
    """
    if max_nu < 1 or max_nu > 4:
        raise ValueError(f"max_nu must be in 1..4, got {max_nu}")
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    keys: List[Key] = [(1, 0)]
    if max_nu >= 2:
        keys.extend((2, total) for total in range(1, lmax + 1))
    if max_nu >= 3:
        for left in range(1, lmax + 1):
            for right in range(left, lmax + 1):
                if left + right <= lmax:
                    keys.append((3, left, right))
    if max_nu >= 4:
        for left in range(1, lmax + 1):
            for right in range(left + 1, lmax + 1):
                for shared in range(1, lmax - right + 1):
                    keys.append((4, left, right, shared))
    return keys


def cartesian_monomials(unit: Tensor, powers: Tensor) -> Tensor:
    """Evaluate ``x^{lx} y^{ly} z^{lz}`` for every multi-index.

    ``unit`` has shape ``[E, 3]`` and ``powers`` has shape ``[A, 3]``.
    Multiplication keeps the sign of each Cartesian component. A zero-length
    edge (the zero vector) contributes 1 to ``l = 0`` and 0 above it.
    """
    if powers.ndim != 2 or powers.shape[-1] != 3:
        raise ValueError(f"powers must be [A, 3], got {tuple(powers.shape)}")
    n_angular = powers.shape[0]
    if unit.numel() == 0:
        return unit.new_zeros((0, n_angular))
    if unit.ndim != 2 or unit.shape[-1] != 3:
        raise ValueError(f"unit vectors must be [E, 3], got {tuple(unit.shape)}")
    powers = powers.to(device=unit.device)
    result = unit.new_ones(unit.shape[0], n_angular)
    if n_angular == 0:
        return result
    max_power = int(powers.max().item())
    for axis in range(3):
        exponent = powers[:, axis]
        if int(exponent.max().item()) == 0:
            continue
        base = unit[:, axis].unsqueeze(1)
        running = torch.ones_like(result)
        filled = torch.ones_like(result)
        for power in range(1, max_power + 1):
            running = running * base
            filled = torch.where(exponent.unsqueeze(0) == power, running, filled)
        result = result * filled
    return result


def _pack(terms: Sequence[Tuple[int, Tuple[int, ...], int]]):
    if not terms:
        return None
    slots = torch.tensor([term[0] for term in terms], dtype=torch.long)
    angular = torch.tensor([term[1] for term in terms], dtype=torch.long)
    prefactor = torch.tensor(
        [term[2] for term in terms], dtype=torch.get_default_dtype()
    )
    return slots, angular, prefactor, int(slots.max().item()) + 1


def _pair_terms(lmax: int) -> Optional[Tuple[Tensor, Tensor, Tensor, int]]:
    indices = multi_indices(lmax)
    terms = []
    slot = 0
    for total in range(1, lmax + 1):
        for position, multi in enumerate(indices):
            if sum(multi) == total:
                terms.append((slot, (position, position), multinomial(*multi)))
        slot += 1
    return _pack(terms)


def _triple_terms(lmax: int) -> Optional[Tuple[Tensor, Tensor, Tensor, int]]:
    indices = multi_indices(lmax)
    lookup = {multi: position for position, multi in enumerate(indices)}
    terms = []
    slot = 0
    for left in range(1, lmax + 1):
        for right in range(left, lmax + 1):
            if left + right > lmax:
                continue
            for i, first in enumerate(indices):
                if sum(first) != left:
                    continue
                for j, second in enumerate(indices):
                    if sum(second) != right:
                        continue
                    combined = (
                        first[0] + second[0],
                        first[1] + second[1],
                        first[2] + second[2],
                    )
                    coefficient = multinomial(*first) * multinomial(*second)
                    terms.append((slot, (i, j, lookup[combined]), coefficient))
            slot += 1
    return _pack(terms)


def _quadruple_terms(lmax: int) -> Optional[Tuple[Tensor, Tensor, Tensor, int]]:
    indices = multi_indices(lmax)
    lookup = {multi: position for position, multi in enumerate(indices)}
    terms = []
    slot = 0
    for left in range(1, lmax + 1):
        for right in range(left + 1, lmax + 1):
            for shared in range(1, lmax - right + 1):
                for i, first in enumerate(indices):
                    if sum(first) != left:
                        continue
                    for j, second in enumerate(indices):
                        if sum(second) != right:
                            continue
                        for k, middle in enumerate(indices):
                            if sum(middle) != shared:
                                continue
                            left_mid = (
                                first[0] + middle[0],
                                first[1] + middle[1],
                                first[2] + middle[2],
                            )
                            right_mid = (
                                second[0] + middle[0],
                                second[1] + middle[1],
                                second[2] + middle[2],
                            )
                            coefficient = (
                                multinomial(*first)
                                * multinomial(*second)
                                * multinomial(*middle)
                            )
                            terms.append(
                                (
                                    slot,
                                    (i, j, lookup[left_mid], lookup[right_mid]),
                                    coefficient,
                                )
                            )
                slot += 1
    return _pack(terms)


def contraction_tables(lmax: int, max_nu: int) -> Dict[int, Optional[Tuple]]:
    tables: Dict[int, Optional[Tuple]] = {}
    if max_nu >= 2:
        tables[2] = _pair_terms(lmax)
    if max_nu >= 3:
        tables[3] = _triple_terms(lmax)
    if max_nu >= 4:
        tables[4] = _quadruple_terms(lmax)
    return tables


class CartesianAngularBasis(nn.Module):
    """Monomials through ``lmax`` and the shell layout of that list."""

    def __init__(self, lmax: int) -> None:
        super().__init__()
        indices = multi_indices(lmax)
        self.lmax = int(lmax)
        self.n_angular = len(indices)
        self.register_buffer("powers", torch.tensor(indices, dtype=torch.long))
        self.register_buffer(
            "l_index",
            torch.tensor([sum(item) for item in indices], dtype=torch.long),
        )
        bounds = []
        start = 0
        for total in range(self.lmax + 1):
            count = (total + 1) * (total + 2) // 2
            bounds.append((start, start + count))
            start += count
        self.register_buffer("shells", torch.tensor(bounds, dtype=torch.long))

    def forward(self, unit: Tensor) -> Tensor:
        return cartesian_monomials(unit, self.powers)


def _contract(
    density: Tensor,
    slots: Tensor,
    angular: Tensor,
    prefactor: Tensor,
    n_slots: int,
) -> Tensor:
    factors = density[:, :, angular, :]
    product = factors[:, :, :, 0, :]
    for leg in range(1, factors.shape[3]):
        product = product * factors[:, :, :, leg, :]
    weighted = product * prefactor.to(dtype=product.dtype).view(1, 1, -1, 1)
    n_atoms, n_radial, n_terms, n_channels = weighted.shape
    flat = weighted.permute(2, 0, 1, 3).reshape(n_terms, -1)
    acc = flat.new_zeros(n_slots, flat.shape[1])
    acc.index_add_(0, slots, flat)
    return acc.view(n_slots, n_atoms, n_radial, n_channels).permute(1, 2, 0, 3)


class InvariantSymmetrizer(nn.Module):
    """Map an atomic density ``[N, n_radial, n_angular, channels]`` to invariants.

    The ``l = 0`` slice is copied. Each higher slot sums the monomial products
    that share one polynomial class, with the multinomial weights from the
    Cartesian dot-product identity.
    """

    def __init__(self, lmax: int, max_nu: int) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.max_nu = int(max_nu)
        self.keys = invariant_keys(self.lmax, self.max_nu)
        self.n_invariants = len(self.keys)
        self._orders: List[int] = []
        for order, table in contraction_tables(self.lmax, self.max_nu).items():
            if table is None:
                continue
            slots, angular, prefactor, n_slots = table
            self.register_buffer(f"slots_{order}", slots)
            self.register_buffer(f"angular_{order}", angular)
            self.register_buffer(f"prefactor_{order}", prefactor)
            setattr(self, f"n_slots_{order}", n_slots)
            self._orders.append(order)
        if self.n_invariants != 1 + sum(
            getattr(self, f"n_slots_{order}") for order in self._orders
        ):
            raise RuntimeError("invariant key list does not match the contractions")

    def forward(self, density: Tensor) -> Tensor:
        if density.ndim != 4:
            raise ValueError(
                f"atomic density must be [N, radial, angular, channels], "
                f"got {tuple(density.shape)}"
            )
        n_atoms, n_radial, n_angular, n_channels = density.shape
        if n_angular < 1:
            raise ValueError("angular dimension must contain the scalar monomial")
        scalar = density[:, :, 0, :].unsqueeze(2)
        parts = [scalar]
        for order in self._orders:
            parts.append(
                _contract(
                    density,
                    getattr(self, f"slots_{order}"),
                    getattr(self, f"angular_{order}"),
                    getattr(self, f"prefactor_{order}"),
                    getattr(self, f"n_slots_{order}"),
                )
            )
        invariants = torch.cat(parts, dim=2)
        if invariants.shape[2] != self.n_invariants:
            raise RuntimeError(
                f"expected {self.n_invariants} invariants, got {invariants.shape[2]}"
            )
        return invariants
