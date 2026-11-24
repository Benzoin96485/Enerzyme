"""Body-order contractions of equivariant moments into lower-rank tensors."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..basis.cartesian import multi_indices
from .product_ladder import ProductLadder
from .cartesian import packed_l2_to_stf
from ..ops.rejection import vector_rejection


def contract_feature_dyad(
    feature: Tensor, dyad: Tensor, feature_rank: int, dyad_rank: int
) -> Tensor:
    """Fully contract ``feature`` into the leading axes of ``dyad``.

    ``feature`` is ``(C, E, 3^v1)`` and ``dyad`` is ``(E, 3^v2)``. The result
    has rank ``v2 - v1``. Symmetric inputs make the choice of axes irrelevant.
    """
    if feature_rank > dyad_rank:
        raise ValueError(f"feature rank {feature_rank} exceeds dyad rank {dyad_rank}")
    out_rank = dyad_rank - feature_rank
    channels, edges = feature.shape[:2]
    if feature_rank == 0:
        if dyad_rank == 0:
            return feature * dyad.unsqueeze(0)
        return feature.reshape(channels, edges, *([1] * dyad_rank)) * dyad.unsqueeze(0)
    flat_feature = feature.reshape(channels, edges, 3 ** feature_rank)
    flat_dyad = dyad.reshape(edges, 3 ** feature_rank, 3 ** out_rank if out_rank else 1)
    mixed = torch.einsum("cej,ejk->cek", flat_feature, flat_dyad)
    if out_rank == 0:
        return mixed.squeeze(-1)
    return mixed.reshape(channels, edges, *([3] * out_rank))


def contract_moments(moments: Dict[int, Tensor], ranks: Sequence[int]) -> Tensor:
    """Contract every factor but one copy of the highest rank into that copy."""
    if len(ranks) == 1:
        return moments[int(ranks[0])]
    sink = max(range(len(ranks)), key=lambda index: ranks[index])
    sink_rank = int(ranks[sink])
    others = [int(ranks[index]) for index in range(len(ranks)) if index != sink]
    acc_rank = sum(others)
    out_rank = sink_rank - acc_rank
    if out_rank < 0:
        raise RuntimeError(f"hyper-moment ranks {tuple(ranks)} do not contract")
    sink_tensor = moments[sink_rank]
    channels, nodes = sink_tensor.shape[:2]
    spatial_out = 3 ** out_rank if out_rank else 1
    reduced = sink_tensor.reshape(channels, nodes, 3 ** acc_rank, spatial_out)
    consumed = 0
    for rank in others:
        width = 3 ** rank
        remaining = acc_rank - consumed
        reduced = reduced.reshape(
            channels, nodes, width, 3 ** (remaining - rank), spatial_out
        )
        other = moments[rank].reshape(channels, nodes, width)
        reduced = torch.einsum("cnabk,cna->cnbk", reduced, other)
        consumed += rank
    reduced = reduced.reshape(channels, nodes, spatial_out)
    if out_rank == 0:
        return reduced.squeeze(-1)
    return reduced.reshape(channels, nodes, *([3] * out_rank))


Key = Tuple[int, ...]


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


def rejection_inner(
    left: Tensor,
    right: Tensor,
    direction: Tensor,
) -> Tensor:
    """Inner product of the two rejections. Shape ``[E, C]``."""
    rejected_left = vector_rejection(left, direction)
    rejected_right = vector_rejection(right, -direction)
    return angular_inner(rejected_left, rejected_right)


def angular_inner(left: Tensor, right: Tensor) -> Tensor:
    """Channelwise scalar product over the angular components (N, M, C)."""
    return (left * right).sum(dim=1)


def angular_norm_squared(value: Tensor) -> Tensor:
    return value.square().sum(dim=1)


def vector_tensor_invariants(vector: Tensor, packed: Tensor) -> Tuple[Tensor, Tensor]:
    """Return vᵀQv and |Qv|² in the signed Cartesian Gaunt convention."""
    nodes, _, rank_vector = vector.shape
    rank_tensor = packed.shape[-1]
    vectors = vector.transpose(1, 2)
    matrices = packed_l2_to_stf(packed.transpose(1, 2))
    contracted = torch.einsum("neij,nbi->nebj", matrices, vectors)
    cubic = torch.einsum("nai,nebi->nabe", vectors, contracted)
    cubic = cubic.reshape(nodes, rank_vector * rank_vector * rank_tensor)
    cubic = cubic * (-1.0 / math.sqrt(5.0))
    quartic = torch.sum(contracted * contracted, dim=-1).reshape(nodes, rank_tensor * rank_vector)
    return cubic, quartic


def bispectrum_contraction(
    coupling: Tensor,
    value_1: Tensor,
    value_2: Tensor,
    value_3: Tensor,
) -> Tensor:
    nodes = value_1.shape[0]
    rank_1 = value_1.shape[-1]
    rank_2 = value_2.shape[-1]
    rank_3 = value_3.shape[-1]
    # The highest degree has the smallest probe rank. Contract it first
    # to bound the intermediate width and avoid planning this path per call.
    third = torch.einsum("abc,ncC->nabC", coupling, value_3)
    second = torch.einsum("nabC,nbB->naBC", third, value_2)
    full = torch.einsum("naBC,naA->nABC", second, value_1)
    return full.reshape(nodes, rank_1 * rank_2 * rank_3)


def __getattr__(name):
    if name == "SymmetricContraction":
        # Only MACE needs the e3nn/FX contraction optimizer.
        from .e3nn.symmetric_contraction import SymmetricContraction

        return SymmetricContraction
    raise AttributeError(name)
