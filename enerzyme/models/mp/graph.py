"""Per-forward graph context shared by every block of a Core.

Edge convention: ``receiver`` is the atom being updated (Enerzyme ``idx_i``),
``sender`` the neighbor (``idx_j``), and ``vec = R_sender - R_receiver``
(``vij`` from :class:`~enerzyme.models.layers.geometry.DistanceLayer`).
``edge_convention="swap"`` flips the roles for Cores fitted with the opposite
orientation (main-branch MACE), so blocks never see the difference.

Derived quantities (unit vectors, spherical harmonics, reverse edges, triplets,
CSR offsets, rotation frames) are computed on first use and cached for the rest
of the forward pass, so every block shares one copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Hashable, NamedTuple, Optional, Set, Tuple

import torch
from torch import Tensor

EDGE_SETS = ("sr", "lr", "internal")
EDGE_CONVENTIONS = ("receiver_i", "swap")

# A true reverse edge has displacement -v, so the residual is identically zero;
# anything larger is a different periodic image and must not be paired.
_OPPOSITE_TOL = 1e-4


@dataclass(frozen=True)
class GraphRequires:
    """Inputs a Core reads and derived graph quantities its blocks use.

    ``edge_set="internal"`` lets an embedding construct its own graph from node fields.
    Otherwise ``edge_set`` picks the neighbor list: ``"sr"`` reads ``idx_i_sr`` /
    ``idx_j_sr`` / ``vij_sr`` / ``Dij_sr``; ``"lr"`` reads the full list the
    Core's own :class:`DistanceLayer` produces (``idx_i`` / ``idx_j`` /
    ``vij_lr`` / ``Dij_lr``). ``fields`` names extra raw inputs (``Za``,
    ``batch_seg``, ``Q``, ...) passed through ``GraphContext.inputs``.
    """

    edge_set: str = "sr"
    vectors: bool = True
    distances: bool = True
    rbf: bool = True
    cutoff_values: bool = False
    optional_cutoff_values: bool = True
    edge_convention: str = "receiver_i"
    fields: Tuple[str, ...] = ()
    optional_fields: Tuple[str, ...] = ()
    path_triplets: bool = False

    def __post_init__(self) -> None:
        if self.edge_set not in EDGE_SETS:
            raise ValueError(f"edge_set must be one of {EDGE_SETS}, got {self.edge_set!r}")
        if self.edge_convention not in EDGE_CONVENTIONS:
            raise ValueError(f"edge_convention must be one of {EDGE_CONVENTIONS}, got {self.edge_convention!r}")

    @property
    def edge_fields(self) -> Dict[str, str]:
        if self.edge_set == "internal":
            return {}
        if self.edge_set == "sr":
            names = dict(receiver="idx_i_sr", sender="idx_j_sr", vec="vij_sr", dist="Dij_sr",
                         cutoff_values="cutoff_values_sr")
        else:
            names = dict(receiver="idx_i", sender="idx_j", vec="vij_lr", dist="Dij_lr",
                         cutoff_values="cutoff_values_lr")
        return names

    def input_fields(self) -> Set[str]:
        if self.edge_set == "internal":
            return set(self.fields) | set(self.optional_fields)
        names = self.edge_fields
        out = {names["receiver"], names["sender"]}
        if self.vectors:
            out.add(names["vec"])
        if self.distances:
            out.add(names["dist"])
        if self.rbf:
            out.add("rbf")
        if self.cutoff_values or self.optional_cutoff_values:
            out.add(names["cutoff_values"])
        out.update(self.fields)
        out.update(self.optional_fields)
        return out

    def merge(self, other: "GraphRequires") -> "GraphRequires":
        if other.edge_set != self.edge_set or other.edge_convention != self.edge_convention:
            raise ValueError("cannot merge requirements on different edge sets / conventions")
        return replace(
            self,
            vectors=self.vectors or other.vectors,
            path_triplets=self.path_triplets or other.path_triplets,
            distances=self.distances or other.distances,
            rbf=self.rbf or other.rbf,
            cutoff_values=self.cutoff_values or other.cutoff_values,
            optional_cutoff_values=self.optional_cutoff_values or other.optional_cutoff_values,
            fields=tuple(dict.fromkeys(self.fields + other.fields)),
            optional_fields=tuple(dict.fromkeys(self.optional_fields + other.optional_fields)),
        )


class ReceiverCSR(NamedTuple):
    """Edges grouped by receiver: ``perm[offsets[a]:offsets[a+1]]`` are the edges into ``a``."""

    counts: Tensor
    offsets: Tensor
    perm: Tensor


class EdgePairs(NamedTuple):
    """Pairs of edges. For receiver pairs ``first`` is updated and ``second`` is the other bond;
    for path pairs ``first`` arrives at the shared atom and ``second`` leaves it."""

    first: Tensor
    second: Tensor


def receiver_csr(receiver: Tensor, num_nodes: int) -> ReceiverCSR:
    counts = torch.bincount(receiver, minlength=num_nodes)
    offsets = torch.empty(num_nodes + 1, dtype=torch.long, device=receiver.device)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    perm = torch.argsort(receiver, stable=True)
    return ReceiverCSR(counts, offsets, perm)


def receiver_pairs(
    receiver: Tensor,
    num_nodes: int,
    edge_subset: Optional[Tensor] = None,
    include_self: bool = False,
) -> EdgePairs:
    """Ordered pairs of edges that share a receiver (angle at the receiver).

    Pairs are enumerated by ``first`` in edge order, then ``second`` in
    receiver-CSR order. ``edge_subset`` restricts both members to those edge
    ids (three-body cutoffs); returned indices address the full edge list.
    ``include_self`` keeps the diagonal pair ``(e, e)``.
    """
    empty = receiver.new_empty(0)
    kept = edge_subset
    centers = receiver if kept is None else receiver[kept]
    n_kept = int(centers.numel())
    if n_kept == 0:
        return EdgePairs(empty, empty)
    device = receiver.device
    counts, offsets, perm = receiver_csr(centers, num_nodes)
    degree = counts[centers] if include_self else counts[centers] - 1
    total = int(degree.sum().item())
    if total == 0:
        return EdgePairs(empty, empty)
    src_local = torch.repeat_interleave(torch.arange(n_kept, device=device), degree)
    prefix = torch.cumsum(degree, dim=0) - degree
    slot = torch.arange(total, device=device) - torch.repeat_interleave(prefix, degree)
    if include_self:
        other_local = slot
    else:
        local = torch.empty(n_kept, dtype=torch.long, device=device)
        local[perm] = torch.arange(n_kept, device=device) - offsets[centers[perm]]
        other_local = slot + (slot >= local[src_local]).to(slot.dtype)
    dst_local = perm[offsets[centers[src_local]] + other_local]
    if kept is None:
        return EdgePairs(src_local, dst_local)
    return EdgePairs(kept[src_local], kept[dst_local])


def cross_receiver_pairs(query_receiver: Tensor, key_receiver: Tensor, num_nodes: int) -> EdgePairs:
    """Pair each query edge with every key edge that arrives at the same receiver.

    ``first`` indexes the query list in edge order, ``second`` indexes the key
    list in receiver-CSR order. The two lists may be the same graph, in which
    case the diagonal pair is included.
    """
    empty = query_receiver.new_empty(0)
    if query_receiver.numel() == 0 or key_receiver.numel() == 0 or int(num_nodes) == 0:
        return EdgePairs(empty, empty)
    counts, offsets, perm = receiver_csr(key_receiver, num_nodes)
    degree = counts[query_receiver]
    total = int(degree.sum().item())
    if total == 0:
        return EdgePairs(empty, empty)
    device = query_receiver.device
    first = torch.repeat_interleave(torch.arange(query_receiver.numel(), device=device), degree)
    prefix = torch.cumsum(degree, dim=0) - degree
    slot = torch.arange(total, device=device) - torch.repeat_interleave(prefix, degree)
    return EdgePairs(first, perm[offsets[query_receiver[first]] + slot])


def receiver_rank(receiver: Tensor, key: Tensor, num_nodes: int) -> Tensor:
    """Rank of each edge among the edges at its receiver by ascending ``key``.

    Ties keep edge order, so ``receiver_rank(...) < k`` is a deterministic
    k-nearest-neighbor mask.
    """
    by_key = torch.argsort(key, stable=True)
    order = by_key[torch.argsort(receiver[by_key], stable=True)]
    counts = torch.bincount(receiver, minlength=num_nodes)
    start = torch.cumsum(counts, dim=0) - counts
    rank = torch.empty_like(receiver)
    rank[order] = torch.arange(receiver.numel(), device=receiver.device) - start[receiver[order]]
    return rank


def path_pairs(receiver: Tensor, sender: Tensor, num_nodes: int, order: str = "outgoing") -> EdgePairs:
    """Head-to-tail pairs ``(k→j, j→i)``: ``first`` arrives at ``j``, ``second`` leaves it.

    ``order="outgoing"`` enumerates by the outgoing edge in edge order (DimeNet);
    ``order="atom"`` enumerates by the shared atom, then incoming, then outgoing
    edge (DGL line graph, ALIGNN). No backtracking filter is applied.
    """
    empty = receiver.new_empty(0)
    num_edges = int(receiver.numel())
    if num_edges == 0:
        return EdgePairs(empty, empty)
    device = receiver.device
    in_counts, in_offsets, in_perm = receiver_csr(receiver, num_nodes)
    if order == "outgoing":
        n_trip = in_counts[sender]
        total = int(n_trip.sum().item())
        if total == 0:
            return EdgePairs(empty, empty)
        outgoing = torch.repeat_interleave(torch.arange(num_edges, device=device), n_trip)
        prefix = torch.cumsum(n_trip, dim=0) - n_trip
        local = torch.arange(total, device=device) - torch.repeat_interleave(prefix, n_trip)
        incoming = in_perm[torch.repeat_interleave(in_offsets[sender], n_trip) + local]
        return EdgePairs(incoming, outgoing)
    if order != "atom":
        raise ValueError(f"order must be 'outgoing' or 'atom', got {order!r}")
    out_counts, out_offsets, out_perm = receiver_csr(sender, num_nodes)
    n_pairs = in_counts * out_counts
    total = int(n_pairs.sum().item())
    if total == 0:
        return EdgePairs(empty, empty)
    pair_atom = torch.repeat_interleave(torch.arange(num_nodes, device=device), n_pairs)
    pair_prefix = torch.cumsum(n_pairs, dim=0) - n_pairs
    local = torch.arange(total, device=device) - torch.repeat_interleave(pair_prefix, n_pairs)
    width = out_counts[pair_atom]
    in_local = torch.div(local, width, rounding_mode="floor")
    out_local = local - in_local * width
    return EdgePairs(in_perm[in_offsets[pair_atom] + in_local], out_perm[out_offsets[pair_atom] + out_local])


def reverse_edge_index(receiver: Tensor, sender: Tensor, vec: Tensor, num_nodes: int) -> Tensor:
    """Index of the edge ``(sender, receiver, -vec)`` for every edge, periodic images included.

    The match is a neighbor-list identity, so it runs on a detached copy of
    ``vec`` and raises when an edge has no opposite partner.
    """
    num_edges = int(receiver.numel())
    if num_edges == 0:
        return receiver.new_empty(0)
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive when the graph has edges")
    if int(receiver.max()) >= num_nodes or int(sender.max()) >= num_nodes:
        raise ValueError("edge index is outside num_nodes")
    if int(receiver.min()) < 0 or int(sender.min()) < 0:
        raise ValueError("edge index must be non-negative")
    if vec.ndim != 2 or vec.shape[0] != num_edges or vec.shape[-1] != 3:
        raise ValueError(f"vec must have shape ({num_edges}, 3), got {tuple(vec.shape)}")
    disp = vec.detach()
    key = receiver * num_nodes + sender
    rev = sender * num_nodes + receiver
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    start = torch.searchsorted(sorted_key, rev, right=False)
    end = torch.searchsorted(sorted_key, rev, right=True)
    if torch.any(end == start):
        raise ValueError("reverse edges need both directions of every edge in the neighbor list")
    width = int((end - start).max().item())
    if width == 1:
        chosen = order[start]
    else:
        device = receiver.device
        offsets = torch.arange(width, device=device)
        pos = start[:, None] + offsets
        valid = offsets[None, :] < (end - start)[:, None]
        cand = order[pos.clamp(max=num_edges - 1)]
        mismatch = (disp[cand] + disp[:, None, :]).square().sum(dim=-1)
        mismatch = mismatch.masked_fill(~valid, float("inf"))
        best = mismatch.argmin(dim=1)
        chosen = cand[torch.arange(num_edges, device=device), best]
    residual = (disp[chosen] + disp).square().sum(dim=-1).sqrt()
    if not torch.isfinite(residual).all() or bool((residual > _OPPOSITE_TOL).any()):
        raise ValueError(
            "could not match every edge to the opposite displacement; each directed edge "
            "(receiver, sender, v) needs (sender, receiver, -v) in the neighbor list"
        )
    return chosen


@dataclass
class GraphContext:
    """Edges and cached derived quantities for one forward pass."""

    receiver: Optional[Tensor]
    sender: Optional[Tensor]
    num_nodes: int
    vec: Optional[Tensor] = None
    dist: Optional[Tensor] = None
    rbf: Optional[Tensor] = None
    cutoff_values: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[Hashable, Any] = field(default_factory=dict)

    @classmethod
    def from_inputs(cls, inputs: Dict[str, Any], requires: GraphRequires, num_nodes: int,
                    dtype: Optional[torch.dtype] = None) -> "GraphContext":
        if requires.edge_set == "internal":
            return cls(None, None, num_nodes, inputs=inputs, batch=inputs.get("batch_seg"))
        names = requires.edge_fields
        receiver = inputs[names["receiver"]].long()
        sender = inputs[names["sender"]].long()
        vec = inputs.get(names["vec"]) if requires.vectors else None
        dist = inputs.get(names["dist"]) if requires.distances else None
        rbf = inputs.get("rbf") if requires.rbf else None
        cutoff_values = inputs.get(names["cutoff_values"])
        if requires.cutoff_values and cutoff_values is None:
            raise KeyError(f"{names['cutoff_values']} is required by this Core")
        if dtype is not None:
            vec = vec.to(dtype) if vec is not None else None
            dist = dist.to(dtype) if dist is not None else None
            rbf = rbf.to(dtype) if rbf is not None else None
            cutoff_values = cutoff_values.to(dtype) if cutoff_values is not None else None
        if requires.edge_convention == "swap":
            receiver, sender = sender, receiver
            vec = -vec if vec is not None else None
        ctx = cls(
            receiver=receiver, sender=sender, num_nodes=int(num_nodes), vec=vec, dist=dist, rbf=rbf,
            cutoff_values=cutoff_values, batch=inputs.get("batch_seg"), inputs=inputs,
        )

        if requires.path_triplets:
            ctx.path_pairs(exclude_backtracking=True)
        return ctx

    @property
    def num_edges(self) -> int:
        return 0 if self.receiver is None else int(self.receiver.numel())

    def cached(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def unit(self, eps: float = 1e-12) -> Tensor:
        """``vec / max(|vec|, eps)``."""
        return self.cached(("unit", eps), lambda: self.vec / torch.linalg.vector_norm(
            self.vec, dim=-1, keepdim=True).clamp_min(eps))

    def edge_length(self) -> Tensor:
        if self.dist is not None:
            return self.dist
        return self.cached("edge_length", lambda: torch.linalg.vector_norm(self.vec, dim=-1))

    def spherical_harmonics(self, lmax: int, normalization: Optional[str] = "component", normalize: bool = True,
                            layout: str = "e3nn", edge_sign: float = 1.0) -> Tensor:
        """Real spherical harmonics of the edge vectors, ``(E, (lmax+1)^2)``."""
        from .basis.angular import spherical_harmonics

        key = ("sh", int(lmax), normalization, normalize, layout, edge_sign)
        return self.cached(key, lambda: spherical_harmonics(
            self.vec, list(range(int(lmax) + 1)), layout=layout, normalization=normalization,
            normalize_input=normalize, edge_sign=edge_sign))

    def edge_rotation(self, lmax, backend="cg", *, mmax=None, edge_sign=1, frame=None):
        """Cache the rotation of this graph's edges, including any supplied frame."""
        from .equivariant.rotation import edge_rotation

        if edge_sign not in (1, -1):
            raise ValueError("edge_sign must be 1 or -1")
        key = ("edge_rotation", lmax, backend, mmax, edge_sign, frame)
        return self.cached(key, lambda: edge_rotation(
            self.vec if edge_sign == 1 else -self.vec, lmax, backend, frame=frame, mmax=mmax))

    def csr(self) -> ReceiverCSR:
        return self.cached("csr", lambda: receiver_csr(self.receiver, self.num_nodes))

    def reverse_edge(self) -> Tensor:
        return self.cached("reverse_edge", lambda: reverse_edge_index(
            self.receiver, self.sender, self.vec, self.num_nodes))

    def receiver_pairs(self, cutoff: Optional[float] = None, include_self: bool = False,
                       strict: bool = False) -> EdgePairs:
        """Edge pairs sharing a receiver, optionally inside a three-body ``cutoff``.

        ``strict`` uses ``dist < cutoff`` instead of ``dist <= cutoff``.
        """
        def build() -> EdgePairs:
            subset = None
            if cutoff is not None:
                length = self.edge_length()
                inside = length < float(cutoff) if strict else length <= float(cutoff)
                subset = torch.where(inside)[0]
            return receiver_pairs(self.receiver, self.num_nodes, subset, include_self)

        return self.cached(("receiver_pairs", cutoff, include_self, strict), build)

    def path_pairs(self, order: str = "outgoing", exclude_backtracking: bool = False) -> EdgePairs:
        def build():
            incoming, outgoing = path_pairs(self.receiver, self.sender, self.num_nodes, order)
            if exclude_backtracking:
                keep = self.sender[incoming] != self.receiver[outgoing]
                return EdgePairs(incoming[keep], outgoing[keep])
            return EdgePairs(incoming, outgoing)

        return self.cached(("path_pairs", order, exclude_backtracking), build)

    def num_graphs(self) -> int:
        if self.batch is None or self.batch.numel() == 0:
            return 1
        return self.cached("num_graphs", lambda: int(self.batch.max().item()) + 1)
