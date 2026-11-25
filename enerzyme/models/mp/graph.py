"""Per-forward graph context shared by every block of a Core.

Edge convention: ``receiver`` is the atom being updated (Enerzyme ``idx_i``),
``sender`` the neighbor (``idx_j``), and ``vec = R_sender - R_receiver``
(``vij`` from :class:`~enerzyme.models.layers.geometry.DistanceLayer`).
``edge_convention="swap"`` flips the roles for Cores fitted with the opposite
orientation (main-branch MACE), so blocks never see the difference.

The Core builds declared topology, harmonics and rotations on entry. All blocks
share those results and can cache additional quantities for this forward pass.

Cache keys are built only from constants. The context crosses compiler graph
breaks, and Dynamo guards on its cache keys: a key holding a tensor or another
context is new on every call and would force a recompilation each time. Derived
graphs therefore carry a provenance ``key`` (for example ``(("within", 5.0),)``)
instead of being looked up by identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Hashable, NamedTuple, Optional, Protocol, Set, Tuple

import torch
from torch import Tensor

EDGE_SETS = ("sr", "lr", "internal")
EDGE_CONVENTIONS = ("receiver_i", "swap")

# A true reverse edge has displacement -v, so the residual is identically zero;
# anything larger is a different periodic image and must not be paired.
_OPPOSITE_TOL = 1e-4


class TopologyRequest(Protocol):
    def prepare(self, ctx: "GraphContext") -> Any: ...


@dataclass(frozen=True)
class TripletRequest:
    """A path, receiver or line graph; indices refer to the selected edge list."""

    kind: str
    edge_cutoff: Optional[float] = None
    cutoff: Optional[float] = None
    include_self: bool = False
    strict: bool = False
    exclude_backtracking: bool = False
    exclude_same_sender: bool = False

    def prepare(self, ctx):
        return ctx.within(self.edge_cutoff).triplets(
            self.kind, cutoff=self.cutoff, include_self=self.include_self, strict=self.strict,
            exclude_backtracking=self.exclude_backtracking, exclude_same_sender=self.exclude_same_sender)


@dataclass(frozen=True)
class QuadrupletRequest:
    edge_cutoff: float
    interaction_cutoff: float
    images: bool = False

    def prepare(self, ctx):
        return ctx.within(self.edge_cutoff).quadruplets(ctx.within(self.interaction_cutoff), images=self.images)


@dataclass(frozen=True)
class ReverseEdgeRequest:
    edge_cutoff: Optional[float] = None

    def prepare(self, ctx):
        return ctx.within(self.edge_cutoff).reverse_edge()


@dataclass(frozen=True)
class HarmonicsRequest:
    lmax: int
    edge_sign: float = 1.0
    normalization: str = "component"
    normalize: bool = True
    layout: str = "e3nn"

    def prepare(self, ctx):
        return ctx.spherical_harmonics(self.lmax, self.normalization, self.normalize, self.layout, self.edge_sign)


@dataclass(frozen=True)
class RotationRequest:
    lmax: int
    backend: Any = "cg"
    mmax: Optional[int] = None
    edge_sign: int = 1
    frame_field: Optional[str] = None

    def prepare(self, ctx):
        return ctx.edge_rotation(self.lmax, self.backend, mmax=self.mmax, edge_sign=self.edge_sign,
                                 frame_field=self.frame_field or None)


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
    topology: tuple[TopologyRequest, ...] = ()
    csr: bool = False
    unit: bool = False
    harmonics: Tuple[HarmonicsRequest, ...] = ()
    rotations: Tuple[RotationRequest, ...] = ()

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
            csr=self.csr or other.csr,
            unit=self.unit or other.unit,
            harmonics=tuple(dict.fromkeys(self.harmonics + other.harmonics)),
            rotations=tuple(dict.fromkeys(self.rotations + other.rotations)),
            topology=tuple(dict.fromkeys(self.topology + other.topology)),
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


@dataclass
class PreparedTopology:
    """Discrete tables shared with a new, differentiable view of the same geometry."""

    root: dict
    subgraphs: dict
    quadruplets: list

    @classmethod
    def build(cls, graph, requests):
        from .quadruplets import quadruplet_indices

        quadruplets = []
        fixed = {}
        for request in requests:
            if isinstance(request, QuadrupletRequest):
                embedding = graph.within(request.edge_cutoff)
                interaction = graph.within(request.interaction_cutoff)
                quadruplets.append((request, quadruplet_indices(embedding, interaction, request.images)))
            else:
                result = request.prepare(graph)
                if getattr(request, "force_cache_key", None) is not None:
                    fixed[request.force_cache_key] = result

        def indices(view):
            # Subgraphs carry differentiable distances and directions. Keep
            # only discrete tables addressed by stable keys in this view.
            return {k: v for k, v in view.cache.items() if k == "csr" or (
                isinstance(k, tuple) and k[0] in ("edge_indices", "receiver_pairs", "path_pairs", "reverse_edge")
            )}

        subgraphs = {k[1]: indices(v) for k, v in graph.cache.items()
                     if isinstance(k, tuple) and k[0] == "within"}
        return cls({**indices(graph), **fixed}, subgraphs, quadruplets)

    def apply(self, graph):
        from .quadruplets import dihedral_geometry

        graph.cache.update(self.root)
        for cutoff, indices in self.subgraphs.items():
            view = graph.within(cutoff)
            view.cache.update(indices)
            view.cache["_prepared_cutoff"] = cutoff
        for request, indices in self.quadruplets:
            embedding = graph.within(request.edge_cutoff)
            interaction = graph.within(request.interaction_cutoff)
            key = ("_prepared_quads", request.interaction_cutoff, request.images)
            embedding.cache[key] = dihedral_geometry(embedding, interaction, indices)


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
    *, csr: Optional[ReceiverCSR] = None,
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
    counts, offsets, perm = receiver_csr(centers, num_nodes) if csr is None else csr
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


def cross_receiver_pairs(query_receiver: Tensor, key_receiver: Tensor, num_nodes: int, *,
                         csr: Optional[ReceiverCSR] = None) -> EdgePairs:
    """Pair each query edge with every key edge that arrives at the same receiver.

    ``first`` indexes the query list in edge order, ``second`` indexes the key
    list in receiver-CSR order. The two lists may be the same graph, in which
    case the diagonal pair is included.
    """
    empty = query_receiver.new_empty(0)
    if query_receiver.numel() == 0 or key_receiver.numel() == 0 or int(num_nodes) == 0:
        return EdgePairs(empty, empty)
    counts, offsets, perm = receiver_csr(key_receiver, num_nodes) if csr is None else csr
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


def path_pairs(receiver: Tensor, sender: Tensor, num_nodes: int, order: str = "outgoing",
               *, csr: Optional[ReceiverCSR] = None) -> EdgePairs:
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
    in_counts, in_offsets, in_perm = receiver_csr(receiver, num_nodes) if csr is None else csr
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


def reverse_edge_index(receiver: Tensor, sender: Tensor, vec: Tensor, num_nodes: int, *,
                       atol: float = _OPPOSITE_TOL, rtol: float = 0.0, metric: str = "l2",
                       distinct: bool = False) -> Tensor:
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
    if metric not in ("l2", "max"):
        raise ValueError(f"unknown displacement metric {metric!r}")
    disp = vec.detach()
    ids = torch.arange(num_edges, device=receiver.device)
    key = receiver * num_nodes + sender
    rev = sender * num_nodes + receiver
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    start = torch.searchsorted(sorted_key, rev, right=False)
    end = torch.searchsorted(sorted_key, rev, right=True)
    count = end - start
    if torch.any(count == 0):
        raise ValueError("reverse edges need both directions of every edge in the neighbor list")
    width = int(count.max().item())
    if width == 1:
        chosen = order[start]
    else:
        offsets = torch.arange(width, device=receiver.device)
        cand = order[(start[:, None] + offsets).clamp(max=num_edges - 1)]
        gaps = disp[cand] + disp[:, None]
        cost = gaps.abs().amax(-1) if metric == "max" else torch.linalg.vector_norm(gaps, dim=-1)
        valid = offsets < count[:, None]
        if distinct:
            valid = valid & (cand != ids[:, None])
        cost = cost.masked_fill(~valid, float("inf"))
        chosen = cand[ids, cost.argmin(dim=1)]
        if not torch.equal(chosen[chosen], ids):
            # Identical copies of an image can share the same nearest reverse.
            # Greedy one-to-one tie breaking follows cost, then input edge order.
            limit = atol
            if rtol:
                scales = disp.abs().amax(-1).clamp_min(1)
                limit = atol + rtol * torch.maximum(scales[:, None], scales[cand])
            rows, slots = torch.where((cost <= limit) & (ids[:, None] <= cand))
            partners = cand[rows, slots].tolist()
            costs = cost[rows, slots].tolist()
            matching = [-1] * num_edges
            for _, first, second in sorted(zip(costs, rows.tolist(), partners)):
                if matching[first] == matching[second] == -1:
                    matching[first], matching[second] = second, first
            if -1 in matching:
                raise ValueError("could not match every edge to a distinct reverse edge")
            chosen = receiver.new_tensor(matching)
    gap = disp[chosen] + disp
    residual = gap.abs().amax(-1) if metric == "max" else torch.linalg.vector_norm(gap, dim=-1)
    limit = atol
    if rtol:
        limit = atol + rtol * torch.maximum(disp.abs(), disp[chosen].abs()).amax(-1).clamp_min(1)
    if (not torch.isfinite(residual).all() or bool((residual > limit).any()) or
            (distinct and bool((chosen == ids).any()))):
        raise ValueError(
            "could not match every edge to the opposite displacement; each directed edge "
            "(receiver, sender, v) needs (sender, receiver, -v) in the neighbor list"
        )
    return chosen


@dataclass(frozen=True, eq=False)
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
    # Provenance from the Core's root context: ``()`` for the root, one constant
    # step per derived selection, ``None`` when unknown. Graphs with a key can
    # be used as constant cache keys by their relatives.
    key: Optional[Tuple[Hashable, ...]] = None

    @classmethod
    def from_inputs(cls, inputs: Dict[str, Any], requires: GraphRequires, num_nodes: int,
                    dtype: Optional[torch.dtype] = None) -> "GraphContext":
        if requires.edge_set == "internal":
            return cls(None, None, num_nodes, inputs=inputs, batch=inputs.get("batch_seg"), key=())
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
            cutoff_values=cutoff_values, batch=inputs.get("batch_seg"), inputs=inputs, key=(),
        )
        if "_prepared_topology" in inputs:
            inputs["_prepared_topology"].apply(ctx)

        if requires.csr:
            ctx.csr()
        if requires.unit:
            ctx.unit()
        for request in requires.topology + requires.harmonics + requires.rotations:
            request.prepare(ctx)
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

    def edge_rotation(self, lmax, backend="cg", *, mmax=None, edge_sign=1, frame=None, frame_field=None):
        """Rotation of this graph's edges, optionally about per-edge frames.

        ``frame_field`` names an input holding the frames; that result is cached
        like a rotation without frames. A frame tensor passed directly has no
        constant cache key and is rotated on every call.
        """
        from .equivariant.rotation import edge_rotation

        if edge_sign not in (1, -1):
            raise ValueError("edge_sign must be 1 or -1")
        if frame_field is not None:
            if frame is not None:
                raise ValueError("pass either frame or frame_field")
            frame = self.inputs.get(frame_field)

        def build():
            return edge_rotation(self.vec if edge_sign == 1 else -self.vec, lmax, backend, frame=frame, mmax=mmax)

        if frame is not None and frame_field is None:
            return build()
        key = ("edge_rotation", lmax, backend, mmax, edge_sign, None if frame is None else frame_field)
        return self.cached(key, build)

    def csr(self, indices=None, key: Optional[Hashable] = None) -> ReceiverCSR:
        """Receiver CSR of every edge, or of the edges in ``indices``.

        A selection is cached only under a constant ``key`` naming it.
        """
        if indices is None:
            return self.cached("csr", lambda: receiver_csr(self.receiver, self.num_nodes))
        if key is None:
            return receiver_csr(self.receiver[indices], self.num_nodes)
        return self.cached(("csr", key), lambda: receiver_csr(self.receiver[indices], self.num_nodes))

    def edge_indices(self, cutoff, strict=False):
        def select():
            length = self.edge_length()
            return torch.where(length < cutoff if strict else length <= cutoff)[0]
        return self.cached(("edge_indices", cutoff, strict), select)

    def _select(self, indices: Tensor, step: Optional[Hashable], csr_key: Optional[Hashable]) -> "GraphContext":
        fields = (self.sender, self.vec, self.dist, self.rbf, self.cutoff_values)
        if any(t is not None and (not t.ndim or t.shape[0] != self.num_edges) for t in fields):
            raise ValueError("receiver, sender, vectors, distances and radial features must describe the same edges")
        graph = GraphContext(
            self.receiver[indices], self.sender[indices], self.num_nodes,
            vec=self.vec[indices] if self.vec is not None else None,
            dist=self.dist[indices] if self.dist is not None else None,
            rbf=self.rbf[indices] if self.rbf is not None else None,
            cutoff_values=self.cutoff_values[indices] if self.cutoff_values is not None else None,
            batch=self.batch, inputs=self.inputs,
            key=None if self.key is None or step is None else self.key + (step,))
        if csr_key is not None and csr_key in self.cache:
            graph.cache["csr"] = self.cache[csr_key]
        return graph

    def subset(self, indices: Tensor, key: Optional[Hashable] = None) -> "GraphContext":
        """A graph indexed locally in the order given by edge indices or a mask.

        ``key`` is a constant naming the selection within this graph. Named
        subsets are cached and can serve as cache keys for pair tables; an
        unnamed subset is rebuilt on every call.
        """
        if key is None:
            return self._select(indices, None, None)
        return self.cached(("subset", key), lambda: self._select(indices, ("subset", key), None))

    def within(self, cutoff: Optional[float]) -> "GraphContext":
        """Edges strictly inside the cutoff, retaining input edge order."""
        if cutoff is None:
            return self
        return self.cached(("within", cutoff), lambda: self._select(
            self.edge_indices(cutoff, strict=True), ("within", cutoff), ("csr", ("edge_indices", cutoff, True))))

    def _closest(self, cutoff, max_neighbors):
        """Nearest incoming edges inside the cutoff and whether the cap removed any."""
        if max_neighbors < 1 or cutoff <= 0:
            raise ValueError("max_neighbors and cutoff must be positive")

        def select():
            indices = self.edge_indices(cutoff, strict=True)
            counts = self.csr(indices, key=("edge_indices", cutoff, True)).counts
            if indices.numel() and int(counts.max()) > max_neighbors:
                rank = self.cached("distance_rank", lambda: receiver_rank(
                    self.receiver, self.edge_length(), self.num_nodes))
                return indices[rank[indices] < max_neighbors], True
            return indices, False
        return self.cached(("closest", cutoff, max_neighbors), select)

    def closest_indices(self, cutoff, max_neighbors):
        """Indices of the nearest incoming edges strictly inside the cutoff."""
        return self._closest(cutoff, max_neighbors)[0]

    def closest(self, cutoff, max_neighbors):
        def build():
            indices, capped = self._closest(cutoff, max_neighbors)
            csr_key = None if capped else ("csr", ("edge_indices", cutoff, True))
            return self._select(indices, ("closest", cutoff, max_neighbors), csr_key)
        return self.cached(("closest_graph", cutoff, max_neighbors), build)

    def _relative(self, other: "GraphContext") -> Optional[Hashable]:
        """A constant naming ``other`` for this graph's cache, or None."""
        if other is self:
            return "self"
        if self.key is None or other.key is None:
            return None
        return other.key

    def cross_pairs(self, other):
        """For each edge of this graph, enumerate all of the other graph's incoming edges."""
        def build():
            return cross_receiver_pairs(self.receiver, other.receiver, self.num_nodes, csr=other.csr())
        name = self._relative(other)
        return build() if name is None else self.cached(("cross_pairs", name), build)

    def triplets(self, kind, *, cutoff=None, include_self=False, strict=False, exclude_backtracking=False,
                 exclude_same_sender=False):
        """Ordered edge pairs in receiver, directed path, or DGL line-graph order."""
        if kind == "receiver":
            if exclude_same_sender:
                def distinct():
                    first, second = self.receiver_pairs(cutoff, include_self, strict)
                    keep = self.sender[first] != self.sender[second]
                    return EdgePairs(first[keep], second[keep])
                return self.cached(("receiver_pairs", cutoff, include_self, strict, "distinct_sender"), distinct)
            return self.receiver_pairs(cutoff, include_self, strict)
        if kind in ("path", "line_graph"):
            return self.path_pairs("atom" if kind == "line_graph" else "outgoing", exclude_backtracking)
        raise ValueError(f"unknown triplet kind {kind!r}")

    def quadruplets(self, interaction=None, *, images=False):
        """Dihedrals with both outer edges in this graph and a central interaction edge."""
        from .quadruplets import dihedral_quadruplets

        if interaction is None:
            interaction = self
        if "_prepared_cutoff" in interaction.cache:
            key = ("_prepared_quads", interaction.cache["_prepared_cutoff"], images)
            if key in self.cache:
                return self.cache[key]
        name = self._relative(interaction)
        if name is None:
            return dihedral_quadruplets(self, interaction, images)
        return self.cached(("quadruplets", name, images), lambda: dihedral_quadruplets(self, interaction, images))

    def reverse_edge(self, *, atol=_OPPOSITE_TOL, rtol=0.0, metric="l2", distinct=False) -> Tensor:
        return self.cached(("reverse_edge", atol, rtol, metric, distinct), lambda: reverse_edge_index(
            self.receiver, self.sender, self.vec, self.num_nodes,
            atol=atol, rtol=rtol, metric=metric, distinct=distinct))

    def receiver_pairs(self, cutoff: Optional[float] = None, include_self: bool = False,
                       strict: bool = False) -> EdgePairs:
        """Edge pairs sharing a receiver, optionally inside a three-body ``cutoff``.

        ``strict`` uses ``dist < cutoff`` instead of ``dist <= cutoff``.
        """
        def build() -> EdgePairs:
            subset = None if cutoff is None else self.edge_indices(cutoff, strict)
            csr = self.csr(subset, key=None if cutoff is None else ("edge_indices", cutoff, strict))
            return receiver_pairs(self.receiver, self.num_nodes, subset, include_self, csr=csr)

        return self.cached(("receiver_pairs", cutoff, include_self, strict), build)

    def path_pairs(self, order: str = "outgoing", exclude_backtracking: bool = False) -> EdgePairs:
        def build():
            incoming, outgoing = path_pairs(self.receiver, self.sender, self.num_nodes, order, csr=self.csr())
            if exclude_backtracking:
                keep = self.sender[incoming] != self.receiver[outgoing]
                if order == "atom":
                    # DGL line graphs allow a return to a different periodic image.
                    gap = (self.vec[incoming] + self.vec[outgoing]).abs().amax(dim=-1)
                    keep = keep | (gap > _OPPOSITE_TOL)
                return EdgePairs(incoming[keep], outgoing[keep])
            return EdgePairs(incoming, outgoing)

        return self.cached(("path_pairs", order, exclude_backtracking), build)

    def num_graphs(self) -> int:
        if self.batch is None or self.batch.numel() == 0:
            return 1
        return self.cached("num_graphs", lambda: int(self.batch.max().item()) + 1)
