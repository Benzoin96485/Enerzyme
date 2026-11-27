"""Edge frames and latitude–longitude splats for SpinConv.

Shuaibi et al. (arXiv:2106.09575) put a sphere on the source atom of each
directed edge. The polar axis is the bond. A 1D convolution then runs along
longitude, and average pooling removes that roll. ``idx_i`` is the
destination and ``idx_j`` is the source, with ``vij = R[source] - R[destination]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..mp.basis.envelope import cosine_transition
from ..mp.graph import EdgePairs, path_pairs, receiver_csr, receiver_rank


def symmetric_neighbor_mask(
    idx_i: Tensor,
    idx_j: Tensor,
    distance: Tensor,
    num_nodes: int,
    max_neighbors: int,
) -> Tensor:
    """Keep an undirected pair when either direction is among the closest edges.

    Incoming edges are ranked by distance at their destination. A pair is
    kept when at least one direction sits in that destination's
    ``max_neighbors`` shortest edges. Distinct periodic images of the same
    two atoms share one pair id, matching the released hash
    ``min(i, j) * N + max(i, j)``. Ties break by the original edge order.
    The mask is all true when every destination already has degree at most
    ``max_neighbors``.
    """
    if int(max_neighbors) < 1:
        raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")
    if idx_i.shape != idx_j.shape or idx_i.shape != distance.shape:
        raise ValueError(
            "idx_i, idx_j, and distance must share a shape, got "
            f"{tuple(idx_i.shape)}, {tuple(idx_j.shape)}, {tuple(distance.shape)}"
        )
    if idx_i.numel() == 0 or int(num_nodes) == 0:
        return torch.ones(idx_i.shape[0], dtype=torch.bool, device=idx_i.device)
    counts = torch.bincount(idx_i, minlength=int(num_nodes))
    if int(counts.max().item()) <= int(max_neighbors):
        return torch.ones(idx_i.shape[0], dtype=torch.bool, device=idx_i.device)

    selected = receiver_rank(idx_i, distance, int(num_nodes)) < int(max_neighbors)

    lo = torch.minimum(idx_i, idx_j).to(torch.long)
    hi = torch.maximum(idx_i, idx_j).to(torch.long)
    pair = lo * int(num_nodes) + hi
    selected_pairs = torch.unique(pair[selected])
    pos = torch.searchsorted(selected_pairs, pair)
    in_range = pos < selected_pairs.shape[0]
    pos = pos.clamp(max=max(int(selected_pairs.shape[0]) - 1, 0))
    return in_range & (selected_pairs[pos] == pair)


def incoming_edge_table(idx_i: Tensor, num_nodes: int) -> Tensor:
    """Edge indices arriving at each destination, padded with ``-1``.

    Shape ``(num_nodes, max_degree)``. The row order follows a stable sort
    on the destination, so the table does not depend on a prior grouping
    of the neighbor list.
    """
    if int(num_nodes) == 0:
        return idx_i.new_zeros((0, 0))
    if idx_i.numel() == 0:
        return idx_i.new_full((int(num_nodes), 0), -1)
    counts, offsets, order = receiver_csr(idx_i, int(num_nodes))
    max_degree = int(counts.max().item())
    local = torch.arange(idx_i.numel(), device=idx_i.device) - offsets[idx_i[order]]
    table = idx_i.new_full((int(num_nodes) * max_degree,), -1)
    table[idx_i[order] * max_degree + local] = order
    return table.view(int(num_nodes), max_degree)


def source_neighbor_pairs(idx_i: Tensor, idx_j: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Pair each edge with the edges that arrive at its source.

    Returns ``(source_edge, query_edge)``. ``query_edge`` is the edge whose
    sphere is being painted. ``source_edge`` is an incoming edge of
    ``idx_j[query_edge]``, the source atom.
    """
    return path_pairs(idx_i, idx_j, num_nodes)


# Sine of the angle between the bond and the second axis. Below this the
# cross product is numerical noise, and ``_unit`` would return a zero normal.
_FRAME_SINE_FLOOR = 1e-5


def _unit(vector: Tensor) -> Tensor:
    length = vector.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return vector / length


def _fallback_normal(polar: Tensor) -> Tensor:
    """A vector perpendicular to ``polar`` when the second axis is unusable.

    Lab ``z`` is the reference. When ``polar`` itself lies along ``z``, lab
    ``x`` is used. The longitude origin is then a lab convention. Average
    pooling over longitude removes that origin from the edge message.
    """
    z_axis = polar.new_zeros(polar.shape)
    z_axis[..., 2] = 1.0
    crossed = torch.cross(polar, z_axis, dim=-1)
    x_axis = polar.new_zeros(polar.shape)
    x_axis[..., 0] = 1.0
    crossed_x = torch.cross(polar, x_axis, dim=-1)
    along_z = crossed.norm(dim=-1, keepdim=True) < _FRAME_SINE_FLOOR
    return torch.where(along_z, crossed_x, crossed)


def _receiver_sum(vectors: Tensor, receiver: Tensor, num_nodes: int) -> Tensor:
    averaged = vectors.new_zeros((num_nodes, 3))
    if vectors.numel():
        averaged.index_add_(0, receiver, vectors)
    return averaged


@torch.library.custom_op("enerzyme::spinconv_frame_sums", mutates_args=())
def _frame_sums(vij: Tensor, distance: Tensor, receiver: Tensor, cutoff: float,
                num_nodes: int, num_blocks: int) -> Tensor:
    """Independently sample the detached receiver reduction for each spin block.

    The CUDA reduction order affects longitude at the poles. Keeping the calls
    together inside this operator prevents a compiler from reusing one sample.
    """
    weighted = vij * cosine_transition(distance, cutoff).unsqueeze(-1)
    return torch.stack([_receiver_sum(weighted, receiver, num_nodes) for _ in range(num_blocks)])


@_frame_sums.register_fake
def _frame_sums_shape(vij, distance, receiver, cutoff, num_nodes, num_blocks):
    return vij.new_empty((num_blocks, num_nodes, 3))


class _SpinFrameSamples(NamedTuple):
    pairs: EdgePairs
    receiver_sums: Tensor


def _independent_frames(vij: Tensor) -> bool:
    return vij.is_cuda and not torch.are_deterministic_algorithms_enabled()


def edge_spin_frames(
    vij: Tensor,
    idx_i: Tensor,
    distance: Tensor,
    cutoff: float,
    num_nodes: int,
    *,
    averaged: Optional[Tensor] = None,
) -> Tensor:
    """Rotation whose rows are the local SpinConv axes.

    ``frames @ v`` maps a global column vector into the edge frame. Local
    ``+x`` is the unit ``vij``, from the destination toward the source.
    The sphere is centered on the source, so the direction from the source
    back to the destination has local ``x = -1``. Longitude is fixed by the
    cosine-weighted sum of ``vij`` at the destination
    (:func:`enerzyme.models.mp.basis.envelope.cosine_transition`). That sum is parallel
    to the bond for a degree-1 destination, and it is zero when every edge
    into the destination sits at the cutoff. The normal is then the cross
    product of the bond with a lab axis, so the frame stays a rotation. The
    released lab-frame ``1e-4`` nudge is not added: a non-degenerate second
    axis still rotates with the molecule.
    """
    if vij.ndim != 2 or vij.shape[-1] != 3:
        raise ValueError(f"vij must have shape (E, 3), got {tuple(vij.shape)}")
    if distance.shape != idx_i.shape:
        raise ValueError(
            f"distance shape {tuple(distance.shape)} != idx_i shape {tuple(idx_i.shape)}"
        )
    if averaged is None:
        weight = cosine_transition(distance, float(cutoff)).unsqueeze(-1)
        averaged = _receiver_sum(vij * weight, idx_i, int(num_nodes))
    polar = _unit(vij)
    second = _unit(averaged[idx_i])
    raw_normal = torch.cross(polar, second, dim=-1)
    degenerate = raw_normal.norm(dim=-1, keepdim=True) < _FRAME_SINE_FLOOR
    normal = _unit(torch.where(degenerate, _fallback_normal(polar), raw_normal))
    tangent = _unit(torch.cross(polar, normal, dim=-1))
    return torch.stack((polar, tangent, normal), dim=1)


def project_neighbors(
    vij: Tensor,
    frames: Tensor,
    source_edge: Tensor,
    query_edge: Tensor,
) -> Tensor:
    """Local coordinates of each source-neighbor direction.

    ``frames[query]`` rotates the unit ``vij[source]``. The result has
    shape ``(P, 3)``.
    """
    if source_edge.numel() == 0:
        return vij.new_zeros((0, 3))
    direction = F.normalize(vij[source_edge], dim=-1, eps=1e-12)
    return torch.bmm(frames[query_edge], direction.unsqueeze(-1)).squeeze(-1)


def bilinear_sphere_index(
    local: Tensor,
    query_edge: Tensor,
    n_lat: int,
    n_long: int,
) -> tuple[Tensor, Tensor]:
    """Four bilinear corners and weights on the latitude–longitude grid.

    Longitude is ``atan2(y, z)`` mapped onto ``[0, n_long)``. Latitude is
    ``asin(x)`` mapped onto ``[0, n_lat - 1]`` and shrunk by the released
    ``0.99 y + 0.005`` so a point does not sit on the last row. Both axes
    wrap with modulo, matching the released indexer. Index shape is
    ``(4, P)`` and the four weights sum to 1.
    """
    if int(n_lat) < 2 or int(n_long) < 2:
        raise ValueError(f"sphere grid must be at least 2x2, got {n_lat} x {n_long}")
    longitude = torch.atan2(local[:, 1], local[:, 2])
    longitude = (longitude + math.pi) / (2.0 * math.pi) * int(n_long)
    latitude = local[:, 0].clamp(-1.0, 1.0)
    latitude = (torch.asin(latitude) + (math.pi / 2.0)) / math.pi
    latitude = (0.99 * latitude + 0.005) * (int(n_lat) - 1.0)

    x0 = torch.floor(longitude).long()
    x_del = longitude - x0.to(longitude.dtype)
    x0 = x0 % int(n_long)
    x1 = (x0 + 1) % int(n_long)
    y0 = torch.floor(latitude).long()
    y_del = latitude - y0.to(latitude.dtype)
    y0 = y0 % int(n_lat)
    y1 = (y0 + 1) % int(n_lat)

    base = query_edge * int(n_lat) * int(n_long)
    index = torch.stack(
        (
            base + y0 * int(n_long) + x0,
            base + y0 * int(n_long) + x1,
            base + y1 * int(n_long) + x0,
            base + y1 * int(n_long) + x1,
        )
    )
    weight = torch.stack(
        (
            (1.0 - x_del) * (1.0 - y_del),
            x_del * (1.0 - y_del),
            (1.0 - x_del) * y_del,
            x_del * y_del,
        )
    )
    # The bond partner lies on the pole. ``asin(±1)`` and ``atan2(0, 0)`` have
    # no finite derivative there, and a longitude chart is discontinuous under
    # an infinitesimal bend. The released model drops this Jacobian when
    # ``force_estimator`` is ``grad`` (``with torch.no_grad()`` around the
    # bilinear weights). Holding the weights fixed keeps ``Fa = -∇E`` finite:
    # positions still enter through the distances and the message values.
    return index, weight.detach()


def _projection_table(vij, receiver, distance, cutoff, num_nodes, n_lat, n_long,
                      source, query, averaged):
    frames = edge_spin_frames(vij, receiver, distance, cutoff, num_nodes, averaged=averaged)
    local = project_neighbors(vij, frames, source, query)
    return bilinear_sphere_index(local, query, n_lat, n_long)


@torch.library.custom_op("enerzyme::spinconv_cuda_double_table", mutates_args=(), device_types="cuda")
def _cuda_double_table(vij: Tensor, receiver: Tensor, distance: Tensor, cutoff: float,
                       num_nodes: int, n_lat: int, n_long: int, source: Tensor, query: Tensor,
                       averaged: Optional[Tensor]) -> tuple[Tensor, Tensor]:
    """Preserve native FP64 frame and coordinate arithmetic at the polar grid singularity."""
    return _projection_table(vij, receiver, distance, cutoff, num_nodes, n_lat, n_long,
                             source, query, averaged)


@_cuda_double_table.register_fake
def _cuda_double_table_shape(vij, receiver, distance, cutoff, num_nodes, n_lat, n_long,
                             source, query, averaged):
    return query.new_empty((4, query.numel()), dtype=torch.long), vij.new_empty((4, query.numel()))


@torch.no_grad()
def sphere_projection(vij, receiver, sender, distance, cutoff, num_nodes, n_lat, n_long, *, pairs=None, averaged=None):
    """Graph-dependent interpolation indices and fixed-coordinate weights for the given frame."""
    source, query = source_neighbor_pairs(receiver, sender, num_nodes) if pairs is None else pairs
    if source.numel() == 0:
        return source.new_empty(4, 0), vij.new_empty(4, 0), source
    if torch.compiler.is_compiling() and vij.is_cuda and vij.dtype == torch.float64:
        # Even with identical receiver sums, fusing the frame and dot products
        # changes longitude at the pole. Only the detached lookup needs native
        # arithmetic; the splat, convolution and force path still compile.
        index, weight = _cuda_double_table(vij.detach(), receiver, distance.detach(), cutoff, num_nodes,
                                           n_lat, n_long, source, query, averaged)
    else:
        index, weight = _projection_table(vij, receiver, distance, cutoff, num_nodes, n_lat, n_long,
                                          source, query, averaged)
    return index, weight, source


@dataclass(frozen=True)
class SpinProjectionRequest:
    """Symmetric neighbor cap and detached grid interpolation for all spin blocks."""

    max_neighbors: int
    cutoff: float
    n_lat: int
    n_long: int
    num_blocks: int = 1

    @property
    def force_cache_key(self):
        return ("spin_projection", self.max_neighbors, self.cutoff, self.n_lat, self.n_long, self.num_blocks)

    def prepare(self, ctx):
        return ctx.cached(self.force_cache_key, lambda: self._projection(ctx))

    def selected_graph(self, ctx, selected):
        if selected is None:
            return ctx
        return ctx.subset(selected, key=("spinconv_selected", self.max_neighbors))

    def _projection(self, ctx):
        i, j = ctx.receiver, ctx.sender
        if i.ndim != 1 or j.shape != i.shape or ctx.dist.shape != i.shape or ctx.vec.shape != (i.shape[0], 3):
            raise ValueError("neighbor indices, distances and vectors must describe the same edges")
        valid = ((i >= 0) & (j >= 0) & (i < ctx.num_nodes) & (j < ctx.num_nodes)).all()
        if torch.compiler.is_compiling():
            torch._assert_async(valid, "edge indices are outside the atom range")
        elif not bool(valid):
            raise ValueError("edge indices are outside the atom range")

        selected = None
        if i.numel():
            mask = symmetric_neighbor_mask(i, j, ctx.dist, ctx.num_nodes, self.max_neighbors)
            if not bool(mask.all()):
                selected = torch.where(mask)[0]
        edges = self.selected_graph(ctx, selected)
        pairs = edges.triplets("path")
        if self.num_blocks > 1 and _independent_frames(edges.vec):
            sums = _frame_sums(edges.vec.detach(), edges.dist.detach(), edges.receiver,
                               self.cutoff, ctx.num_nodes, self.num_blocks)
            return selected, _SpinFrameSamples(pairs, sums)
        projection = sphere_projection(
            edges.vec, edges.receiver, edges.sender, edges.dist, self.cutoff, ctx.num_nodes,
            self.n_lat, self.n_long, pairs=pairs,
        )
        return selected, projection


def splat_sphere(
    messages: Tensor,
    n_edges: int,
    n_lat: int,
    n_long: int,
    index: Tensor,
    weight: Tensor,
    source_edge: Tensor,
) -> Tensor:
    """Bilinear splat of edge messages onto ``(E, C, lat, long)``."""
    channels = int(messages.shape[-1])
    grid = messages.new_zeros((int(n_edges) * int(n_lat) * int(n_long), channels))
    if source_edge.numel():
        values = messages[source_edge]
        for corner in range(4):
            grid.index_add_(0, index[corner], values * weight[corner].unsqueeze(-1))
    grid = grid.view(int(n_edges), int(n_lat) * int(n_long), channels)
    grid = grid.transpose(1, 2).contiguous()
    return grid.view(int(n_edges), channels, int(n_lat), int(n_long))
