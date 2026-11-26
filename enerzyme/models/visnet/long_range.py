"""Atom–fragment long-range block for ViSNet-LSRM.

The update follows ``LongShortIneractModel_dis_direct_vector2_drop`` in
``liyy2/LSR-MP`` (the ``Visnorm_shared_LSRMNorm2_2branchSerial`` model).
Query is the atom scalar, key and value are the fragment scalar. Attention is
SiLU of the distance-filtered product, divided by ``sqrt(head_dim)``. The
vector message is gated along ``(P_j - p_i) / r`` and along the fragment
vector. The released code uses that direction; the paper writes
``(p_i - P_j) / r``.

The radial features are an exponential-normal basis of the raw distance.
Section 4.1 also writes a size-dependent distance ``w(Z_i, H_j) r + b``.
That map is not in the released class, so this block does not apply it.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor

from ..mp.nonlocal_ import FragmentLR as FragmentBipartiteBlock  # noqa: F401
from ..layers.fragments import (
    build_kmeans_fragments,
    build_spectral_fragments,
    centers_from_ids,
    flatten_cluster_ids,
    weighted_fragment_centers,
)


_CENTERS = {
    "atomic_number": "atomic_number",
    "z": "atomic_number",
    "com": "atomic_number",
    "center_of_mass": "atomic_number",
    "geometric": "geometric",
    "mean": "geometric",
}
_FRAGMENT_MODES = ("kmeans", "spectral", "precomputed")


def normalize_center(value: object) -> str:
    """Return ``atomic_number`` or ``geometric``.

    ``center_of_mass`` is the released name. That path weights coordinates by
    atomic number, which is the proxy the LSR-MP code uses for mass.
    """
    key = str(value).strip().lower().replace("-", "_")
    if key not in _CENTERS:
        choices = "atomic_number, center_of_mass, geometric"
        raise ValueError(f"center must be one of {choices}, got {value!r}")
    return _CENTERS[key]


def normalize_fragment_mode(value: object) -> str:
    """Return ``kmeans``, ``spectral``, or ``precomputed``."""
    key = str(value).strip().lower().replace("-", "_")
    if key == "rdkit":
        key = "precomputed"
    if key not in _FRAGMENT_MODES:
        choices = ", ".join(_FRAGMENT_MODES)
        raise ValueError(f"fragment_mode must be one of {choices}, got {value!r}")
    return key


def _fragment_distance(delta: Tensor) -> Tensor:
    # A singleton's center follows its atom, so its own distance is constant.
    # Differentiating the Euclidean norm at zero would introduce 0/0 in a Hessian.
    coincident = (delta == 0).all(dim=-1)
    safe = torch.where(coincident.unsqueeze(-1), torch.ones_like(delta), delta)
    distance = torch.linalg.vector_norm(safe, dim=-1)
    return torch.where(coincident, torch.zeros_like(distance), distance)


def _fragment_direction(delta: Tensor, distance: Tensor) -> Tensor:
    unit = delta / distance.clamp_min(1e-12).unsqueeze(-1)
    return torch.where((distance < 1e-8).unsqueeze(-1), torch.zeros_like(unit), unit)


def bipartite_radius_edges(
    atom_pos: Tensor,
    cluster_pos: Tensor,
    batch_seg: Tensor,
    cluster_batch: Tensor,
    cutoff: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Atom–fragment pairs in the same graph with distance ``<= cutoff``.

    The direction is ``(P_j - p_i) / r``. A pair sitting on the fragment
    center contributes a zero direction so the positional gate vanishes.
    Own fragments are kept: the released interaction graph is every fragment
    inside the cutoff, including the atom's own fragment.

    Returns ``idx_atom``, ``idx_fragment``, ``distance``, ``direction``.
    """
    device = atom_pos.device
    num_atoms = int(atom_pos.shape[0])
    if num_atoms == 0 or cluster_pos.shape[0] == 0:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return empty_i, empty_i, atom_pos.new_zeros(0), atom_pos.new_zeros(0, 3)

    idx_atom = []
    idx_fragment = []
    distances = []
    directions = []
    num_graphs = int(batch_seg.max().item()) + 1
    limit = float(cutoff)
    for graph in range(num_graphs):
        atoms = torch.where(batch_seg == graph)[0]
        fragments = torch.where(cluster_batch == graph)[0]
        if atoms.numel() == 0 or fragments.numel() == 0:
            continue
        # [Na, Nc, 3], fragment minus atom, matching the released ``u_ij``.
        delta = cluster_pos[fragments].unsqueeze(0) - atom_pos[atoms].unsqueeze(1)
        dist = _fragment_distance(delta)
        within = dist <= limit
        if not bool(within.any()):
            continue
        local_atom, local_fragment = torch.where(within)
        chosen = dist[local_atom, local_fragment]
        vec = delta[local_atom, local_fragment]
        direction = _fragment_direction(vec, chosen)
        idx_atom.append(atoms[local_atom])
        idx_fragment.append(fragments[local_fragment])
        distances.append(chosen)
        directions.append(direction)

    if not idx_atom:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return empty_i, empty_i, atom_pos.new_zeros(0), atom_pos.new_zeros(0, 3)
    return (
        torch.cat(idx_atom),
        torch.cat(idx_fragment),
        torch.cat(distances),
        torch.cat(directions),
    )


def resolve_long_range_fragments(
    Ra: Tensor,
    batch_seg: Tensor,
    fragment_mode: str,
    center: str,
    Za: Optional[Tensor],
    cluster_ids: Optional[Tensor],
    min_nodes_per_group: int,
    random_state: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Flat fragment ids, live centers, and the graph id of each fragment.

    Clustering labels are detached. Centers are recomputed from ``Ra`` so a
    force can move a fragment when its atoms move. ``atomic_number`` weights
    each atom by ``Za``. ``geometric`` is the uniform mean.
    """
    mode = normalize_fragment_mode(fragment_mode)
    if mode == "precomputed":
        if cluster_ids is None:
            raise ValueError(
                "fragment_mode='precomputed' requires cluster_ids "
                "(offline BRICS or another fragment label)"
            )
        local = cluster_ids.reshape(-1).long()
        if local.shape[0] != Ra.shape[0]:
            raise ValueError(
                f"cluster_ids must have length {Ra.shape[0]}, got {local.shape[0]}"
            )
        local, _, cluster_batch = centers_from_ids(Ra, local, batch_seg)
    elif mode == "kmeans":
        local, _, _ = build_kmeans_fragments(
            Ra,
            batch_seg,
            min_nodes_per_group=min_nodes_per_group,
            random_state=random_state,
        )
        local, _, cluster_batch = centers_from_ids(Ra, local, batch_seg)
    else:
        local, _, _ = build_spectral_fragments(
            Ra,
            batch_seg,
            min_nodes_per_group=min_nodes_per_group,
            random_state=random_state,
        )
        local, _, cluster_batch = centers_from_ids(Ra, local, batch_seg)
    flat_ids = flatten_cluster_ids(local, batch_seg, cluster_batch)
    weights = None
    if center == "atomic_number":
        if Za is None:
            raise ValueError("center='atomic_number' requires atomic numbers Za")
        weights = Za.reshape(-1)
    centers = weighted_fragment_centers(Ra, flat_ids, weights)
    return flat_ids, centers, cluster_batch


@torch.library.custom_op("enerzyme::lsrm_fragment_indices", mutates_args=())
def _fragment_indices(
    positions: Tensor, batch: Tensor, mode: str, center: str,
    numbers: Optional[Tensor], labels: Optional[Tensor],
    min_nodes: int, seed: int, cutoff: float,
) -> Tuple[Tensor, Tensor]:
    """Discrete clustering and radius selection, independent of the force pullback."""
    ids, centers, cluster_batch = resolve_long_range_fragments(
        positions, batch, mode, center, numbers, labels, min_nodes, seed
    )
    atoms, fragments, _, _ = bipartite_radius_edges(
        positions, centers, batch, cluster_batch, cutoff
    )
    return ids, torch.stack((atoms, fragments))


@_fragment_indices.register_fake
def _fragment_index_shapes(positions, batch, mode, center, numbers, labels, min_nodes, seed, cutoff):
    n_edges = torch.library.get_ctx().new_dynamic_size()
    return torch.empty_like(batch), batch.new_empty((2, n_edges))


class FragmentTopology(NamedTuple):
    """Detached assignments and selected pairs, shared by the force and feature paths."""

    ids: Tensor
    edges: Tensor

    def apply(self, ctx):
        ctx.cache["lsrm_fragment_indices"] = self


def select_fragment_topology(positions, batch, mode, center, numbers, labels, min_nodes, seed, cutoff):
    ids, edges = _fragment_indices(positions.detach(), batch, mode, center, numbers, labels, min_nodes, seed, cutoff)
    return FragmentTopology(ids, edges)


def fragment_geometry(
    positions: Tensor, batch: Tensor, mode: str, center: str,
    numbers: Optional[Tensor], labels: Optional[Tensor],
    min_nodes: int, seed: int, cutoff: float, prepared: Optional[FragmentTopology] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the discrete indices and their differentiable distances and directions."""
    if prepared is not None or torch.compiler.is_compiling():
        if prepared is None:
            ids, edges = _fragment_indices(
                positions.detach(), batch, mode, center, numbers, labels, min_nodes, seed, cutoff
            )
        else:
            ids, edges = prepared.ids, prepared.edges
        atoms, fragments = edges.unbind(0)
        if not torch.compiler.is_compiling() and not atoms.numel():
            return ids, atoms, fragments, positions.new_zeros(0), positions.new_zeros((0, 3))
        weights = numbers if center == "atomic_number" else None
        centers = weighted_fragment_centers(positions, ids, weights)
        delta = centers[fragments] - positions[atoms]
        distance = _fragment_distance(delta)
        direction = _fragment_direction(delta, distance)
    else:
        ids, centers, cluster_batch = resolve_long_range_fragments(
            positions, batch, mode, center, numbers, labels, min_nodes, seed
        )
        atoms, fragments, distance, direction = bipartite_radius_edges(
            positions, centers, batch, cluster_batch, cutoff
        )
    return ids, atoms, fragments, distance, direction
