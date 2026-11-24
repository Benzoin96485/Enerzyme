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

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Sequential, SiLU

from ..mp.ops import scatter_sum
from ..layers.fragments import (
    build_kmeans_fragments,
    build_spectral_fragments,
    centers_from_ids,
    flatten_cluster_ids,
    weighted_fragment_centers,
)
from .interaction import VecLayerNorm, _xavier_linear


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
        dist = torch.linalg.vector_norm(delta, dim=-1)
        within = dist <= limit
        if not bool(within.any()):
            continue
        local_atom, local_fragment = torch.where(within)
        chosen = dist[local_atom, local_fragment]
        vec = delta[local_atom, local_fragment]
        direction = vec / chosen.clamp_min(1e-12).unsqueeze(-1)
        direction = torch.where(
            (chosen < 1e-8).unsqueeze(-1), torch.zeros_like(direction), direction
        )
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


class FragmentBipartiteBlock(Module):
    """One pre-norm fragment→atom update. Returns residuals ``(dx, dvec)``."""

    def __init__(self, hidden_channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        num_heads = int(num_heads)
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"dim_embedding ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout_p = float(dropout)
        self.norm_node = LayerNorm(hidden_channels)
        self.norm_group = LayerNorm(hidden_channels)
        self.norm_node_vec = VecLayerNorm(
            hidden_channels, trainable=False, norm_type="max_min"
        )
        self.norm_group_vec = VecLayerNorm(
            hidden_channels, trainable=False, norm_type="max_min"
        )
        self.q_proj = Linear(hidden_channels, hidden_channels)
        self.k_proj = Linear(hidden_channels, hidden_channels)
        self.v_proj = Linear(hidden_channels, hidden_channels)
        self.mlp_pos = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mlp_vec = Sequential(
            Linear(hidden_channels, hidden_channels),
            SiLU(),
            Linear(hidden_channels, hidden_channels),
        )
        self.mix = ModuleList(
            Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)
        )
        self.drop_scalar = Dropout(self.dropout_p)
        self.drop_vector = Dropout(self.dropout_p)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.norm_node.reset_parameters()
        self.norm_group.reset_parameters()
        self.norm_node_vec.reset_parameters()
        self.norm_group_vec.reset_parameters()
        for module in self.modules():
            if isinstance(module, Linear):
                _xavier_linear(module)

    def forward(
        self,
        node_scalar: Tensor,
        node_vector: Tensor,
        group_scalar: Tensor,
        group_vector: Tensor,
        idx_atom: Tensor,
        idx_fragment: Tensor,
        edge_attr: Tensor,
        direction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        n_atoms = int(node_scalar.shape[0])
        scalar = self.norm_node(node_scalar)
        vector = self.norm_node_vec(node_vector)
        group_scalar = self.norm_group(group_scalar)
        group_vector = self.norm_group_vec(group_vector)
        if self.dropout_p > 0.0:
            group_scalar = self.drop_scalar(group_scalar)
            group_vector = self.drop_vector(group_vector)

        n_edges = int(idx_atom.shape[0])
        if n_edges == 0:
            return node_scalar.new_zeros(n_atoms, self.hidden_channels), node_vector.new_zeros(
                n_atoms, 3, self.hidden_channels
            )

        query = self.q_proj(scalar).reshape(n_atoms, self.num_heads, self.head_dim)
        key = self.k_proj(group_scalar).reshape(-1, self.num_heads, self.head_dim)
        value = self.v_proj(group_scalar).reshape(-1, self.num_heads, self.head_dim)
        query_i = query.index_select(0, idx_atom)
        key_j = key.index_select(0, idx_fragment)
        value_j = value.index_select(0, idx_fragment)
        filt = edge_attr.reshape(n_edges, self.num_heads, self.head_dim)
        attention = (query_i * key_j * filt).sum(dim=-1) / math.sqrt(self.head_dim)
        attention = F.silu(attention)
        scalar_message = (value_j * attention.unsqueeze(-1)).reshape(
            n_edges, self.hidden_channels
        )
        fragment_vector = group_vector.index_select(0, idx_fragment)
        vector_message = self.mlp_pos(scalar_message).unsqueeze(1) * direction.unsqueeze(-1)
        vector_message = vector_message + self.mlp_vec(scalar_message).unsqueeze(1) * fragment_vector
        aggregated_scalar = scatter_sum(scalar_message, idx_atom, n_atoms)
        aggregated_vector = scatter_sum(vector_message, idx_atom, n_atoms)
        left = self.mix[2](vector)
        right = self.mix[3](vector)
        dot = (left * right).sum(dim=1)
        delta_scalar = dot * self.mix[4](aggregated_scalar) + self.mix[5](aggregated_scalar)
        delta_vector = aggregated_vector + self.mix[0](aggregated_scalar).unsqueeze(1) * self.mix[1](vector)
        return delta_scalar, delta_vector
