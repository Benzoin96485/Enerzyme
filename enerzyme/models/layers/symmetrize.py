"""DeePMD embedding-matrix symmetrization.

The contraction is the one used by the Deep Potential descriptor and by
DPA3 (Zhang et al., arXiv:2506.01686, Eqs. 17–19): an invariant neighbor
feature and a Cartesian vector form a matrix, the leading rows are kept,
and the Gram matrix over the three Cartesian components is flattened.
Dividing by 3 is the released DPA3 normalization of that inner product.
The map is invariant under rotations of the vector and under a permutation
of the neighbors.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..functional import reduce_edge


def dp_symmetrize(
    invariant: Tensor,
    vector: Tensor,
    owner: Tensor,
    num_nodes: int,
    axis_neuron: int,
    inv_sqrt_neighbors: float,
) -> Tensor:
    """Flattened Gram matrix ``(N, axis_neuron * F)``.

    ``invariant`` is ``(E, F)`` and ``vector`` is ``(E, 3)``. Both already
    include whatever radial gate the caller wants inside the sum. The
    neighbor scale multiplies the invariant once, which is
    :math:`N_m^{-1/2}` in the DPA3 symmetrizer. ``owner`` indexes the atom
    that receives each row.
    """
    if invariant.ndim != 2 or vector.ndim != 2 or vector.shape[-1] != 3:
        raise ValueError(
            "dp_symmetrize expects invariant (E, F) and vector (E, 3), "
            f"got {tuple(invariant.shape)} and {tuple(vector.shape)}"
        )
    if invariant.shape[0] != vector.shape[0] or invariant.shape[0] != owner.shape[0]:
        raise ValueError("invariant, vector, and owner must share the edge count")
    feat_dim = int(invariant.shape[-1])
    if axis_neuron < 1 or axis_neuron > feat_dim:
        raise ValueError(
            f"axis_neuron must lie in 1..F ({feat_dim}), got {axis_neuron}"
        )
    out_dim = axis_neuron * feat_dim
    if invariant.shape[0] == 0:
        return invariant.new_zeros(num_nodes, out_dim)

    scale = invariant.new_tensor(float(inv_sqrt_neighbors))
    scaled = invariant * scale
    pair = vector.unsqueeze(-1) * scaled.unsqueeze(-2)
    flat = pair.reshape(pair.shape[0], 3 * feat_dim)
    gathered = reduce_edge(flat, owner, (num_nodes, 3 * feat_dim))
    matrix = gathered.view(num_nodes, 3, feat_dim)
    leading = matrix[:, :, :axis_neuron]
    gram = torch.matmul(leading.transpose(1, 2), matrix) / 3.0
    return gram.reshape(num_nodes, out_dim)
