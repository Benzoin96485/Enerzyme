"""Dihedral geometry for two directed edge sets on the same atoms."""

from typing import NamedTuple

import torch
from torch import Tensor

from .graph import _OPPOSITE_TOL

_INV_SQRT_EPS = 1e-12


class QuadrupletGraph(NamedTuple):
    """Quadruplets ``c → a − b ← d`` on one neighbor list."""

    edge_ca: Tensor
    edge_db: Tensor
    int_local: Tensor
    angle_cab: Tensor
    angle_abd: Tensor
    dihedral: Tensor
    atom_c: Tensor
    atom_a: Tensor
    atom_b: Tensor
    atom_d: Tensor


def angle_between(left: Tensor, right: Tensor) -> Tensor:
    """Unsigned angle in ``[0, π]`` between row vectors."""
    if left.numel() == 0:
        return left.new_zeros(0)
    sine = torch.linalg.cross(left, right, dim=-1).norm(dim=-1)
    sine = torch.maximum(sine, sine.new_tensor(1e-9))
    cosine = (left * right).sum(dim=-1)
    return torch.atan2(sine, cosine)


def _reject(vec: Tensor, normal: Tensor) -> Tensor:
    denom = (normal * normal).sum(dim=-1, keepdim=True).clamp(min=_INV_SQRT_EPS)
    coef = (vec * normal).sum(dim=-1, keepdim=True) / denom
    return vec - coef * normal


def _distinct_image(atom_a: Tensor, vec_a: Tensor, atom_b: Tensor, vec_b: Tensor) -> Tensor:
    """True when the two neighbor images are not the same atom in the same cell."""
    gap = (vec_a - vec_b).detach().square().sum(dim=-1).sqrt()
    return (atom_a != atom_b) | (gap > _OPPOSITE_TOL)


def dihedral_quadruplets(embedding, interaction, images=False):
    """Edge triples c→a, b→a, d→b with unsigned angles at the central bond.

    Outer indices address `embedding`; the central index addresses
    `interaction`. With `images`, distinct periodic images of the same atom
    may participate. Filtering topology never detaches the output angles.
    """
    idx_i_emb, idx_j_emb = embedding.receiver, embedding.sender
    a_node, b_node = interaction.receiver, interaction.sender
    empty_l = idx_i_emb.new_empty(0)
    empty_f = embedding.vec.new_zeros(0)
    blank = QuadrupletGraph(
        empty_l, empty_l, empty_l, empty_f, empty_f, empty_f,
        empty_l, empty_l, empty_l, empty_l,
    )
    if embedding.num_edges == 0 or interaction.num_edges == 0:
        return blank

    counts, offsets, perm = embedding.csr()
    n_a = counts[a_node]
    n_b = counts[b_node]
    n_quad = n_a * n_b
    total = int(n_quad.sum().item())
    if total == 0:
        return blank

    device = idx_i_emb.device
    int_local = torch.repeat_interleave(torch.arange(interaction.num_edges, device=device), n_quad)
    prefix = torch.cumsum(n_quad, dim=0) - n_quad
    slot = torch.arange(total, device=device) - torch.repeat_interleave(prefix, n_quad)
    n_b_rep = torch.repeat_interleave(n_b, n_quad)
    c_local = torch.div(slot, n_b_rep, rounding_mode="floor")
    d_local = slot - c_local * n_b_rep
    a_rep = torch.repeat_interleave(a_node, n_quad)
    b_rep = torch.repeat_interleave(b_node, n_quad)
    edge_ca = perm[offsets[a_rep] + c_local]
    edge_db = perm[offsets[b_rep] + d_local]
    atom_c = idx_j_emb[edge_ca]
    atom_d = idx_j_emb[edge_db]
    if images:
        vij_emb = embedding.vec
        vij_ab = interaction.vec[int_local]
        vij_ca = vij_emb[edge_ca]
        vij_db = vij_emb[edge_db]
        keep = (
            _distinct_image(atom_c, vij_ca, b_rep, vij_ab)
            & _distinct_image(a_rep, -vij_ab, atom_d, vij_db)
            & _distinct_image(atom_c, vij_ca, atom_d, vij_ab + vij_db)
        )
    else:
        keep = (atom_c != b_rep) & (a_rep != atom_d) & (atom_c != atom_d)
    edge_ca = edge_ca[keep]
    edge_db = edge_db[keep]
    int_local = int_local[keep]
    atom_c = atom_c[keep]
    atom_a = a_rep[keep]
    atom_b = b_rep[keep]
    atom_d = atom_d[keep]
    if edge_ca.numel() == 0:
        return blank

    vij_emb = embedding.vec
    vij_ab = interaction.vec[int_local]
    vij_ca = vij_emb[edge_ca]
    vij_db = vij_emb[edge_db]
    angle_cab = angle_between(vij_ca, vij_ab)
    angle_abd = angle_between(-vij_ab, vij_db)
    dihedral = angle_between(_reject(vij_ca, vij_ab), _reject(vij_db, -vij_ab))
    return QuadrupletGraph(
        edge_ca, edge_db, int_local, angle_cab, angle_abd, dihedral,
        atom_c, atom_a, atom_b, atom_d,
    )
