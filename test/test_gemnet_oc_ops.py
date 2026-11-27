"""GemNet-OC radial basis, neighbor caps, and quadruplet images."""

from __future__ import annotations

import torch
from numpy.testing import assert_allclose

from enerzyme.models.mp.basis.envelope import polynomial_envelope
from enerzyme.models.gemnet.basis import legendre_outer, y_lm0
from enerzyme.models.gemnet.geometry import (
    cap_incoming_mask,
    mixed_triplets,
    quadruplet_graph,
    reverse_edge_index,
    symmetrize_edges,
)
from enerzyme.models.layers.rbf import (
    GaussianRBFLayer,
    gemnet_oc_gaussian,
    gemnet_oc_gaussian_terms,
)


def test_gemnet_oc_gaussian_is_scaled_and_enveloped():
    cutoff = 4.0
    layer = GaussianRBFLayer(
        5, cutoff, apply_cutoff_fn=False, flavor="gemnet-oc", envelope_exponent=5
    )
    dist = torch.tensor([0.0, 1.5, 3.9, 4.0, 4.2])
    got = layer._get_rbf(dist)
    center, coeff = gemnet_oc_gaussian_terms(5)
    scaled = dist / cutoff
    raw = torch.exp(coeff * (scaled.unsqueeze(-1) - center) ** 2)
    env = polynomial_envelope(dist, cutoff, 5).unsqueeze(-1)
    assert_allclose(got.detach().numpy(), (env * raw).detach().numpy(), atol=1e-6)
    assert torch.count_nonzero(got[-2:]) == 0
    shorter = layer.expansion(dist, 2.0)
    assert shorter.shape == got.shape
    assert not torch.allclose(shorter[1], got[1])
    assert torch.count_nonzero(shorter[2:]) == 0
    try:
        GaussianRBFLayer(4, cutoff, flavor="gemnet-oc", apply_cutoff_fn=True)
    except ValueError as exc:
        assert "apply_cutoff_fn" in str(exc)
    else:
        raise AssertionError("gemnet-oc must reject a second cutoff function")
    grad = dist.clone().detach().requires_grad_(True)
    layer._get_rbf(grad).sum().backward()
    assert torch.isfinite(grad.grad).all()
    assert gemnet_oc_gaussian(dist[:0], cutoff, center, coeff, 5).shape == (0, 5)


def test_cap_keeps_the_closest_incoming_edges():
    idx_i = torch.tensor([0, 0, 0, 1])
    distance = torch.tensor([2.0, 0.5, 1.0, 0.2])
    mask = cap_incoming_mask(idx_i, distance, num_nodes=2, max_neighbors=2, cutoff=3.0)
    assert mask.tolist() == [False, True, True, True]
    outside = cap_incoming_mask(
        idx_i, distance, num_nodes=2, max_neighbors=3, cutoff=0.4
    )
    assert outside.tolist() == [False, False, False, True]
    loose = cap_incoming_mask(idx_i, distance, num_nodes=2, max_neighbors=8, cutoff=5.0)
    assert loose.all()


def test_symmetrize_mirrors_the_canonical_half():
    idx_i = torch.tensor([0, 1, 1])
    idx_j = torch.tensor([1, 0, 0])
    dist = torch.tensor([1.0, 1.0, 2.0])
    vij = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    out_i, out_j, out_d, out_v, swap = symmetrize_edges(idx_i, idx_j, dist, vij)
    # Only sender < receiver is canonical: receiver 1, sender 0. Both copies
    # of that direction are kept, and each gains a reverse.
    assert out_i.tolist() == [1, 1, 0, 0]
    assert out_j.tolist() == [0, 0, 1, 1]
    assert_allclose(out_d.detach().numpy(), torch.tensor([1.0, 2.0, 1.0, 2.0]).numpy())
    assert_allclose(out_v[swap].detach().numpy(), (-out_v).detach().numpy(), atol=1e-6)
    partner = reverse_edge_index(out_i, out_j, out_v, num_nodes=2)
    assert partner.tolist() == swap.tolist()

    one_way_i, one_way_j, _, _, one_way_swap = symmetrize_edges(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1.5]),
        torch.tensor([[1.5, 0.0, 0.0]]),
    )
    assert one_way_i.numel() == 0 and one_way_swap.numel() == 0
    assert one_way_j.numel() == 0


def test_mixed_triplets_keep_a_second_image_of_the_same_atom():
    out_i = torch.tensor([0, 0])
    out_j = torch.tensor([1, 1])
    out_v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    trips = mixed_triplets(out_i, out_j, out_v, out_i, out_j, out_v, num_nodes=2)
    # Each edge pairs with the other image, not with itself.
    assert trips.idx_edge.tolist() == [0, 1]
    assert trips.idx_other.tolist() == [1, 0]
    assert trips.angle.shape == (2,)


def test_legendre_outer_is_the_product_of_zonal_harmonics():
    angle = torch.tensor([0.4, 1.2])
    dihedral = torch.tensor([0.3, 2.0])
    got = legendre_outer(angle, dihedral, 3)
    left = y_lm0(angle, 3)
    right = y_lm0(dihedral, 3)
    ref = (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(2, 9)
    assert got.shape == (2, 9)
    assert_allclose(got.detach().numpy(), ref.detach().numpy(), atol=1e-6)
    assert legendre_outer(angle[:0], dihedral[:0], 3).shape == (0, 9)


def test_periodic_images_remain_quadruplets_only_when_requested():
    # Central bond 0 → 1. Both ends see atom 2, but the displacements put
    # those images in different places.
    idx_i = torch.tensor([0, 1, 0])
    idx_j = torch.tensor([2, 2, 1])
    dist = torch.tensor([1.0, 1.0, 1.0])
    vij = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    emb = torch.tensor([0, 1])
    inter = torch.tensor([2])
    plain = quadruplet_graph(idx_i, idx_j, dist, vij, emb, inter, num_nodes=3)
    imaged = quadruplet_graph(
        idx_i, idx_j, dist, vij, emb, inter, num_nodes=3, images=True
    )
    assert plain.edge_ca.numel() == 0
    assert imaged.edge_ca.numel() == 1

    # Four distinct atoms. The image flag does not drop that quadruplet.
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.0, 0.1], [1.1, 0.9, 0.2]]
    )
    pairs_i, pairs_j = [], []
    for i in range(4):
        for j in range(4):
            if i != j:
                pairs_i.append(i)
                pairs_j.append(j)
    idx_i = torch.tensor(pairs_i)
    idx_j = torch.tensor(pairs_j)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    emb = torch.arange(idx_i.numel())
    same = quadruplet_graph(idx_i, idx_j, dist, vij, emb, emb, 4, images=False)
    other = quadruplet_graph(idx_i, idx_j, dist, vij, emb, emb, 4, images=True)
    assert same.edge_ca.numel() == other.edge_ca.numel()
    assert same.edge_ca.numel() > 0
