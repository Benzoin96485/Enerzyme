"""GemNet geometry, radial basis, and bilinear aggregation."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose

from enerzyme.models.cutoff import polynomial_envelope
from enerzyme.models.dimenet.basis import real_sph_harm_l0
from enerzyme.models.gemnet.basis import (
    SphericalBesselRadial,
    joint_zonal,
    real_spherical_harmonics,
    y_lm0,
)
from enerzyme.models.gemnet.geometry import (
    angle_between,
    quadruplet_graph,
    reverse_edge_index,
    triplet_graph,
)
from enerzyme.models.gemnet.layers import BilinearAggregate, he_orthogonal_
from enerzyme.models.layers.rbf import BesselRBFLayer


def _radius(pos: torch.Tensor, cutoff: float):
    dist_mat = torch.cdist(pos, pos)
    eye = torch.eye(pos.shape[0], dtype=torch.bool)
    idx_i, idx_j = torch.where((dist_mat < cutoff) & ~eye)
    vij = pos[idx_j] - pos[idx_i]
    return idx_i, idx_j, dist_mat[idx_i, idx_j], vij


def test_gemnet_bessel_matches_closed_form_and_not_dimenet():
    cutoff = 5.0
    layer = BesselRBFLayer(
        3, cutoff, flavor="gemnet", trainable=False, envelope_exponent=5
    )
    dist = torch.tensor([0.0, 2.5, 4.9, 5.0])
    got = layer._get_rbf(dist)
    env = polynomial_envelope(dist, cutoff, 5).unsqueeze(-1)
    weights = layer.bessel_weights.view(1, -1)
    scaled = dist.view(-1, 1) / cutoff
    safe = dist.view(-1, 1).clamp_min(1e-12)
    quotient = torch.where(
        dist.view(-1, 1) == 0, weights / cutoff, torch.sin(weights * scaled) / safe
    )
    ref = env * math.sqrt(2.0 / cutoff) * quotient
    ref = torch.where(dist.view(-1, 1) >= cutoff, torch.zeros_like(ref), ref)
    assert_allclose(got.detach().numpy(), ref.detach().numpy(), atol=1e-6, rtol=1e-6)

    dimenet = BesselRBFLayer(
        3, cutoff, flavor="dimenet", trainable=False, envelope_exponent=5
    )
    assert not torch.allclose(got[1], dimenet._get_rbf(dist[1:2])[0])
    longer = BesselRBFLayer(
        3, 10.0, flavor="gemnet", trainable=False, basis_cutoff=5.0
    )
    assert longer.basis_cutoff == 5.0
    assert_allclose(
        longer._get_rbf(dist).detach().numpy(), got.detach().numpy(), atol=1e-6
    )
    grad_dist = dist.clone().detach().requires_grad_(True)
    layer._get_rbf(grad_dist).sum().backward()
    assert torch.isfinite(grad_dist.grad).all()


def test_reverse_edge_matches_images_and_rejects_one_way_edges():
    idx_i = torch.tensor([0, 0, 1, 1])
    idx_j = torch.tensor([1, 1, 0, 0])
    vij = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    swap = reverse_edge_index(idx_i, idx_j, vij, num_nodes=2)
    assert swap.tolist() == [2, 3, 0, 1]
    try:
        reverse_edge_index(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            num_nodes=3,
        )
    except ValueError as exc:
        assert "both directions" in str(exc)
    else:
        raise AssertionError("one-way edges should fail")

    # Same atoms, but the only return edge is a different image.
    try:
        reverse_edge_index(
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            num_nodes=2,
        )
    except ValueError as exc:
        assert "opposite displacement" in str(exc)
    else:
        raise AssertionError("a non-opposite image should fail")

    # The index is a neighbor-list identity: moving v continuously does not
    # move the partner, and the partner stays out of the autograd graph.
    moving = vij.clone().requires_grad_(True)
    swap_moving = reverse_edge_index(idx_i, idx_j, moving, num_nodes=2)
    assert swap_moving.tolist() == swap.tolist()
    assert swap_moving.dtype == torch.long
    assert not swap_moving.requires_grad


def test_triplet_angle_is_at_the_receiver():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    idx_i, idx_j, _, vij = _radius(pos, cutoff=2.0)
    graph = triplet_graph(idx_i, idx_j, vij, num_nodes=3)
    at_origin = idx_i[graph.idx_edge] == 0
    assert at_origin.any()
    assert_allclose(
        graph.angle[at_origin].detach().numpy(),
        torch.full((int(at_origin.sum()),), math.pi / 2).numpy(),
        atol=1e-5,
    )
    # A dimer has no triplet.
    dimer_i = torch.tensor([0, 1])
    dimer_j = torch.tensor([1, 0])
    dimer_v = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    dimer = triplet_graph(dimer_i, dimer_j, dimer_v, num_nodes=2)
    assert dimer.idx_edge.numel() == 0


def test_planar_quadruplet_dihedral_is_zero_and_trans_is_pi():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    idx_i, idx_j, dist, vij = _radius(pos, cutoff=1.2)
    emb = torch.where(dist < 1.2)[0]
    graph = quadruplet_graph(idx_i, idx_j, dist, vij, emb, emb, num_nodes=4)
    select = (
        (graph.atom_c == 2)
        & (graph.atom_a == 0)
        & (graph.atom_b == 1)
        & (graph.atom_d == 3)
    )
    assert int(select.sum()) == 1
    assert_allclose(graph.angle_cab[select].item(), math.pi / 2, atol=1e-5)
    assert_allclose(graph.angle_abd[select].item(), math.pi / 2, atol=1e-5)
    assert_allclose(graph.dihedral[select].item(), 0.0, atol=1e-5)

    pos[3, 1] = -1.0
    idx_i, idx_j, dist, vij = _radius(pos, cutoff=1.2)
    emb = torch.where(dist < 1.2)[0]
    graph = quadruplet_graph(idx_i, idx_j, dist, vij, emb, emb, num_nodes=4)
    select = (
        (graph.atom_c == 2)
        & (graph.atom_a == 0)
        & (graph.atom_b == 1)
        & (graph.atom_d == 3)
    )
    assert int(select.sum()) == 1
    assert_allclose(graph.dihedral[select].item(), math.pi, atol=1e-5)
    # Four distinct atoms, and the central bond is not reused as a message edge.
    packed = torch.stack(
        [graph.atom_c, graph.atom_a, graph.atom_b, graph.atom_d], dim=-1
    )
    assert torch.all(torch.tensor([len(set(row.tolist())) == 4 for row in packed]))


def test_interaction_cutoff_keeps_a_longer_central_bond():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [2.0, 0.0, 0.3], [3.0, 0.1, 0.0]]
    )
    idx_i, idx_j, dist, vij = _radius(pos, cutoff=3.2)
    emb = torch.where(dist < 1.5)[0]
    inter = torch.where(dist < 3.2)[0]
    graph = quadruplet_graph(idx_i, idx_j, dist, vij, emb, inter, num_nodes=4)
    assert graph.edge_ca.numel() > 0
    central = dist[inter[graph.int_local]]
    assert torch.any(central > 1.5)
    short = quadruplet_graph(idx_i, idx_j, dist, vij, emb, emb, num_nodes=4)
    if short.edge_ca.numel():
        assert torch.all(dist[emb[short.int_local]] <= 1.5)


def test_m0_harmonics_match_dimenet_and_dipole():
    theta = torch.tensor([0.2, 0.7, 1.3, 2.4])
    phi = torch.tensor([0.1, -0.4, 1.1, 2.5])
    y = real_spherical_harmonics(theta, phi, num_spherical=4)
    zonal = y_lm0(theta, 4)
    for degree in range(4):
        assert_allclose(
            y[:, degree * degree + degree].detach().numpy(),
            zonal[:, degree].detach().numpy(),
            atol=1e-6,
        )
        assert_allclose(
            y[:, degree * degree + degree].detach().numpy(),
            real_sph_harm_l0(degree, theta).detach().numpy(),
            atol=1e-6,
        )
    y1 = real_spherical_harmonics(theta, phi, num_spherical=2)
    expected_plus = math.sqrt(3.0 / (4.0 * math.pi)) * torch.sin(theta) * torch.cos(phi)
    expected_minus = math.sqrt(3.0 / (4.0 * math.pi)) * torch.sin(theta) * torch.sin(phi)
    assert_allclose(y1[:, 3].detach().numpy(), expected_plus.numpy(), atol=1e-6)
    assert_allclose(y1[:, 1].detach().numpy(), expected_minus.numpy(), atol=1e-6)
    # Inversion of both vectors leaves the unsigned angle unchanged.
    left = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.2]])
    right = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.1, 0.0]])
    assert_allclose(
        angle_between(left, right).numpy(),
        angle_between(-left, -right).numpy(),
        atol=1e-6,
    )


def test_bilinear_aggregate_matches_explicit_sum():
    torch.manual_seed(0)
    num_edges, n_sph, emb, interm, out = 4, 3, 2, 2, 2
    msg = torch.randn(num_edges, emb)
    radial = torch.randn(num_edges, n_sph, 3)
    down = torch.empty(n_sph, 3, interm)
    he_orthogonal_(down)
    radial_w = torch.einsum("eln,lni->eil", radial, down)
    angular = torch.randn(5, n_sph)
    idx_src = torch.tensor([0, 1, 2, 3, 1])
    idx_dst = torch.tensor([1, 1, 0, 2, 0])
    weight = torch.empty(emb, interm, out)
    he_orthogonal_(weight)
    layer = BilinearAggregate(emb, interm, out)
    with torch.no_grad():
        layer.weight.copy_(weight)
    got = layer(msg, idx_src, idx_dst, angular, radial_w, num_edges)

    acc = torch.zeros(num_edges, n_sph, emb)
    for src, dst, y in zip(idx_src.tolist(), idx_dst.tolist(), angular):
        acc[dst] += y[:, None] * msg[src]
    mixed = torch.einsum("eil,els->eis", radial_w, acc)
    ref = torch.einsum("eis,sio->eo", mixed, weight)
    assert_allclose(got.detach().numpy(), ref.detach().numpy(), atol=1e-5, rtol=1e-5)

    basis = SphericalBesselRadial(3, 2, cutoff=4.0, repeat_m=True)
    values = basis(torch.tensor([0.5, 1.5]))
    assert values.shape == (2, 9, 2)
    zonal = SphericalBesselRadial(2, 2, 4.0)(torch.tensor([0.5, 1.0]))
    flat = joint_zonal(zonal, y_lm0(torch.tensor([0.3, 1.0]), 2))
    assert flat.shape == (2, 4)


def test_bilinear_empty_neighbors_are_zero():
    layer = BilinearAggregate(2, 2, 2)
    msg = torch.randn(3, 2)
    radial_w = torch.randn(3, 2, 2)
    got = layer(
        msg,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, 2),
        radial_w,
        3,
    )
    assert got.shape == (3, 2)
    assert torch.count_nonzero(got) == 0
