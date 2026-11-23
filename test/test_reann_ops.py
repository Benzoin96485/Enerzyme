"""REANN densities, activations, and the derivative polarizability."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from numpy.testing import assert_allclose

from enerzyme.models.activation import ReluLike, Swish, TanhLike
from enerzyme.models.layers.response import (
    DerivativePolarizabilityLayer,
    assemble_polarizability,
)
from enerzyme.models.reann.density import (
    angular_channel_count,
    angular_momentum_index,
    cartesian_angular,
    default_n_orbit,
    embedded_atom_density,
    squared_cosine_window,
)


def _rotation(seed: int = 0, improper: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    if improper:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _density(pos, species, edges, coeff, rs, inta, hyper, cutoff, lmax):
    idx_i = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
    idx_j = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
    vij = pos[idx_j] - pos[idx_i]
    distance = torch.linalg.vector_norm(vij, dim=-1)
    return embedded_atom_density(
        coeff,
        idx_i,
        idx_j,
        vij,
        distance,
        species,
        rs,
        inta,
        hyper,
        angular_momentum_index(lmax),
        cutoff,
    )


def test_activation_formulas_match_the_paper_and_the_repository():
    features = torch.tensor([[0.4, -0.7, 1.2]], dtype=torch.float64)
    relu = ReluLike(dim_feature=3, initial_alpha=0.5, initial_beta=1.7, learnable=True).double()
    tanh = TanhLike(dim_feature=3, initial_alpha=0.3, initial_beta=0.8, learnable=True).double()
    swish = Swish(dim_feature=3, initial_alpha=1.2, initial_beta=0.4, learnable=True).double()
    assert_allclose(
        relu(features).detach().numpy(),
        (0.5 * F.silu(1.7 * features)).numpy(),
    )
    expected_tanh = 0.3 * features / torch.sqrt(1.0 + (0.8 * features).square())
    assert_allclose(tanh(features).detach().numpy(), expected_tanh.numpy())
    expected_swish = 1.2 * features * torch.sigmoid(0.4 * features)
    assert_allclose(swish(features).detach().numpy(), expected_swish.numpy())


def test_orbit_width_and_cartesian_channel_order():
    assert default_n_orbit(8, 2) == 108
    assert angular_channel_count(0) == 1
    assert angular_channel_count(1) == 4
    assert angular_channel_count(2) == 13
    assert angular_momentum_index(2).tolist() == [0, 1, 1, 1] + [2] * 9

    vectors = torch.tensor([[0.2, -0.4, 0.5], [0.3, 0.1, -0.2]], dtype=torch.float64)
    window = torch.tensor([0.7, 0.4], dtype=torch.float64)
    angular = cartesian_angular(vectors, window, 2)
    # Released product: einsum("ji,ki->jki") on [C, E] and [3, E].
    released = [window.view(1, -1)]
    direction = vectors.T.contiguous()
    for _ in range(2):
        released.append(
            torch.einsum("ji,ki->jki", released[-1], direction).reshape(-1, vectors.shape[0])
        )
    assert_allclose(angular.numpy(), torch.cat(released, dim=0).numpy())


def test_density_is_a_coherent_square_of_neighbor_orbitals():
    cutoff = 3.0
    lmax = 0
    rs = torch.zeros(9, 1, dtype=torch.float64)
    rs[8, 0] = 1.0
    inta = -torch.ones(9, 1, dtype=torch.float64)
    hyper = torch.ones(1, 1, 1, dtype=torch.float64)
    coeff = torch.ones(3, 1, dtype=torch.float64)
    species = torch.tensor([1, 8, 8])
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    window = float(squared_cosine_window(torch.tensor(1.0), cutoff))
    one = _density(pos[:2], species[:2], [(0, 1)], coeff[:2], rs, inta, hyper, cutoff, lmax)
    assert_allclose(one[0].numpy(), [window**2], atol=1e-12)
    assert_allclose(one[1].numpy(), [0.0], atol=1e-12)
    # A central-atom Gaussian at rs=0 would give exp(-1), not 1.
    wrong = (window * torch.exp(torch.tensor(-1.0))) ** 2
    assert abs(float(one[0]) - float(wrong)) > 1e-3

    two = _density(pos, species, [(0, 1), (0, 2)], coeff, rs, inta, hyper, cutoff, lmax)
    assert_allclose(two[0].numpy(), [4.0 * window**2], atol=1e-12)
    flipped = _density(pos, species, [(0, 2), (0, 1)], coeff, rs, inta, hyper, cutoff, lmax)
    assert_allclose(two.numpy(), flipped.numpy(), atol=1e-12)

    lmax = 1
    hyper = torch.ones(2, 1, 1, dtype=torch.float64)
    rs = torch.zeros(9, 1, dtype=torch.float64)
    rs[8, 0] = 0.5
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.3, 0.0]], dtype=torch.float64)
    distance = torch.linalg.vector_norm(pos[1] - pos[0])
    window = float(squared_cosine_window(distance, cutoff))
    angular = _density(pos, species[:2], [(0, 1)], coeff[:2], rs, inta, hyper, cutoff, lmax)
    assert_allclose(angular[0].numpy(), [window**2 * (1.0 + float(distance) ** 2)], atol=1e-12)


def test_density_is_invariant_and_vanishes_at_the_cutoff():
    torch.manual_seed(2)
    cutoff = 4.0
    lmax = 2
    nwave = 3
    n_orbit = 4
    n_atoms = 4
    pos = torch.randn(n_atoms, 3, dtype=torch.float64)
    species = torch.tensor([1, 6, 8, 1])
    edges = [(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]
    coeff = torch.randn(n_atoms, nwave, dtype=torch.float64)
    rs = torch.rand(9, nwave, dtype=torch.float64) * cutoff
    inta = -(torch.rand(9, nwave, dtype=torch.float64) + 0.2)
    hyper = torch.randn(lmax + 1, nwave, n_orbit, dtype=torch.float64)
    base = _density(pos, species, edges, coeff, rs, inta, hyper, cutoff, lmax)
    for improper in (False, True):
        rotated = pos @ _rotation(seed=4 + int(improper), improper=improper)
        other = _density(rotated, species, edges, coeff, rs, inta, hyper, cutoff, lmax)
        assert_allclose(base.numpy(), other.numpy(), atol=1e-10)
    shifted = _density(pos + torch.tensor([1.2, -0.4, 0.7]), species, edges, coeff, rs, inta, hyper, cutoff, lmax)
    assert_allclose(base.numpy(), shifted.numpy(), atol=1e-10)

    pos = torch.tensor([[0.0, 0.0, 0.0], [cutoff, 0.0, 0.0]], dtype=torch.float64)
    coeff = torch.ones(2, 1, dtype=torch.float64)
    rs = torch.zeros(9, 1, dtype=torch.float64)
    inta = -torch.ones(9, 1, dtype=torch.float64)
    hyper = torch.ones(1, 1, 1, dtype=torch.float64)
    at_cutoff = _density(pos, torch.tensor([1, 8]), [(0, 1)], coeff, rs, inta, hyper, cutoff, 0)
    assert_allclose(at_cutoff.numpy(), torch.zeros_like(at_cutoff).numpy(), atol=1e-12)


def test_polarizability_assembly_and_latent_derivative():
    positions = torch.tensor([[1.0, 0.0, 0.0]])
    grad = torch.tensor([[2.0, 0.0, 0.0]])
    isotropic = torch.tensor([3.0])
    batch = torch.zeros(1, dtype=torch.long)
    paper = assemble_polarizability(positions, grad, grad, isotropic, batch, "paper")
    assert_allclose(
        paper.numpy(),
        [[[11.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
    )
    grad_b = torch.tensor([[0.0, 1.0, 0.0]])
    repo = assemble_polarizability(positions, grad, grad_b, isotropic, batch, "repo")
    assert_allclose(
        repo.numpy(),
        [[[3.0, 3.0, 0.0], [3.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
    )

    layer = DerivativePolarizabilityLayer(dim_embedding=3, variant="paper")
    with torch.no_grad():
        layer.head[0].weight.zero_()
        layer.head[0].weight[0, 0] = 1.0
    ra = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    alpha = layer.get_output(ra, ra, batch)["polarizability"]
    assert_allclose(alpha.detach().numpy(), [[[3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    repo_layer = DerivativePolarizabilityLayer(dim_embedding=3, variant="repo")
    with torch.no_grad():
        repo_layer.head[0].weight.zero_()
        repo_layer.head[0].weight[0, 0] = 1.0
        repo_layer.head[0].weight[1, 1] = 1.0
    repo_layer.train()
    ra = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    alpha = repo_layer.get_output(ra, ra, batch)["polarizability"]
    assert_allclose(
        alpha.detach().numpy(),
        [[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
    )
    alpha.sum().backward()
    assert repo_layer.head[0].weight.grad is not None
    assert float(repo_layer.head[0].weight.grad.abs().sum()) > 0.0

    try:
        layer.get_output(ra.detach(), ra.detach(), batch)
    except ValueError as exc:
        assert "requires_grad" in str(exc)
    else:
        raise AssertionError("polarizability without coordinate gradients must be rejected")
