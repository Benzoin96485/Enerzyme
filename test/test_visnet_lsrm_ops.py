"""ViSNet-LSRM fragment geometry and one bipartite block."""

from __future__ import annotations

import torch

from enerzyme.models.visnet.long_range import (
    FragmentBipartiteBlock,
    bipartite_radius_edges,
    fragment_geometry,
    resolve_long_range_fragments,
)


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_radius_edges_keep_own_fragment_and_drop_far_ones():
    atoms = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [8.0, 0.0, 0.0]],
    )
    centers = torch.tensor([[0.1, 0.0, 0.0], [8.0, 0.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    cluster_batch = torch.zeros(2, dtype=torch.long)
    idx_atom, idx_fragment, distance, direction = bipartite_radius_edges(
        atoms, centers, batch, cluster_batch, cutoff=3.0
    )
    # Atom 2 is its own fragment and is inside the cutoff. Atom 0 does not
    # reach fragment 1. The direction is fragment minus atom.
    pairs = {(int(a), int(f)) for a, f in zip(idx_atom.tolist(), idx_fragment.tolist())}
    assert (0, 0) in pairs
    assert (1, 0) in pairs
    assert (2, 1) in pairs
    assert (0, 1) not in pairs
    assert (2, 0) not in pairs
    own = (idx_atom == 0) & (idx_fragment == 0)
    assert torch.allclose(direction[own], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)
    assert torch.allclose(distance[own], torch.tensor([0.1]), atol=1e-5)


def test_radius_edges_do_not_cross_graphs():
    atoms = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    batch = torch.tensor([0, 1])
    cluster_batch = torch.tensor([0, 1])
    idx_atom, idx_fragment, _, _ = bipartite_radius_edges(
        atoms, centers, batch, cluster_batch, cutoff=9.0
    )
    assert idx_atom.tolist() == [0, 1]
    assert idx_fragment.tolist() == [0, 1]


def test_zero_separation_direction_is_zero():
    atoms = torch.zeros(1, 3)
    centers = torch.zeros(1, 3)
    batch = torch.zeros(1, dtype=torch.long)
    cluster_batch = torch.zeros(1, dtype=torch.long)
    _, _, distance, direction = bipartite_radius_edges(
        atoms, centers, batch, cluster_batch, cutoff=9.0
    )
    assert distance.item() == 0.0
    assert torch.count_nonzero(direction) == 0


def test_atomic_number_center_matches_z_weighted_mean():
    ra = torch.tensor(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
    )
    batch = torch.zeros(3, dtype=torch.long)
    ids = torch.tensor([0, 0, 1])
    za = torch.tensor([1.0, 3.0, 8.0])
    flat, centers, _ = resolve_long_range_fragments(
        ra,
        batch,
        fragment_mode="precomputed",
        center="atomic_number",
        Za=za,
        cluster_ids=ids,
        min_nodes_per_group=2,
        random_state=0,
    )
    expected = torch.tensor(
        [
            [(1 * 0 + 3 * 3) / 4, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
    )
    assert flat.tolist() == [0, 0, 1]
    assert torch.allclose(centers, expected, atol=1e-6)


def test_geometric_center_ignores_atomic_number():
    ra = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    _, centers, _ = resolve_long_range_fragments(
        ra,
        batch,
        fragment_mode="precomputed",
        center="geometric",
        Za=torch.tensor([1.0, 100.0]),
        cluster_ids=torch.zeros(2, dtype=torch.long),
        min_nodes_per_group=2,
        random_state=0,
    )
    assert torch.allclose(centers, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_block_rotation_and_empty_edges():
    torch.manual_seed(0)
    block = FragmentBipartiteBlock(4, 2, dropout=0.0)
    block.eval()
    n_atoms, n_groups, edges = 3, 2, 4
    scalar = torch.randn(n_atoms, 4)
    vector = torch.randn(n_atoms, 3, 4)
    group_scalar = torch.randn(n_groups, 4)
    group_vector = torch.randn(n_groups, 3, 4)
    idx_atom = torch.tensor([0, 0, 1, 2])
    idx_fragment = torch.tensor([0, 1, 1, 0])
    edge_attr = torch.randn(edges, 4)
    direction = torch.nn.functional.normalize(torch.randn(edges, 3), dim=-1)
    dx, dv = block(
        scalar, vector, group_scalar, group_vector, idx_atom, idx_fragment, edge_attr, direction
    )
    rotation = _rotation(1)
    dx_r, dv_r = block(
        scalar,
        torch.einsum("ncf,cd->ndf", vector, rotation),
        group_scalar,
        torch.einsum("ncf,cd->ndf", group_vector, rotation),
        idx_atom,
        idx_fragment,
        edge_attr,
        direction @ rotation,
    )
    assert torch.allclose(dx, dx_r, atol=1e-5, rtol=1e-5)
    assert torch.allclose(torch.einsum("ncf,cd->ndf", dv, rotation), dv_r, atol=1e-5, rtol=1e-5)

    empty = torch.zeros(0, dtype=torch.long)
    dx0, dv0 = block(
        scalar,
        vector,
        group_scalar,
        group_vector,
        empty,
        empty,
        scalar.new_zeros(0, 4),
        scalar.new_zeros(0, 3),
    )
    assert torch.count_nonzero(dx0) == 0
    assert torch.count_nonzero(dv0) == 0


def test_precomputed_requires_labels_and_unknown_mode_fails():
    from enerzyme.models.presets.cartesian_vector import visnet_lsrm

    try:
        visnet_lsrm(dim_embedding=4, num_rbf=2, num_heads=2, num_layers=1, fragment_mode="nope")
    except ValueError as exc:
        assert "fragment_mode" in str(exc)
    else:
        raise AssertionError("expected an unknown fragment mode to fail")

    core = visnet_lsrm(
        dim_embedding=4,
        num_rbf=2,
        num_heads=2,
        num_layers=1,
        long_layers=1,
        fragment_mode="precomputed",
        max_Za=8,
        neighbor_embedding=False,
    )
    try:
        core.get_output(
            atom_embedding=torch.randn(2, 4),
            rbf=torch.zeros(0, 2),
            idx_i_sr=torch.zeros(0, dtype=torch.long),
            idx_j_sr=torch.zeros(0, dtype=torch.long),
            vij_sr=torch.zeros(0, 3),
            Dij_sr=torch.zeros(0),
            Ra=torch.randn(2, 3),
            Za=torch.tensor([1, 6]),
        )
    except ValueError as exc:
        assert "cluster_ids" in str(exc)
    else:
        raise AssertionError("expected missing cluster_ids to fail")


def test_compiled_fragment_geometry_preserves_indices_and_center_force_derivatives():
    positions = torch.tensor([
        [0., 0., 0.], [0.3, 0.1, -0.2], [1., 0., 0.], [1.3, 0.2, 0.],
        [0., 0.1, 0.1], [0.4, 0.2, 0.], [1., 0.1, 0.], [1.3, 0.1, -0.1],
    ], dtype=torch.double, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2])
    numbers = torch.tensor([1, 6, 6, 8, 1, 8, 6, 7])
    labels = torch.tensor([7, 7, 3, 3, 3, 3, 8, 8])

    def geometry(x, mode):
        return fragment_geometry(x, batch, mode, "atomic_number", numbers, labels, 24, 7, 0.9)

    def evaluate(fn, mode):
        result = fn(positions, mode)
        # An anisotropic probe checks displacement directions as well as distances.
        projection = result[4] @ positions.new_tensor([0.1, -0.2, 0.3])
        loss = (result[3].square() + projection).sum()
        gradient, = torch.autograd.grad(loss, positions, create_graph=True)
        hessian, = torch.autograd.grad(gradient.square().sum(), positions)
        return (*result, gradient, hessian)

    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True, capture_scalar_outputs=True):
        compiled = torch.compile(geometry, backend="eager", fullgraph=True)
        for mode in ("precomputed", "kmeans"):
            torch.testing.assert_close(evaluate(compiled, mode), evaluate(geometry, mode))
