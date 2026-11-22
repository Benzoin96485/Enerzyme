"""SpinConv Gaussian width, neighbor cap, edge frame, and spin pool."""

from __future__ import annotations

import math

import torch
from numpy.testing import assert_allclose


def test_spinconv_gaussian_width_is_three_spacings():
    from enerzyme.models.layers.rbf import (
        GaussianRBFLayer,
        gaussian_expansion,
        spinconv_gaussian_terms,
    )

    cutoff = 6.0
    bins = 512
    layer = GaussianRBFLayer(
        num_rbf=bins,
        cutoff_sr=cutoff,
        flavor="spinconv",
        apply_cutoff_fn=False,
        basis_width_scalar=3.0,
    )
    centers, coeff = spinconv_gaussian_terms(cutoff, bins, 3.0)
    delta = (centers[1] - centers[0]).item()
    expected = torch.tensor(-0.5 / (3.0 * delta) ** 2, dtype=layer.coeff.dtype)
    assert torch.allclose(layer.coeff, expected)
    assert abs(float(layer.basis_width_scalar) - 3.0) < 1e-8
    schnet = -0.5 / delta**2
    assert abs(float(layer.coeff) - schnet) > 1.0
    distance = torch.tensor([0.0, cutoff / 2, cutoff])
    manual = gaussian_expansion(distance, centers, coeff)
    assert_allclose(layer._get_rbf(distance).detach().numpy(), manual.detach().numpy(), atol=1e-6)
    assert abs(float(layer._get_rbf(centers[:1]).squeeze()[0]) - 1.0) < 1e-5

    default = GaussianRBFLayer(
        num_rbf=8, cutoff_sr=cutoff, flavor="spinconv", apply_cutoff_fn=False
    )
    assert abs(float(default.basis_width_scalar) - 3.0) < 1e-8
    try:
        GaussianRBFLayer(
            num_rbf=8,
            cutoff_sr=cutoff,
            flavor="SchNet",
            apply_cutoff_fn=False,
            basis_width_scalar=3.0,
        )
    except ValueError as exc:
        assert "spinconv" in str(exc)
    else:
        raise AssertionError("basis_width_scalar on a SchNet basis must be rejected")
    try:
        spinconv_gaussian_terms(cutoff, 1, 3.0)
    except ValueError as exc:
        assert "bins" in str(exc)
    else:
        raise AssertionError("a one-bin SpinConv expansion must be rejected")


def test_symmetric_cap_keeps_a_pair_from_either_direction():
    from enerzyme.models.spinconv.geometry import symmetric_neighbor_mask

    # Destinations 0, 1, 2. Closest incoming edge is listed first in the story
    # below; the stored order is scrambled.
    # Node 0: from 1 at 1.0, from 2 at 3.0
    # Node 1: from 0 at 1.0, from 2 at 1.2
    # Node 2: from 1 at 1.2, from 0 at 3.0
    idx_i = torch.tensor([0, 0, 1, 1, 2, 2])
    idx_j = torch.tensor([2, 1, 2, 0, 0, 1])
    distance = torch.tensor([3.0, 1.0, 1.2, 1.0, 3.0, 1.2])
    keep = symmetric_neighbor_mask(idx_i, idx_j, distance, 3, max_neighbors=1)
    # k = 1 selects 1->0, 0->1, and 1->2. Pairs {0,1} and {1,2} stay.
    # 2->0 is kept because 1->2 selected the pair {1,2}? No: 2->0 is pair {0,2}.
    # Pair {0,2} is never selected. Pair {1,2} is selected by 1->2, so 2->1 stays.
    kept = {
        (int(i), int(j))
        for i, j, flag in zip(idx_i.tolist(), idx_j.tolist(), keep.tolist())
        if flag
    }
    assert kept == {(0, 1), (1, 0), (1, 2), (2, 1)}
    wide = symmetric_neighbor_mask(idx_i, idx_j, distance, 3, max_neighbors=2)
    assert bool(wide.all())
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    keep_perm = symmetric_neighbor_mask(idx_i[perm], idx_j[perm], distance[perm], 3, 1)
    kept_perm = {
        (int(i), int(j))
        for i, j, flag in zip(idx_i[perm].tolist(), idx_j[perm].tolist(), keep_perm.tolist())
        if flag
    }
    assert kept_perm == kept


def test_frame_is_spin_stable_and_rotates_with_the_molecule():
    from enerzyme.models.spinconv.geometry import (
        bilinear_sphere_index,
        edge_spin_frames,
        project_neighbors,
        source_neighbor_pairs,
        splat_sphere,
    )

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.3, 0.15], [0.1, 0.2, 1.2]]
    )
    idx_i, idx_j = [], []
    for i in range(4):
        for j in range(4):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)
    vij = pos[idx_j] - pos[idx_i]
    distance = vij.norm(dim=-1)
    frames = edge_spin_frames(vij, idx_i, distance, 5.0, 4)
    # Local +x is vij, destination -> source.
    assert_allclose(
        frames[:, 0].detach().numpy(),
        torch.nn.functional.normalize(vij, dim=-1).detach().numpy(),
        atol=1e-5,
    )
    source_edge, query_edge = source_neighbor_pairs(idx_i, idx_j, 4)
    local = project_neighbors(vij, frames, source_edge, query_edge)

    axis = torch.tensor([0.3, -0.5, 0.8])
    axis = axis / axis.norm()
    angle = 0.9
    skew = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = torch.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    rotated = pos @ rotation
    vij_r = rotated[idx_j] - rotated[idx_i]
    frames_r = edge_spin_frames(vij_r, idx_i, vij_r.norm(dim=-1), 5.0, 4)
    local_r = project_neighbors(
        vij_r, frames_r, *source_neighbor_pairs(idx_i, idx_j, 4)
    )
    assert_allclose(local.detach().numpy(), local_r.detach().numpy(), atol=1e-5)

    # Inversion flips local z and leaves local x and y unchanged.
    vij_i = -vij
    frames_i = edge_spin_frames(vij_i, idx_i, distance, 5.0, 4)
    local_i = project_neighbors(
        vij_i, frames_i, *source_neighbor_pairs(idx_i, idx_j, 4)
    )
    assert_allclose(local[:, 0].detach().numpy(), local_i[:, 0].detach().numpy(), atol=1e-5)
    assert_allclose(local[:, 1].detach().numpy(), local_i[:, 1].detach().numpy(), atol=1e-5)
    assert_allclose(local[:, 2].detach().numpy(), (-local_i[:, 2]).detach().numpy(), atol=1e-5)

    index, weight = bilinear_sphere_index(local, query_edge, 8, 12)
    assert_allclose(weight.sum(0).detach().numpy(), torch.ones(weight.shape[1]).numpy(), atol=1e-5)
    n_edges = int(idx_i.shape[0])
    point = torch.arange(source_edge.shape[0])
    grid = splat_sphere(torch.ones(source_edge.shape[0], 1), n_edges, 8, 12, index, weight, point)
    # A roll around local x preserves the mass on each latitude.
    spun = local.clone()
    cosine, sine = math.cos(0.4), math.sin(0.4)
    y = spun[:, 1] * cosine - spun[:, 2] * sine
    z = spun[:, 1] * sine + spun[:, 2] * cosine
    spun = torch.stack((spun[:, 0], y, z), dim=-1)
    index_s, weight_s = bilinear_sphere_index(spun, query_edge, 8, 12)
    grid_s = splat_sphere(
        torch.ones(source_edge.shape[0], 1),
        n_edges,
        8,
        12,
        index_s,
        weight_s,
        point,
    )
    assert_allclose(grid.sum(-1).detach().numpy(), grid_s.sum(-1).detach().numpy(), atol=1e-5)


def _assert_rotation(frames: torch.Tensor) -> None:
    gram = frames @ frames.transpose(-1, -2)
    eye = torch.eye(3).expand_as(gram)
    assert_allclose(gram.detach().numpy(), eye.detach().numpy(), atol=1e-5)
    # normal = polar × second and tangent = polar × normal, so det is -1.
    det = torch.linalg.det(frames)
    assert_allclose(det.detach().numpy(), -torch.ones_like(det).numpy(), atol=1e-5)


def test_degenerate_second_axis_stays_a_rotation():
    """A leaf and a cutoff edge have no usable second axis.

    The destination of a leaf has one incoming bond, parallel to the polar
    axis. An edge at the cutoff has cosine weight zero, so the average is
    the zero vector. Both used to yield a zero normal. The off-axis neighbor
    of the source must keep a transverse local coordinate.
    """
    from enerzyme.models.spinconv.geometry import (
        edge_spin_frames,
        project_neighbors,
        source_neighbor_pairs,
    )

    def transverse(pos: torch.Tensor, pairs: list[tuple[int, int]], query: tuple[int, int], neighbor: tuple[int, int], cutoff: float) -> torch.Tensor:
        idx_i = torch.tensor([dst for dst, _ in pairs])
        idx_j = torch.tensor([src for _, src in pairs])
        vij = pos[idx_j] - pos[idx_i]
        distance = vij.norm(dim=-1)
        frames = edge_spin_frames(vij, idx_i, distance, cutoff, pos.shape[0])
        _assert_rotation(frames)
        source_edge, query_edge = source_neighbor_pairs(idx_i, idx_j, pos.shape[0])
        local = project_neighbors(vij, frames, source_edge, query_edge)
        query_id = pairs.index(query)
        neighbor_id = pairs.index(neighbor)
        keep = (query_edge == query_id) & (source_edge == neighbor_id)
        assert int(keep.sum()) == 1
        return local[keep][0]

    angle = 104.5 * math.pi / 180.0
    water = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [0.96 * math.cos(angle), 0.96 * math.sin(angle), 0.0],
        ]
    )
    bonds = [(1, 0), (0, 1), (2, 0), (0, 2)]
    # Edge into the leaf H, neighbor is the other H on the oxygen.
    leaf = transverse(water, bonds, (1, 0), (0, 2), 5.0)
    assert float(torch.hypot(leaf[1], leaf[2])) > 0.5

    upright = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.96, 0.0, 0.0],
        ]
    )
    along_z = transverse(upright, bonds, (1, 0), (0, 2), 5.0)
    assert float(torch.hypot(along_z[1], along_z[2])) > 0.5

    cutoff = 5.0
    placed = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [cutoff, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    at_cutoff = transverse(placed, bonds, (1, 0), (0, 2), cutoff)
    assert float(torch.hypot(at_cutoff[1], at_cutoff[2])) > 0.5


def test_spin_pool_is_invariant_to_a_longitude_roll():
    from enerzyme.models.spinconv.interaction import SpinConvolution

    torch.manual_seed(0)
    block = SpinConvolution(8, 16, n_lat=4, n_long=12, group_channels=8, cutoff=6.0)
    grid = torch.randn(3, 8, 4, 12)
    rolled = torch.roll(grid, 3, dims=-1)
    reflected = torch.flip(grid, dims=(-1,))
    pooled = block.pool_grid(grid)
    assert_allclose(pooled.detach().numpy(), block.pool_grid(rolled).detach().numpy(), atol=1e-5)
    assert float((pooled - block.pool_grid(reflected)).detach().abs().max()) > 1e-3
    empty = block.pool_grid(grid[:0])
    assert empty.shape == (0, 16)


def test_pair_scale_starts_near_identity():
    from enerzyme.models.spinconv.interaction import PairDistanceScale

    torch.manual_seed(0)
    scale = PairDistanceScale(max_Za=8, scale_max=2.0)
    distance = torch.tensor([0.4, 1.2, 2.5])
    source = torch.tensor([1, 6, 8])
    target = torch.tensor([6, 1, 2])
    scaled = scale(distance, source, target)
    assert float((scaled - distance).detach().abs().max()) < 1e-3
    with torch.no_grad():
        scale.gain.weight.fill_(4.0)
    stretched = scale(distance, source, target)
    assert float((stretched - distance).detach().abs().min()) > 0.1
    try:
        scale(distance, torch.tensor([9, 0, 0]), target)
    except ValueError as exc:
        assert "max_Za" in str(exc)
    else:
        raise AssertionError("an atomic number above max_Za must be rejected")
    try:
        scale(distance, torch.tensor([1, -1, 0]), target)
    except ValueError as exc:
        assert "max_Za" in str(exc)
    else:
        raise AssertionError("a negative atomic number must be rejected")
    try:
        scale(distance, source, torch.tensor([1, 6, -3]))
    except ValueError as exc:
        assert "max_Za" in str(exc)
    else:
        raise AssertionError("a negative target atomic number must be rejected")
