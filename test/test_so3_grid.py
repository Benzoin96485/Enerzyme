"""Unified flat SO3Grid / S2GridProjector checks."""

from __future__ import annotations

import torch


def test_so3_grid_roundtrip_is_near_identity():
    from enerzyme.models.so3.grid import SO3Grid

    torch.manual_seed(0)
    grid = SO3Grid(2, 2, normalization="component")
    x = torch.randn(5, 9, 4, dtype=torch.float64)
    y = grid.from_grid(grid.to_grid(x.float())).double()
    # Soft reconstruction on lat-long grids (not exact orthonormal).
    assert torch.isfinite(y).all()
    assert y.shape == x.shape
    rel = (y - x).norm() / x.norm()
    assert rel < 0.35


def test_so3_grid_implements_s2_projector_protocol():
    from enerzyme.models.so3.grid import SO3Grid
    from enerzyme.models.so3.lebedev import S2LebedevProjector
    from enerzyme.models.so3.s2_projector import S2GridProjector

    lat = SO3Grid(1, 1)
    leb = S2LebedevProjector(1)
    assert isinstance(lat, S2GridProjector)
    assert isinstance(leb, S2GridProjector)
    x = torch.randn(3, 4, 2)
    assert lat.to_grid(x).shape[0] == 3
    assert leb.to_grid(x).shape == (3, leb.grid_size, 2)


def test_build_so3_grid_table_shapes():
    from enerzyme.models.so3.grid import build_so3_grid_table

    table = build_so3_grid_table(2, normalization="integral", rescale_by_mmax=False)
    assert len(table) == 3 and len(table[1]) == 3
    g = table[2][1]
    assert g.lmax == 2 and g.mmax == 1
    x = torch.randn(2, g.n_coeff, 3)
    assert g.to_grid(x).shape == (2, g.grid_size, 3)


def test_so3_grid_aliases():
    from enerzyme.models.so3.grid import SO3Grid, SO3GridResolved, SO3_Grid

    assert SO3_Grid is SO3Grid
    assert SO3GridResolved is SO3Grid
