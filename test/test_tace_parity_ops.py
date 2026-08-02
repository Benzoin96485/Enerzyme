"""Numerical parity: TACE cartnn vs vendored tace v0.1.0; spherical vs e3nn TP."""

from __future__ import annotations

import sys

import torch
from e3nn import o3
from torch_scatter import scatter_sum

sys.path.extend(["..", "."])

from tace_parity_utils import PARITY_HPARAMS, assert_close  # noqa: E402


def test_cartnn_cartesian_harmonics_match_fixture():
    from enerzyme.models.tace.cartnn import CartesianHarmonics as EZ
    from cartnn import CartesianHarmonics as Off

    torch.manual_seed(0)
    lmax = PARITY_HPARAMS["lmax"]
    n = PARITY_HPARAMS["num_nodes"]
    ez = EZ(lmax, normalize=True)
    off = Off(lmax, normalize=True)
    vec = torch.randn(n, 3)
    assert_close(ez(vec), off(vec), atol=1e-6, rtol=1e-6)


def _assert_nested_close(a, b, atol=1e-12, rtol=1e-12):
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert_close(
            torch.as_tensor(a, dtype=torch.float64),
            torch.as_tensor(b, dtype=torch.float64),
            atol=atol,
            rtol=rtol,
        )
        return
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_nested_close(x, y, atol=atol, rtol=rtol)
        return
    assert a == b


def test_cartnn_ictd_match_fixture():
    from enerzyme.models.tace.cartnn import ICTD as EZ_ICTD
    from cartnn import ICTD as Off_ICTD

    for r in (0, 1, 2):
        _assert_nested_close(EZ_ICTD(r, r), Off_ICTD(r, r))


def test_o3_scatter_tp_matches_internal_e3nn_tp_scatter():
    """Spherical TACE path: O3ScatterTensorProduct == e3nn TP(x[src], y) + scatter.

    ``O3ScatterTensorProduct`` wraps ``o3.TensorProduct``; this locks the scatter
    wiring against a manual edge expansion (upstream ACE-style message).
    """
    from enerzyme.models.e3nn_nn import (
        O3ScatterTensorProduct,
        generate_paths,
        to_possible_tp_irreps,
    )

    torch.manual_seed(2)
    n, e = PARITY_HPARAMS["num_nodes"], PARITY_HPARAMS["num_edges"]
    ir_in = o3.Irreps("4x0e+4x1o")
    ir_sh = o3.Irreps.spherical_harmonics(1)
    ir_out = (to_possible_tp_irreps(ir_in, ir_sh, parity=False, lmax=1) * 4).regroup()
    tp = O3ScatterTensorProduct(ir_in, ir_sh, ir_out)

    x = torch.randn(n, ir_in.dim)
    y = torch.randn(e, ir_sh.dim)
    w = torch.randn(e, tp.weight_numel)
    src = torch.randint(0, n, (e,))
    dst = torch.randint(0, n, (e,))
    edge_index = torch.stack([src, dst], dim=0)
    out_ez = tp(x, y, w, edge_index)

    out_edge = tp.tp(x[src], y, w)
    out_ref = scatter_sum(out_edge, dst, dim=0, dim_size=n)
    assert_close(out_ez, out_ref, atol=1e-6, rtol=1e-6)

    paths, actual = generate_paths(ir_out, ir_in, ir_sh, e3nn_mode="uvu")
    assert len(paths) > 0
    assert actual.dim > 0
