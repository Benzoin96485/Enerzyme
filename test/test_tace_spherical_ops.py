"""Spherical TACE op smoke tests."""

import torch
from e3nn import o3

from enerzyme.models.tace.paths import generate_paths, to_possible_tp_irreps
from enerzyme.models.tace.spherical.product import CgtpACE
from enerzyme.models.tace.tensor_product import O3ScatterTensorProduct, UUUTensorProduct


def test_generate_paths_uvu_and_uuu():
    ir_in = o3.Irreps("2x0e+2x1o")
    ir_sh = o3.Irreps.spherical_harmonics(1)
    ir_out = to_possible_tp_irreps(ir_in, ir_sh, parity=False, lmax=1)
    paths, actual = generate_paths(ir_out * 2, ir_in, ir_sh, e3nn_mode="uvu")
    assert len(paths) > 0
    assert actual.dim > 0


def test_o3_scatter_tp_shapes():
    n, e = 5, 8
    ir_in = o3.Irreps("4x0e")
    ir_sh = o3.Irreps.spherical_harmonics(1)
    ir_out = (to_possible_tp_irreps(ir_in, ir_sh, False, 1) * 4).regroup()
    tp = O3ScatterTensorProduct(ir_in, ir_sh, ir_out)
    x = torch.randn(n, ir_in.dim)
    y = torch.randn(e, ir_sh.dim)
    w = torch.randn(e, tp.weight_numel)
    edge_index = torch.stack(
        [torch.randint(0, n, (e,)), torch.randint(0, n, (e,))], dim=0
    )
    out = tp(x, y, w, edge_index)
    assert out.shape == (n, tp.irreps_out.dim)


def test_uuu_and_cgtpace_forward():
    ir = o3.Irreps("4x0e+4x1o")
    target = o3.Irreps("1x0e+1x1o")
    uuu = UUUTensorProduct(ir, ir, (target * 4).regroup(), identical_inputs=True)
    x = torch.randn(3, ir.dim)
    y = uuu(x, x)
    assert y.shape[0] == 3
    ace = CgtpACE(
        layer=0,
        num_layers=1,
        num_elements=5,
        num_channel=4,
        Lmax=1,
        lmax=1,
        irreps_in=ir,
        correlation=2,
        target_irreps=target,
        parity=False,
    )
    attrs = torch.nn.functional.one_hot(torch.tensor([1, 2, 1]), num_classes=5).float()
    out = ace(x, attrs, sc=None)
    assert out.shape == (3, ace.irreps_out.dim)
