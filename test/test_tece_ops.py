"""TECE / SO(2) op smoke tests (WignerD, SO2Gate, ComplexProduct, RRA)."""

import math

import torch

from enerzyme.models.so3 import (
    ComplexProductBasis,
    LayoutTransform,
    SO2Gate,
    WignerD,
    so2_expand_index,
    uvSO2Linear,
)
from enerzyme.models.activation import ScaledSigmoid
from enerzyme.models.tece.interaction import UvSO2TensorProduct


def test_so2_expand_index_shapes():
    n, idx = so2_expand_index(mmax=2, lmax=2)
    assert n == 6  # 3 + 2 + 1 for m=0,1,2 truncated
    assert idx.numel() == n + (n - 3)  # +-m duplicates for m>0


def test_layout_transform_roundtrip():
    from e3nn.o3 import Irreps

    irreps = Irreps("4x0e+4x1o+4x2e")
    lt = LayoutTransform(irreps)
    x = irreps.randn(3, -1)
    y = lt(x)
    assert y.shape == (3, 9, 4)
    z = lt.inverse(y)
    assert torch.allclose(x, z, atol=1e-6)


def test_wigner_d_recursive_shapes_and_orthogonality():
    torch.manual_seed(0)
    wd = WignerD(mmax=2, lmax=2, wigner_type="recursive")
    edge = torch.randn(5, 3)
    edge = edge / edge.norm(dim=-1, keepdim=True)
    w, w_inv = wd.get_wigner(edge)
    # m-primary rows × full SO3 columns
    assert w.shape[0] == 5 and w.shape[2] == 9
    assert w_inv.shape[0] == 5 and w_inv.shape[1] == 9
    # approximate left-inverse on the retained m subspace
    eyeish = torch.bmm(w, w_inv)
    assert eyeish.shape[1] == eyeish.shape[2]
    diag = torch.diagonal(eyeish, dim1=1, dim2=2)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)


def test_uv_so2_linear_and_gate():
    torch.manual_seed(0)
    lin = uvSO2Linear(mmax=1, lmax=1, num_channel_in=4, num_channel_out=4, weight_type="w1")
    # components: m0:2, m1:1*2 → 4
    x = torch.randn(3, 4, 4)
    y = lin(x)
    assert y.shape == x.shape
    gate = SO2Gate(
        mmax=1,
        lmax=1,
        num_channel=4,
        scalar_act=ScaledSigmoid(),
        tensor_act=ScaledSigmoid(),
        gate_m0=False,
    )
    # gates for m>=1 only: num_components = 1 (lmax+1-m for m=1)
    g = torch.randn(3, gate.num_components * 4)
    out = gate(y, g)
    assert out.shape == y.shape


def test_complex_product_basis_shapes():
    torch.manual_seed(0)
    ece = ComplexProductBasis(mmax=1, lmax=1, num_channel=4, m1m2=">=")
    # channel-wise SO2 layout: m0: l+1=2, m>0: 2*(l+1)=4 → total 6
    x = torch.randn(3, 6, 4)
    y = torch.randn(3, 6, 4)
    w = torch.randn(3, ece.weight_numel)
    out = ece(x, y, w)
    assert out.shape == x.shape


def test_uv_so2_tensor_product_ece_rra_forward():
    from e3nn.o3 import Irreps

    torch.manual_seed(0)
    irreps = Irreps("4x0e+4x1o")
    n_nodes, n_edges = 4, 6
    reshape_in = LayoutTransform(irreps)
    reshape_out = LayoutTransform(irreps)
    tp = UvSO2TensorProduct(
        mmax=1,
        lmax=1,
        num_channel=4,
        num_head=1,
        use_temperature=True,
        edge_ace_hidden=4,
        edge_wise_hidden=4,
        num_radial_basis=4,
        so2_linear_type="w1",
        gate_m0=False,
        use_so2_edge_ace=True,
        use_graph_softmax=True,
        reshape_in=reshape_in,
        reshape_out=reshape_out,
        scalar_act=ScaledSigmoid(),
        tensor_act=ScaledSigmoid(),
        use_radial_phase=True,
    )
    x = irreps.randn(n_nodes, -1)
    w = torch.randn(n_edges, tp.weight_numel)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]])
    cutoff = torch.ones(n_edges, 1)
    wd = WignerD(mmax=1, lmax=1)
    edge_vec = torch.randn(n_edges, 3)
    edge_vec = edge_vec / edge_vec.norm(dim=-1, keepdim=True)
    wigner, wigner_inv = wd.get_wigner(edge_vec)
    rbf = torch.rand(n_edges, 4)
    out = tp(x, w, edge_index, cutoff, wigner, wigner_inv, rbf)
    assert out.shape == x.shape
    out.sum().backward()
    assert math.isfinite(out.detach().sum().item())
