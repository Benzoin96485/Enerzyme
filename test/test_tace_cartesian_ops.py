"""Cartesian TACE / cartnn op smoke tests."""

import subprocess
import sys

import torch

from enerzyme.models.tace.cartesian.core_blocks import (
    CartesianContraction,
    InterEinsum,
    _generate_combs,
    _split_cartesian_harmonics,
)
from enerzyme.models.tace.cartnn import ICTD, CartesianHarmonics


def test_cartnn_eager_import_skips_product_basis():
    """ICTD/CartesianHarmonics must not pull SymmetricContraction (opt_einsum_fx)."""
    code = (
        "import sys\n"
        "from enerzyme.models.tace.cartnn import ICTD, CartesianHarmonics\n"
        "assert 'enerzyme.models.tace.cartnn._product_basis' not in sys.modules\n"
        "assert ICTD is not None and CartesianHarmonics is not None\n"
    )
    subprocess.check_call([sys.executable, "-c", code], cwd=".")


def test_ictd_and_cartesian_harmonics():
    eh = CartesianHarmonics(
        list(range(3)),
        normalize=True,
        normalization="component",
        norm=True,
        traceless=True,
    )
    v = torch.randn(7, 3)
    flat = eh(v)
    assert flat.shape[0] == 7
    assert flat.shape[1] == sum(3**l for l in range(3))
    parts = _split_cartesian_harmonics(flat, 2)
    assert 0 in parts and 1 in parts and 2 in parts
    ds = ICTD(1, 1)[1]
    assert ds[0].shape[-1] == 3


def test_inter_einsum_and_contraction():
    comb = (0, 1, 1, 0)
    assert comb in _generate_combs(0, 1)
    tc = InterEinsum(comb)
    t1 = torch.randn(5, 4)  # [E, C]
    t2 = torch.randn(5, 4, 3)  # [E, C, 3]
    out = tc(t1, t2)
    assert out.shape == (5, 4, 3)

    ctr = CartesianContraction(num_channel=4, lmax_in=0, lmax_out=1)
    node = {0: torch.randn(6, 4)}
    edge = {0: torch.ones(8), 1: torch.randn(8, 3)}
    w = torch.randn(8, ctr.weight_numel)
    edge_index = torch.stack(
        [torch.randint(0, 6, (8,)), torch.randint(0, 6, (8,))], dim=0
    )
    msg = ctr(node, edge, w, edge_index)
    assert 0 in msg or 1 in msg


def test_self_interaction_zero_pads_missing_ranks():
    """Layer-0 BB residual must emit all ls_out even when feats only have l=0."""
    from enerzyme.models.tace.cartesian.core_blocks import DictSkipIdentity, SelfInteraction

    feats = {0: torch.randn(4, 8)}
    attrs = torch.nn.functional.one_hot(torch.tensor([1, 2, 0, 3]), num_classes=5).float()
    for element_aware in (False, True):
        si = SelfInteraction(
            8, 8, ls=[0, 1], bias=True, element_aware=element_aware, num_elements=5
        )
        out = si(feats, attrs if element_aware else None)
        assert set(out) == {0, 1}
        assert out[1].shape == (4, 8, 3)
        assert torch.allclose(out[1], torch.zeros_like(out[1]))
        assert torch.isfinite(out[0]).all()

    skip = DictSkipIdentity([0, 1], 8)
    out_id = skip(feats)
    assert set(out_id) == {0, 1}
    assert torch.allclose(out_id[1], torch.zeros_like(out_id[1]))
