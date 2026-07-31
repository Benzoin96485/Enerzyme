"""Regression tests for SimpleReadout / HierachicalReadout field splitting.

Bugbot flagged ``output[:, i]`` as broken for ``shallow_ensemble_size > 1``.
That slice is intentional: DenseLayer returns ``(N, fields, ensemble)``, and
field ``i`` must remain ``(N, ensemble)`` for ChargeConservation, AtomicAffine,
EnergyReduce, Force, WeightedLoss, and ShallowEnsembleReduce.
"""
from __future__ import annotations

import torch
from torch.nn import Module

from enerzyme.models.blocks.mlp import DenseLayer
from enerzyme.models.layers.denormalize import AtomicAffineLayer
from enerzyme.models.layers.electrostatics import ChargeConservationLayer
from enerzyme.models.layers.readout import HierachicalReadout, SimpleReadout
from enerzyme.models.layers.reduce import EnergyReduceLayer, ShallowEnsembleReduceLayer
from enerzyme.models.layers.spin import SpinConservationLayer
from enerzyme.models.loss import MSELoss


class _DummyPrev(Module):
    def __init__(self, dim_feature_out: int = 8):
        super().__init__()
        self.dim_feature_out = dim_feature_out


def test_dense_layer_ensemble_layout_matches_physnet_slice():
    n, d, fields, ensemble = 5, 8, 2, 4
    y = DenseLayer(d, fields, shallow_ensemble_size=ensemble)(torch.randn(n, d))
    assert y.shape == (n, fields, ensemble)
    assert y.select(1, 0).shape == (n, ensemble)
    assert y[:, 0].shape == (n, ensemble)


def test_simple_readout_size_one_returns_per_atom_scalars():
    n, d = 6, 8
    readout = SimpleReadout(
        output_fields={"Qa", "Sa"},
        built_layers=[_DummyPrev(d)],
        head_type="dense",
        shallow_ensemble_size=1,
    )
    out = readout.get_output(torch.randn(n, d))
    assert set(out) == {"Qa", "Sa"}
    assert out["Qa"].shape == (n,)
    assert out["Sa"].shape == (n,)


def test_simple_readout_ensemble_keeps_trailing_member_axis():
    n, d, ensemble = 6, 8, 4
    readout = SimpleReadout(
        output_fields={"Ea", "Qa", "Sa"},
        built_layers=[_DummyPrev(d)],
        head_type="dense",
        shallow_ensemble_size=ensemble,
    )
    out = readout.get_output(torch.randn(n, d))
    assert out["Ea"].shape == (n, ensemble)
    assert out["Qa"].shape == (n, ensemble)
    assert out["Sa"].shape == (n, ensemble)


def test_simple_readout_3d_atom_feature_uses_last_block():
    n, d, blocks, ensemble = 4, 8, 3, 5
    readout = SimpleReadout(
        output_fields={"Qa", "Sa"},
        built_layers=[_DummyPrev(d)],
        head_type="dense",
        shallow_ensemble_size=ensemble,
    )
    feat = torch.randn(n, d, blocks)
    out = readout.get_output(feat)
    assert out["Qa"].shape == (n, ensemble)
    assert out["Sa"].shape == (n, ensemble)


def test_simple_readout_residual_layer_ensemble():
    n, d, ensemble = 4, 8, 3
    readout = SimpleReadout(
        output_fields={"Qa", "Sa"},
        built_layers=[_DummyPrev(d)],
        head_type="residual_layer",
        shallow_ensemble_size=ensemble,
        activation_fn="swish",
    )
    out = readout.get_output(torch.randn(n, d))
    assert out["Qa"].shape == (n, ensemble)
    assert out["Sa"].shape == (n, ensemble)


def test_hierachical_readout_ensemble_and_one_head_per_block():
    n, d, blocks, ensemble = 5, 8, 3, 4
    readout = HierachicalReadout(
        num_blocks=blocks,
        output_fields={"Ea", "Qa"},
        built_layers=[_DummyPrev(d)],
        head_type="dense",
        shallow_ensemble_size=ensemble,
        use_nhloss=True,
    )
    assert len(readout.heads) == blocks
    out = readout.get_output(torch.randn(n, d, blocks))
    assert out["Ea"].shape == (n, ensemble)
    assert out["Qa"].shape == (n, ensemble)
    assert torch.is_tensor(out["nh_loss"]) and out["nh_loss"].ndim == 0


def test_ensemble_readout_compatible_with_conservation_affine_reduce_loss():
    n, d, ensemble = 6, 8, 4
    readout = SimpleReadout(
        output_fields={"Ea", "Qa", "Sa"},
        built_layers=[_DummyPrev(d)],
        head_type="dense",
        shallow_ensemble_size=ensemble,
    )
    pred = readout.get_output(torch.randn(n, d))
    za = torch.tensor([1, 6, 8, 1, 6, 8])
    batch_seg = torch.tensor([0, 0, 0, 1, 1, 1])
    q = torch.tensor([0.0, 1.0])
    s = torch.tensor([0.0, 1.0])

    qa = ChargeConservationLayer().get_output(
        Za=za, Qa=pred["Qa"], Q=q, batch_seg=batch_seg
    )
    sa = SpinConservationLayer().get_output(
        Za=za, Sa=pred["Sa"], S=s, batch_seg=batch_seg
    )
    assert qa["Qa"].shape == (n, ensemble)
    assert qa["Q"].shape == (2, ensemble)
    assert sa["Sa"].shape == (n, ensemble)
    assert sa["S"].shape == (2, ensemble)

    ea = AtomicAffineLayer(max_Za=86).get_output(Za=za, Ea=pred["Ea"])["Ea"]
    assert ea.shape == (n, ensemble)
    reduced = EnergyReduceLayer().get_output(Ea=ea, batch_seg=batch_seg, Za=za)
    assert reduced["E"].shape == (2, ensemble)

    stats = ShallowEnsembleReduceLayer(
        reduce_mean=["E", "Q"], var=["E"]
    ).get_output(E=reduced["E"], Q=qa["Q"])
    assert stats["E"].shape == (2,)
    assert stats["Q"].shape == (2,)
    assert stats["E_var"].shape == (2,)

    targets = {"Q": q.clone(), "S": s.clone()}
    loss = MSELoss(Q=1.0, S=1.0)({"Q": qa["Q"], "S": sa["S"]}, targets)
    assert torch.isfinite(loss)
    assert targets["Q"].shape == (2,)
    assert targets["S"].shape == (2,)
