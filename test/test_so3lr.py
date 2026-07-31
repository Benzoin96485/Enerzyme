"""Tests for SO3LR physics priors and so3lr architecture wiring."""

from __future__ import annotations

import math
import sys

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])

_BOHR = 0.5291772105638411
_HARTREE_EV = 27.211386245988


def test_zbl_kehalf_from_units():
    from enerzyme.models.layers import ZBLRepulsionEnergyLayer

    zbl_ha = ZBLRepulsionEnergyLayer(Bohr_in_R=_BOHR, Hartree_in_E=1.0)
    zbl_ev = ZBLRepulsionEnergyLayer(Bohr_in_R=_BOHR, Hartree_in_E=_HARTREE_EV)
    assert_allclose(zbl_ha.kehalf, 0.5 * _BOHR, rtol=1e-12)
    assert_allclose(zbl_ev.kehalf, 0.5 * _BOHR * _HARTREE_EV, rtol=1e-12)


def test_zbl_positive_decays_and_switch_off():
    from enerzyme.models.layers import ZBLRepulsionEnergyLayer

    zbl = ZBLRepulsionEnergyLayer(
        Bohr_in_R=_BOHR, Hartree_in_E=_HARTREE_EV, switch_off=1.5
    )
    za = torch.tensor([1, 1], dtype=torch.long)
    energies = []
    for r in (0.5, 1.0, 2.0):
        dij = torch.tensor([r, r])
        idx_i = torch.tensor([0, 1], dtype=torch.long)
        idx_j = torch.tensor([1, 0], dtype=torch.long)
        e = zbl.get_E_zbl_a(za, dij, idx_i, idx_j, torch.ones_like(dij))
        energies.append(e.sum().item())
        assert torch.all(e >= 0)
    assert energies[0] > energies[1] > energies[2]
    e_far = zbl.get_E_zbl_a(
        za,
        torch.tensor([5.0, 5.0]),
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([1, 0], dtype=torch.long),
        torch.ones(2),
    )
    assert torch.all(e_far < 1e-2)


def test_erf_coulomb_h2_analytic():
    from enerzyme.models.layers import ElectrostaticEnergyLayer

    layer = ElectrostaticEnergyLayer(
        flavor="SO3LR",
        Bohr_in_R=_BOHR,
        Hartree_in_E=_HARTREE_EV,
        electrostatic_energy_scale=4.0,
        cutoff_lr=None,
        neighborlist_format_lr="sparse",
    )
    qa = torch.tensor([1.0, 1.0])
    dij = torch.tensor([1.0, 1.0])
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    e = layer.get_E_ele_a(dij, qa, idx_i, idx_j)
    expected_edge = layer.kehalf * math.erf(1.0 / 4.0)
    assert_allclose(e[0].item(), expected_edge, rtol=1e-6)
    assert_allclose(e[1].item(), expected_edge, rtol=1e-6)


def test_erf_coulomb_cutoff_zero_beyond():
    from enerzyme.models.layers import ElectrostaticEnergyLayer

    layer = ElectrostaticEnergyLayer(
        flavor="SO3LR",
        Bohr_in_R=_BOHR,
        Hartree_in_E=_HARTREE_EV,
        electrostatic_energy_scale=4.0,
        cutoff_lr=3.0,
        neighborlist_format_lr="sparse",
    )
    qa = torch.tensor([1.0, -1.0])
    dij = torch.tensor([5.0, 5.0])
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    e = layer.get_E_ele_a(dij, qa, idx_i, idx_j)
    assert torch.allclose(e, torch.zeros_like(e))


def test_tsqdo_dispersion_negative_attractive():
    from enerzyme.models.layers import TSQDODispersionEnergyLayer

    layer = TSQDODispersionEnergyLayer(
        dispersion_energy_scale=1.2,
        cutoff_lr=None,
        neighborlist_format_lr="sparse",
        Hartree_in_E=_HARTREE_EV,
        Bohr_in_R=_BOHR,
    )
    za = torch.tensor([6, 6], dtype=torch.long)
    ha = torch.tensor([1.0, 1.0])
    dij = torch.tensor([3.0, 3.0])
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    e = layer.get_E_disp_a(za, ha, dij, idx_i, idx_j)
    assert torch.all(e < 0)


def test_tsqdo_scales_with_hartree_in_e():
    """Dispersion must follow Hartree_in_E (not a hard-coded eV factor)."""
    from enerzyme.models.layers import TSQDODispersionEnergyLayer

    kwargs = dict(
        dispersion_energy_scale=1.2,
        cutoff_lr=None,
        neighborlist_format_lr="sparse",
        Bohr_in_R=_BOHR,
    )
    e_ha = TSQDODispersionEnergyLayer(Hartree_in_E=1.0, **kwargs)
    e_ev = TSQDODispersionEnergyLayer(Hartree_in_E=_HARTREE_EV, **kwargs)
    za = torch.tensor([6, 6], dtype=torch.long)
    ha = torch.tensor([1.0, 1.0])
    dij = torch.tensor([3.0, 3.0])
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    out_ha = e_ha.get_E_disp_a(za, ha, dij, idx_i, idx_j)
    out_ev = e_ev.get_E_disp_a(za, ha, dij, idx_i, idx_j)
    assert_allclose(out_ev.numpy(), (out_ha * _HARTREE_EV).numpy(), rtol=1e-6)


def test_hirshfeld_and_partial_charge_shapes():
    from enerzyme.models.layers import HirshfeldReadout, PartialChargeReadout

    F = 8
    N = 4
    za = torch.randint(1, 10, (N,))
    feat = torch.randn(N, F)
    qa = PartialChargeReadout(dim_embedding=F).get_Qa(feat, za)
    ha = HirshfeldReadout(dim_embedding=F).get_ha(feat, za)
    assert qa.shape == (N,)
    assert ha.shape == (N,)
    assert torch.all(ha >= 0)


def test_charge_spin_embedding_shapes():
    from enerzyme.models.layers import ChargeSpinEmbeddingLayer

    N = 5
    F = 16
    za = torch.randint(1, 8, (N,))
    batch = torch.tensor([0, 0, 0, 1, 1])
    q = torch.tensor([0.0, 1.0])
    emb = ChargeSpinEmbeddingLayer(dim_embedding=F, max_Za=118, attribute="charge")
    out = emb.get_output(Za=za, batch_seg=batch, Q=q)
    assert out["charge_embedding"].shape == (N, F)
    assert emb.Wq.in_features == 118


def test_charge_spin_one_hot_matches_z_table_convention():
    from enerzyme.models.layers import ChargeSpinEmbeddingLayer

    layer = ChargeSpinEmbeddingLayer(dim_embedding=4, max_Za=10, attribute="charge")
    with torch.no_grad():
        layer.Wq.weight.zero_()
        layer.Wq.weight[:, 0] = 1.0
        layer.Wk.zero_()
        layer.Wv.zero_()
        layer.Wv[0] = 1.0
        for m in layer.mlp:
            if hasattr(m, "weight"):
                m.weight.zero_()
    out = layer.get_output(
        Za=torch.tensor([1, 2]),
        batch_seg=torch.zeros(2, dtype=torch.long),
        Q=torch.tensor([1.0]),
    )
    assert out["charge_embedding"].shape == (2, 4)


def test_so3lr_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    build_params = {
        "dim_embedding": 8,
        "num_rbf": 8,
        "max_Za": 20,
        "cutoff_sr": 4.5,
        "cutoff_fn": "phys",
        "cutoff_lr": 6.0,
        "Bohr_in_R": _BOHR,
        "Hartree_in_E": _HARTREE_EV,
    }
    layers = [
        {"name": "RangeSeparation", "params": {"cutoff_fn": "phys"}},
        {"name": "BernsteinRBF", "params": {"cutoff_fn": "phys"}},
        {"name": "RandomAtomEmbedding"},
        {"name": "ChargeSpinEmbedding", "params": {"attribute": "charge"}},
        {"name": "GatherAtomEmbedding", "params": {"scale_by_sqrt_count": True}},
        {
            "name": "Core",
            "params": {
                "degrees": [1, 2],
                "num_features": 8,
                "num_heads": 2,
                "num_layers": 1,
                "message_normalization": "avg_num_neighbors",
                "avg_num_neighbors": 4.0,
                "initialize_ev_to_zeros": True,
                "cutoff_fn": "phys",
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "keep_feature": True,
            },
        },
        {"name": "PartialChargeReadout"},
        {"name": "ChargeConservation"},
        {"name": "HirshfeldReadout"},
        {"name": "ZBLRepulsionEnergy", "params": {"switch_off": 1.5}},
        {
            "name": "ElectrostaticEnergy",
            "params": {"flavor": "SO3LR", "electrostatic_energy_scale": 4.0},
        },
        {
            "name": "TSQDODispersionEnergy",
            "params": {
                "dispersion_energy_scale": 1.2,
                "cutoff_lr_damping": 2.0,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model("so3lr", layer_params=layers, build_params=build_params, verbose=0)
    model.eval()

    N = 4
    idx_i, idx_j = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    idx_i = torch.tensor(idx_i, dtype=torch.long)
    idx_j = torch.tensor(idx_j, dtype=torch.long)
    ra = torch.randn(N, 3, dtype=torch.float64, requires_grad=True) * 0.8
    out = model(
        {
            "Ra": ra,
            "Za": torch.tensor([1, 6, 1, 8], dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": torch.zeros(N, dtype=torch.long),
            "Q": torch.tensor([0.0]),
            "S": torch.tensor([0.0]),
        }
    )
    assert torch.isfinite(out["E"]).all()
    assert out["Fa"].shape == (N, 3)
    assert torch.isfinite(out["Fa"]).all()
    assert "ha" in out and "Qa" in out


def test_electrostatic_lr_cutoff_attribute_fixed():
    from enerzyme.models.layers import ElectrostaticEnergyLayer

    layer = ElectrostaticEnergyLayer(
        cutoff_sr=5.0, cutoff_lr=10.0, flavor="SpookyNet", Hartree_in_E=27.211
    )
    qa = torch.tensor([0.1, -0.1])
    dij = torch.tensor([2.0, 2.0])
    idx_i = torch.tensor([0, 1], dtype=torch.long)
    idx_j = torch.tensor([1, 0], dtype=torch.long)
    e = layer.get_E_ele_a(dij, qa, idx_i, idx_j)
    assert torch.isfinite(e).all()
