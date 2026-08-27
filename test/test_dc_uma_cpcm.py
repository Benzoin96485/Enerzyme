"""DC-UMA C-PCM Stage A: dielectric interface, adapter, residual loss, gradients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "enerzyme" / "config" / "dc_uma_cpcm_layers_example.yaml"
PLACEHOLDER = "UMA_CHECKPOINT"

ENERGY_ATOL = 1e-5
ENERGY_RTOL = 1e-5
FORCE_ATOL = 1e-5
FORCE_RTOL = 1e-5
FD_EPS = 1.0e-3
FD_ATOL = 5.0e-3
FD_RTOL = 5.0e-3


def _checkpoint_available() -> bool:
    from enerzyme.models.esen.runtime import resolve_uma_checkpoint

    try:
        return resolve_uma_checkpoint().is_file()
    except FileNotFoundError:
        return False


def _pinned_fairchem_ok() -> bool:
    from enerzyme.models.esen.runtime import PinnedFairchemError, ensure_pinned_fairchem

    try:
        ensure_pinned_fairchem()
    except PinnedFairchemError:
        return False
    return True


requires_live = pytest.mark.skipif(
    not (_checkpoint_available() and _pinned_fairchem_ok()),
    reason="need pinned fairchem PYTHONPATH and UMA_CHECKPOINT for live C-PCM tests",
)


def test_dc_uma_cpcm_config_shape():
    with open(CFG) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["build_params"]["checkpoint_path"] == PLACEHOLDER
    assert cfg["build_params"]["task_name"] == "omol"
    assert cfg["build_params"]["frozen"] is True
    assert cfg["build_params"]["adapter_hidden"] == [256, 128]
    assert cfg["build_params"]["adapter_mode"] == "stage_a"
    assert [layer["name"] for layer in cfg["layers"]] == ["Core"]
    assert CFG.read_text().count("/home/") == 0
    assert CFG.read_text().count("/global/") == 0


def test_get_ff_core_registers_dc_uma_cpcm():
    from enerzyme.models.ff import get_ff_core

    core, build_params, layer_params = get_ff_core("dc_uma_cpcm")
    assert core.__name__ == "DCUMACPCMWrapper"
    assert build_params["task_name"] == "omol"
    assert build_params["adapter_hidden"] == [256, 128]
    assert build_params["adapter_mode"] == "stage_a"
    assert layer_params == [{"name": "Core"}]


def test_dielectric_validation_rejects_missing_nan_and_below_one():
    from enerzyme.data.datatype import (
        dielectric_from_atoms,
        validate_dielectric_array,
        validate_dielectric_tensor,
        validate_dielectric_value,
    )

    with pytest.raises(ValueError, match="missing"):
        validate_dielectric_value(None)
    with pytest.raises(ValueError, match="NaN|Inf"):
        validate_dielectric_array(np.nan)
    with pytest.raises(ValueError, match=">= 1"):
        validate_dielectric_array(0.5)
    assert validate_dielectric_value(1.0) == pytest.approx(1.0)
    assert validate_dielectric_value(80) == pytest.approx(80.0)

    atoms = Atoms("H")
    with pytest.raises(ValueError, match="missing"):
        dielectric_from_atoms(atoms)
    atoms.info["dielectric"] = float("nan")
    with pytest.raises(ValueError, match="NaN|Inf"):
        dielectric_from_atoms(atoms)
    atoms.info["dielectric"] = 0.99
    with pytest.raises(ValueError, match=">= 1"):
        dielectric_from_atoms(atoms)
    atoms.info["dielectric"] = 7.3
    assert dielectric_from_atoms(atoms) == pytest.approx(7.3)

    with pytest.raises(ValueError, match=">= 1"):
        validate_dielectric_tensor(torch.tensor([1.0, 0.5]))
    flat = validate_dielectric_tensor(torch.tensor(80.0))
    assert flat.shape == (1,)
    assert float(flat.item()) == pytest.approx(80.0)


def _write_dielectric_aselmdb(db_path, dielectric, energy_ev=-1.0):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    forces_ev = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=energy_ev, forces=forces_ev)
    data = {"charge": 0, "spin": 1, "index": 0}
    if dielectric is not None:
        data["dielectric"] = dielectric
    with connect(str(db_path)) as db:
        db.write(atoms, data=data, index=0)
    return energy_ev, forces_ev


def test_aselmdb_dielectric_roundtrip(tmp_path):
    from enerzyme.data.datahub import ASELMDBDataset, SingleDataHub

    db_path = tmp_path / "eps.aselmdb"
    _write_dielectric_aselmdb(db_path, 80.0)
    ds = ASELMDBDataset(str(db_path), new_energy_unit="eV")
    assert "dielectric" in ds
    assert ds["dielectric"][0] == pytest.approx(80.0)

    hub = SingleDataHub(
        dump_dir=str(tmp_path / "out"),
        data_path=str(db_path),
        data_format="aselmdb",
        preload=False,
        features={"Ra": "Ra", "Za": "Za", "N": "N", "Q": "Q", "dielectric": "dielectric"},
        targets={"E": "E", "Fa": "Fa"},
        neighbor_list="",
        compressed=False,
    )
    assert float(hub.data["dielectric"][0]) == pytest.approx(80.0)


@pytest.mark.parametrize("bad_eps", [0.5, float("nan")])
def test_aselmdb_dielectric_rejects_invalid(tmp_path, bad_eps):
    from enerzyme.data.datahub import ASELMDBDataset

    db_path = tmp_path / "bad_eps.aselmdb"
    _write_dielectric_aselmdb(db_path, bad_eps)
    ds = ASELMDBDataset(str(db_path), new_energy_unit="eV")
    with pytest.raises(ValueError):
        _ = ds["dielectric"][0]


def test_aselmdb_missing_dielectric_rejected_when_declared(tmp_path):
    from enerzyme.data.datahub import SingleDataHub

    db_path = tmp_path / "no_eps.aselmdb"
    _write_dielectric_aselmdb(db_path, None)
    with pytest.raises((ValueError, KeyError)):
        SingleDataHub(
            dump_dir=str(tmp_path / "out"),
            data_path=str(db_path),
            data_format="aselmdb",
            preload=False,
            features={
                "Ra": "Ra",
                "Za": "Za",
                "N": "N",
                "dielectric": "dielectric",
            },
            targets={"E": "E"},
            neighbor_list="",
            compressed=False,
        )


def test_batch_dielectric_validation():
    from enerzyme.tasks.batch import _decorate_batch_input

    good = [
        (
            {
                "N": 2,
                "Ra": np.zeros((2, 3)),
                "Za": np.array([1, 1]),
                "dielectric": 80.0,
            },
            None,
        )
    ]
    feats, _ = _decorate_batch_input(good, dtype=torch.float32, device=torch.device("cpu"), otf_graph=False)
    assert feats["dielectric"].shape == (1,)
    assert float(feats["dielectric"].item()) == pytest.approx(80.0)

    bad = [
        (
            {
                "N": 2,
                "Ra": np.zeros((2, 3)),
                "Za": np.array([1, 1]),
                "dielectric": 0.2,
            },
            None,
        )
    ]
    with pytest.raises(ValueError, match=">= 1"):
        _decorate_batch_input(bad, dtype=torch.float32, device=torch.device("cpu"), otf_graph=False)


def test_calculator_reads_dielectric_and_rejects_invalid():
    from enerzyme.data.transform import Transform
    from enerzyme.tasks.calculator import ASECalculator

    class Dummy(torch.nn.Module):
        _input_fields = {"Ra", "Za", "batch_seg", "Q", "S", "dielectric"}

        def forward(self, net_input):
            n = net_input["Ra"].shape[0]
            g = int(net_input["batch_seg"].max().item()) + 1
            return {
                "E": torch.zeros(g, dtype=net_input["Ra"].dtype),
                "Fa": torch.zeros(n, 3, dtype=net_input["Ra"].dtype),
            }

    calc = ASECalculator(
        model=Dummy(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        transform=Transform(None),
        neighbor_list_type="full",
        Hartree_in_E=1.0,
    )
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    with pytest.raises(ValueError, match="missing"):
        calc.get_potential_energy(atoms)
    atoms.info["dielectric"] = 0.5
    with pytest.raises(ValueError, match=">= 1"):
        calc.get_potential_energy(atoms)
    atoms.info["dielectric"] = 1.0
    energy = calc.get_potential_energy(atoms)
    assert energy == pytest.approx(0.0)


def test_encode_dielectric_and_eps1_zero_gate():
    from enerzyme.models.esen.cpcm import StageASolvationAdapter, encode_dielectric

    eps = torch.tensor([1.0, np.e, 80.0], dtype=torch.float32)
    enc = encode_dielectric(eps)
    assert enc.shape == (3, 2)
    np.testing.assert_allclose(enc[0].detach().numpy(), [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(enc[1, 1].detach().numpy(), 1.0, atol=1e-5)
    np.testing.assert_allclose(enc[:, 0].detach().numpy(), (1.0 - 1.0 / eps).numpy(), atol=1e-6)

    torch.manual_seed(0)
    adapter = StageASolvationAdapter(sphere_channels=8, hidden=(16, 8))
    node = torch.randn(5, 8)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    dE = adapter(node, batch, torch.tensor([1.0, 1.0]))
    np.testing.assert_allclose(dE.detach().numpy(), [0.0, 0.0], atol=0.0)

    mixed = adapter(node, batch, torch.tensor([1.0, 80.0]))
    assert float(mixed[0].item()) == 0.0
    assert float(mixed[1].item()) != 0.0


def test_x_only_adapter_ignores_dielectric_above_one():
    from enerzyme.models.esen.cpcm import StageASolvationAdapter

    torch.manual_seed(3)
    adapter = StageASolvationAdapter(sphere_channels=8, hidden=(16, 8), x_only=True)
    adapter.eval()
    node = torch.randn(4, 8)
    batch = torch.zeros(4, dtype=torch.long)
    dE4 = adapter(node, batch, torch.tensor([4.0]))
    dE80 = adapter(node, batch, torch.tensor([80.0]))
    np.testing.assert_allclose(dE4.detach().numpy(), dE80.detach().numpy(), atol=1e-6)
    dE1 = adapter(node, batch, torch.tensor([1.0]))
    np.testing.assert_allclose(dE1.detach().numpy(), [0.0], atol=0.0)


def test_adapter_eps1_parameter_gradients_are_zero():
    from enerzyme.models.esen.cpcm import StageASolvationAdapter

    torch.manual_seed(1)
    adapter = StageASolvationAdapter(sphere_channels=8, hidden=(16, 8))
    node = torch.randn(4, 8)
    batch = torch.zeros(4, dtype=torch.long)
    dE = adapter(node, batch, torch.ones(1))
    dE.sum().backward()
    for param in adapter.parameters():
        assert param.grad is not None
        assert torch.count_nonzero(param.grad) == 0

    adapter.zero_grad()
    dE80 = adapter(node, batch, torch.tensor([80.0]))
    dE80.sum().backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in adapter.parameters())


def test_adapter_permutation_invariance():
    from enerzyme.models.esen.cpcm import StageASolvationAdapter

    torch.manual_seed(2)
    adapter = StageASolvationAdapter(sphere_channels=8, hidden=(16, 8))
    adapter.eval()
    node = torch.randn(6, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    eps = torch.tensor([4.0, 80.0])
    perm = torch.tensor([2, 0, 1, 5, 3, 4])
    dE = adapter(node, batch, eps)
    dE_perm = adapter(node[perm], batch[perm], eps)
    np.testing.assert_allclose(dE.detach().numpy(), dE_perm.detach().numpy(), atol=1e-6)


def test_residual_huber_per_atom_energy_and_force_rms():
    from enerzyme.models.loss import HuberLoss, ResidualHuberLoss

    batch_seg = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    output = {
        "dE": torch.tensor([2.0, 3.0]),
        "dFa": torch.zeros(5, 3),
        "batch_seg": batch_seg,
    }
    target = {
        "dE": torch.tensor([0.0, 0.0]),
        "dFa": torch.zeros(5, 3),
    }
    loss_fn = ResidualHuberLoss(delta=100.0, dE=1.0, dFa=0.0)
    # Per-atom errors 2/2=1 and 3/3=1; Huber≈MSE/2 in the quadratic region with large delta.
    loss = loss_fn(output, target)
    expected = 0.5 * ((1.0) ** 2)
    assert float(loss.item()) == pytest.approx(expected, rel=1e-5)

    # Same graph energy error (2 eV) is smaller per atom on a larger molecule.
    loss_n2 = ResidualHuberLoss(delta=100.0, dE=1.0, dFa=0.0)(
        {
            "dE": torch.tensor([2.0]),
            "dFa": torch.zeros(2, 3),
            "batch_seg": torch.zeros(2, dtype=torch.long),
        },
        {"dE": torch.tensor([0.0]), "dFa": torch.zeros(2, 3)},
    )
    loss_n4 = ResidualHuberLoss(delta=100.0, dE=1.0, dFa=0.0)(
        {
            "dE": torch.tensor([2.0]),
            "dFa": torch.zeros(4, 3),
            "batch_seg": torch.zeros(4, dtype=torch.long),
        },
        {"dE": torch.tensor([0.0]), "dFa": torch.zeros(4, 3)},
    )
    assert float(loss_n2.item()) == pytest.approx(0.5, rel=1e-5)
    assert float(loss_n4.item()) == pytest.approx(0.5 * 0.25, rel=1e-5)
    assert float(loss_n4.item()) < float(loss_n2.item())

    pred_f = torch.ones(4, 3)
    tgt_f = torch.zeros(4, 3)
    rms_fn = ResidualHuberLoss(delta=100.0, force_rms=2.0, dE=0.0, dFa=1.0)
    force_loss = rms_fn(
        {"dE": torch.zeros(1), "dFa": pred_f, "batch_seg": torch.zeros(4, dtype=torch.long)},
        {"dE": torch.zeros(1), "dFa": tgt_f},
    )
    # pred/rms = 0.5; mean 0.5 * (0.5**2)
    assert float(force_loss.item()) == pytest.approx(0.5 * 0.25, rel=1e-5)

    huber = HuberLoss(delta=1.0, E=1.0)
    scalar = huber({"E": torch.tensor([0.0])}, {"E": torch.tensor([0.0])})
    assert float(scalar.item()) == pytest.approx(0.0)


def test_trainer_reads_accumulate_grad_batches():
    from enerzyme.tasks.trainer import Trainer

    trainer = Trainer(
        out_dir=".",
        metric_config={},
        Splitter={"method": "random", "parts": [{"name": "training", "sources": []}]},
        accumulate_grad_batches=4,
        cuda=False,
        num_workers=1,
    )
    assert trainer.accumulate_grad_batches == 4


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _water():
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1, "name": "H2O"})
    atoms.pbc = False
    return atoms


def _oh_anion():
    atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
    atoms.info.update({"charge": -1, "spin": 1, "name": "OH-"})
    atoms.pbc = False
    return atoms


@pytest.fixture(scope="module")
def live_cpcm_models():
    if not (_checkpoint_available() and _pinned_fairchem_ok()):
        pytest.skip("need pinned fairchem and UMA_CHECKPOINT")
    from enerzyme.models.esen.core import UMAOmolWrapper
    from enerzyme.models.esen.cpcm import DCUMACPCMWrapper
    from enerzyme.models.esen.runtime import OMOL_TASK_NAME, resolve_uma_checkpoint

    device = _device()
    ckpt = str(resolve_uma_checkpoint())
    vacuum = UMAOmolWrapper(
        checkpoint_path=ckpt,
        task_name=OMOL_TASK_NAME,
        frozen=True,
        use_ema=True,
        inference_settings="batch",
        strict=True,
    ).to(device)
    cpcm = DCUMACPCMWrapper(
        checkpoint_path=ckpt,
        task_name=OMOL_TASK_NAME,
        frozen=True,
        use_ema=True,
        inference_settings="batch",
        strict=True,
    ).to(device)
    return vacuum, cpcm, device


def _forward(wrapper, atoms, device, dielectric=None):
    ra = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    za = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
    charge = torch.tensor([int(atoms.info["charge"])], dtype=torch.long, device=device)
    spin = torch.tensor([int(atoms.info["spin"]) - 1], dtype=torch.long, device=device)
    kwargs = dict(Ra=ra, Za=za, batch_seg=batch, Q=charge, S=spin)
    if dielectric is not None:
        kwargs["dielectric"] = torch.as_tensor(dielectric, dtype=torch.float32, device=device)
    return wrapper.get_output(**kwargs)


@requires_live
def test_eps1_identity_matches_uma_omol(live_cpcm_models):
    vacuum, cpcm, device = live_cpcm_models
    cpcm.eval()
    atoms = _water()
    vac = _forward(vacuum, atoms, device)
    sol = _forward(cpcm, atoms, device, dielectric=1.0)
    np.testing.assert_allclose(
        sol["E"].detach().cpu().numpy(),
        vac["E"].detach().cpu().numpy(),
        rtol=ENERGY_RTOL,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(
        sol["Fa"].detach().cpu().numpy(),
        vac["Fa"].detach().cpu().numpy(),
        rtol=FORCE_RTOL,
        atol=FORCE_ATOL,
    )
    np.testing.assert_allclose(sol["dE"].detach().cpu().numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(sol["dFa"].detach().cpu().numpy(), 0.0, atol=0.0)


@requires_live
def test_eps1_adapter_gradients_zero_and_nonzero_at_water(live_cpcm_models):
    _, cpcm, device = live_cpcm_models
    cpcm.train()
    atoms = _water()
    out = _forward(cpcm, atoms, device, dielectric=1.0)
    (out["dE"].sum() + out["dFa"].pow(2).sum()).backward()
    for param in cpcm.adapter.parameters():
        if param.grad is not None:
            assert torch.count_nonzero(param.grad) == 0

    cpcm.zero_grad()
    out80 = _forward(cpcm, atoms, device, dielectric=80.0)
    (out80["dE"].sum() + out80["dFa"].pow(2).sum()).backward()
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in cpcm.adapter.parameters()
    )
    cpcm.eval()
    cpcm.zero_grad()


def _finite_difference_forces(wrapper, atoms, device, dielectric, energy_key="E"):
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    n = len(atoms)
    fd = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        for d in range(3):
            plus = atoms.copy()
            minus = atoms.copy()
            p = pos.copy()
            m = pos.copy()
            p[i, d] += FD_EPS
            m[i, d] -= FD_EPS
            plus.set_positions(p)
            minus.set_positions(m)
            plus.info.update(atoms.info)
            minus.info.update(atoms.info)
            ep = float(_forward(wrapper, plus, device, dielectric)[energy_key].detach().cpu().numpy().reshape(-1)[0])
            em = float(_forward(wrapper, minus, device, dielectric)[energy_key].detach().cpu().numpy().reshape(-1)[0])
            fd[i, d] = -(ep - em) / (2.0 * FD_EPS)
    return fd


@requires_live
@pytest.mark.parametrize("eps", [1.001, 1.05])
def test_finite_difference_forces_and_nonzero_residual(live_cpcm_models, eps):
    vacuum, cpcm, device = live_cpcm_models
    cpcm.eval()
    atoms = _water()
    out = _forward(cpcm, atoms, device, dielectric=eps)
    vac = _forward(vacuum, atoms, device)
    assert not np.allclose(out["E"].detach().cpu().numpy(), vac["E"].detach().cpu().numpy(), atol=1e-8)
    fa = out["Fa"].detach().cpu().numpy()
    fd = _finite_difference_forces(cpcm, atoms, device, dielectric=eps, energy_key="E")
    np.testing.assert_allclose(fa, fd, rtol=FD_RTOL, atol=FD_ATOL)
    dfa = out["dFa"].detach().cpu().numpy()
    dfd = _finite_difference_forces(cpcm, atoms, device, dielectric=eps, energy_key="dE")
    np.testing.assert_allclose(dfa, dfd, rtol=FD_RTOL, atol=FD_ATOL)


@requires_live
def test_mixed_eps_batch_matches_singles(live_cpcm_models):
    _, cpcm, device = live_cpcm_models
    cpcm.eval()
    mols = [_water(), _oh_anion()]
    eps = [1.0, 80.0]
    ra = torch.cat(
        [torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device) for atoms in mols]
    )
    za = torch.cat(
        [torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device) for atoms in mols]
    )
    batch = torch.cat(
        [torch.full((len(atoms),), i, dtype=torch.long, device=device) for i, atoms in enumerate(mols)]
    )
    charge = torch.tensor([int(atoms.info["charge"]) for atoms in mols], dtype=torch.long, device=device)
    spin = torch.tensor([int(atoms.info["spin"]) - 1 for atoms in mols], dtype=torch.long, device=device)
    dielectric = torch.tensor(eps, dtype=torch.float32, device=device)
    mixed = cpcm.get_output(Ra=ra, Za=za, batch_seg=batch, Q=charge, S=spin, dielectric=dielectric)
    offset = 0
    for i, atoms in enumerate(mols):
        single = _forward(cpcm, atoms, device, dielectric=eps[i])
        n = len(atoms)
        np.testing.assert_allclose(
            mixed["E"][i].detach().cpu().numpy(),
            single["E"].detach().cpu().numpy().reshape(-1),
            rtol=ENERGY_RTOL,
            atol=ENERGY_ATOL,
        )
        np.testing.assert_allclose(
            mixed["Fa"][offset : offset + n].detach().cpu().numpy(),
            single["Fa"].detach().cpu().numpy(),
            rtol=FORCE_RTOL,
            atol=FORCE_ATOL,
        )
        offset += n


@requires_live
def test_translation_rotation_permutation(live_cpcm_models):
    _, cpcm, device = live_cpcm_models
    cpcm.eval()
    atoms = _water()
    eps = 7.3
    base = _forward(cpcm, atoms, device, dielectric=eps)
    e0 = base["E"].detach().cpu().numpy().reshape(-1)
    f0 = base["Fa"].detach().cpu().numpy()

    shifted = atoms.copy()
    shifted.info.update(atoms.info)
    shifted.set_positions(atoms.get_positions() + np.array([1.2, -0.4, 0.7]))
    shift_out = _forward(cpcm, shifted, device, dielectric=eps)
    np.testing.assert_allclose(shift_out["E"].detach().cpu().numpy().reshape(-1), e0, rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    np.testing.assert_allclose(shift_out["Fa"].detach().cpu().numpy(), f0, rtol=FORCE_RTOL, atol=FORCE_ATOL)

    rng = np.random.default_rng(20260817)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = atoms.copy()
    rotated.info.update(atoms.info)
    rotated.set_positions(atoms.get_positions() @ q)
    rot_out = _forward(cpcm, rotated, device, dielectric=eps)
    np.testing.assert_allclose(rot_out["E"].detach().cpu().numpy().reshape(-1), e0, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        rot_out["Fa"].detach().cpu().numpy(),
        f0 @ q,
        rtol=1e-4,
        atol=1e-4,
    )

    perm = np.array([2, 0, 1])
    permuted = atoms.copy()
    permuted.info.update(atoms.info)
    permuted.set_atomic_numbers(atoms.get_atomic_numbers()[perm])
    permuted.set_positions(atoms.get_positions()[perm])
    perm_out = _forward(cpcm, permuted, device, dielectric=eps)
    np.testing.assert_allclose(perm_out["E"].detach().cpu().numpy().reshape(-1), e0, rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    np.testing.assert_allclose(perm_out["Fa"].detach().cpu().numpy(), f0[perm], rtol=FORCE_RTOL, atol=FORCE_ATOL)


@requires_live
def test_train_skips_vacuum_eval_adds_vacuum(live_cpcm_models):
    vacuum, cpcm, device = live_cpcm_models
    atoms = _water()
    cpcm.train()
    train_out = _forward(cpcm, atoms, device, dielectric=80.0)
    cpcm.eval()
    eval_out = _forward(cpcm, atoms, device, dielectric=80.0)
    vac = _forward(vacuum, atoms, device)
    np.testing.assert_allclose(
        train_out["E"].detach().cpu().numpy(),
        train_out["dE"].detach().cpu().numpy(),
        atol=0.0,
    )
    np.testing.assert_allclose(
        eval_out["dE"].detach().cpu().numpy(),
        train_out["dE"].detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        eval_out["E"].detach().cpu().numpy(),
        vac["E"].detach().cpu().numpy() + eval_out["dE"].detach().cpu().numpy(),
        rtol=ENERGY_RTOL,
        atol=ENERGY_ATOL,
    )
    assert not np.allclose(
        train_out["E"].detach().cpu().numpy(),
        eval_out["E"].detach().cpu().numpy(),
        atol=1e-6,
    )


STAGE_B_CFG = ROOT / "enerzyme" / "config" / "dc_uma_cpcm_stage_b.yaml"


def test_stage_b_config_shape():
    with open(STAGE_B_CFG) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["build_params"]["adapter_mode"] == "stage_b"
    assert cfg["build_params"]["adapter_hidden"] == [256, 128]
    assert cfg["trainer"]["seed"] == 20260818
    assert cfg["trainer"]["lowdin_aux_weight"] == pytest.approx(0.05)
    assert cfg["trainer"]["gate"]["charged_energy_improve"] == pytest.approx(0.10)
    assert cfg["trainer"]["gate"]["force_degrade_max"] == pytest.approx(0.05)
    assert STAGE_B_CFG.read_text().count("/home/") == 0
    assert STAGE_B_CFG.read_text().count("/global/") == 0


def test_stillinger_gb_self_energy_and_charge_conservation():
    from enerzyme.models.esen.cpcm import (
        COULOMB_KE_EV_A,
        MIN_BORN_RADIUS,
        StageBSolvationAdapter,
        stillinger_gb_energy,
    )

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)
    qa = torch.tensor([0.4, -0.4], dtype=torch.float32)
    radii = torch.tensor([1.0, 1.5], dtype=torch.float32)
    batch = torch.zeros(2, dtype=torch.long)
    e_self, e_pair = stillinger_gb_energy(pos, qa, radii, batch)
    expected_self = -0.5 * COULOMB_KE_EV_A * ((0.4**2) / 1.0 + ((-0.4) ** 2) / 1.5)
    assert float(e_self.item()) == pytest.approx(expected_self, rel=1e-5)
    assert float(e_pair.item()) != 0.0
    assert MIN_BORN_RADIUS == pytest.approx(0.5)

    torch.manual_seed(0)
    adapter = StageBSolvationAdapter(sphere_channels=8, hidden=(16, 8))
    node = torch.randn(4, 8)
    za = torch.tensor([8, 1, 1, 6], dtype=torch.long)
    seg = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    q_tot = torch.tensor([-1.0, 2.0])
    qa_out = adapter.conserved_charges(node, seg, za, q_tot)
    sums = torch.zeros(2)
    sums.index_add_(0, seg, qa_out)
    np.testing.assert_allclose(sums.detach().numpy(), q_tot.numpy(), atol=1e-5)


def test_stage_b_eps1_zero_gate_and_component_sum():
    from enerzyme.models.esen.cpcm import StageBSolvationAdapter

    torch.manual_seed(1)
    adapter = StageBSolvationAdapter(sphere_channels=8, hidden=(16, 8))
    node = torch.randn(3, 8)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    za = torch.tensor([8, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    q = torch.tensor([-1.0])
    out1 = adapter(node, batch, torch.tensor([1.0]), pos, za, q)
    np.testing.assert_allclose(out1["dE"].detach().numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(out1["E_local"].detach().numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(out1["E_self"].detach().numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(out1["E_pair"].detach().numpy(), [0.0], atol=0.0)

    out80 = adapter(node, batch, torch.tensor([80.0]), pos, za, q)
    recon = out80["E_local"] + out80["E_self"] + out80["E_pair"]
    np.testing.assert_allclose(out80["dE"].detach().numpy(), recon.detach().numpy(), atol=1e-6)
    assert float(out80["dE"].item()) != 0.0


def test_stage_b_batch_isolation_and_finite_difference_forces():
    from enerzyme.models.esen.cpcm import StageBSolvationAdapter, energy_gradient_forces, stillinger_gb_energy

    torch.manual_seed(2)
    adapter = StageBSolvationAdapter(sphere_channels=8, hidden=(16, 8))
    adapter.eval()
    node_a = torch.randn(2, 8)
    node_b = torch.randn(3, 8)
    pos_a = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float32)
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.4, 0.7, 0.0]], dtype=torch.float32)
    za_a = torch.tensor([8, 1], dtype=torch.long)
    za_b = torch.tensor([6, 1, 1], dtype=torch.long)
    q_a = torch.tensor([-1.0])
    q_b = torch.tensor([1.0])
    eps_a = torch.tensor([80.0])
    eps_b = torch.tensor([4.0])
    single_a = adapter(node_a, torch.zeros(2, dtype=torch.long), eps_a, pos_a, za_a, q_a)
    single_b = adapter(node_b, torch.zeros(3, dtype=torch.long), eps_b, pos_b, za_b, q_b)

    node = torch.cat([node_a, node_b], dim=0)
    pos = torch.cat([pos_a, pos_b], dim=0)
    za = torch.cat([za_a, za_b], dim=0)
    batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    mixed = adapter(node, batch, torch.tensor([80.0, 4.0]), pos, za, torch.tensor([-1.0, 1.0]))
    np.testing.assert_allclose(
        mixed["dE"].detach().numpy(),
        torch.cat([single_a["dE"], single_b["dE"]]).detach().numpy(),
        atol=1e-5,
    )

    pos_shift = pos.clone()
    pos_shift[:2] = pos_shift[:2] + 3.0
    shifted = adapter(node, batch, torch.tensor([80.0, 4.0]), pos_shift, za, torch.tensor([-1.0, 1.0]))
    np.testing.assert_allclose(
        shifted["dE"][1].detach().numpy(), mixed["dE"][1].detach().numpy(), atol=1e-5
    )

    pos_fd = pos_a.clone().requires_grad_(True)
    qa = torch.tensor([0.3, -0.3], dtype=torch.float32)
    radii = torch.tensor([1.1, 1.2], dtype=torch.float32)
    e_self, e_pair = stillinger_gb_energy(pos_fd, qa, radii, torch.zeros(2, dtype=torch.long))
    fa = energy_gradient_forces(e_self + e_pair, pos_fd, create_graph=False)
    fd = np.zeros((2, 3), dtype=np.float64)
    pos0 = pos_a.detach().numpy()
    eps = 1.0e-3
    for i in range(2):
        for d in range(3):
            pplus = pos_a.clone()
            pminus = pos_a.clone()
            pplus[i, d] += eps
            pminus[i, d] -= eps
            ep = stillinger_gb_energy(pplus, qa, radii, torch.zeros(2, dtype=torch.long))
            em = stillinger_gb_energy(pminus, qa, radii, torch.zeros(2, dtype=torch.long))
            fd[i, d] = -((ep[0] + ep[1]) - (em[0] + em[1])).item() / (2.0 * eps)
    np.testing.assert_allclose(fa.detach().numpy(), fd, rtol=5e-3, atol=5e-3)


def test_stage_b_warm_start_loads_stage_a_mlp(tmp_path):
    from enerzyme.models.esen.cpcm import (
        StageASolvationAdapter,
        StageBSolvationAdapter,
        load_stage_a_adapter_weights,
    )

    torch.manual_seed(4)
    stage_a = StageASolvationAdapter(sphere_channels=8, hidden=(16, 8))
    ckpt = tmp_path / "adapter.pt"
    torch.save({"adapter_state_dict": stage_a.state_dict()}, ckpt)
    stage_b = StageBSolvationAdapter(sphere_channels=8, hidden=(16, 8))
    info = load_stage_a_adapter_weights(stage_b, ckpt)
    assert any(k.startswith("mlp.") for k in info["loaded_keys"])
    assert any("charge_head" in k for k in info["missing_keys"])
    for key, tensor in stage_a.state_dict().items():
        np.testing.assert_allclose(
            stage_b.state_dict()[key].detach().numpy(),
            tensor.detach().numpy(),
            atol=0.0,
        )


def test_stage_b_keep_drop_gate():
    from enerzyme.data.dc_uma_cpcm.diagnostic import decide_stage_b_keep_drop

    stage_a = {
        "test": {"dFa_mae": 0.026},
        "test_slices": {
            "by_charge": {
                "-1": {"n": 126, "dE_mae": 0.20},
                "1": {"n": 126, "dE_mae": 0.10},
            }
        },
    }
    keep_b = {
        "test": {"dFa_mae": 0.0265},
        "test_slices": {
            "by_charge": {
                "-1": {"n": 126, "dE_mae": 0.16},
                "1": {"n": 126, "dE_mae": 0.08},
            }
        },
    }
    gate = decide_stage_b_keep_drop(stage_a, keep_b)
    assert gate["keep"] is True
    assert gate["decision"] == "keep"

    drop_b = {
        "test": {"dFa_mae": 0.030},
        "test_slices": {
            "by_charge": {
                "-1": {"n": 126, "dE_mae": 0.19},
                "1": {"n": 126, "dE_mae": 0.10},
            }
        },
    }
    no = decide_stage_b_keep_drop(stage_a, drop_b)
    assert no["keep"] is False
    assert "charged_test_energy_mae_not_improved_10pct" in no["flags"]
    assert "full_test_force_mae_degraded_over_5pct" in no["flags"]


@requires_live
def test_stage_b_live_eps1_identity_and_fd(live_cpcm_models):
    vacuum, _, device = live_cpcm_models
    from enerzyme.models.esen.cpcm import DCUMACPCMWrapper
    from enerzyme.models.esen.runtime import OMOL_TASK_NAME, resolve_uma_checkpoint

    cpcm = DCUMACPCMWrapper(
        checkpoint_path=str(resolve_uma_checkpoint()),
        task_name=OMOL_TASK_NAME,
        frozen=True,
        use_ema=True,
        inference_settings="batch",
        strict=True,
        adapter_mode="stage_b",
    ).to(device)
    cpcm.eval()
    atoms = _water()
    vac = _forward(vacuum, atoms, device)
    sol = _forward(cpcm, atoms, device, dielectric=1.0)
    np.testing.assert_allclose(
        sol["E"].detach().cpu().numpy(),
        vac["E"].detach().cpu().numpy(),
        rtol=ENERGY_RTOL,
        atol=ENERGY_ATOL,
    )
    np.testing.assert_allclose(sol["dE"].detach().cpu().numpy(), [0.0], atol=0.0)
    np.testing.assert_allclose(sol["dFa"].detach().cpu().numpy(), 0.0, atol=0.0)

    charged = _oh_anion()
    out = _forward(cpcm, charged, device, dielectric=80.0)
    qa = out["Qa"].detach().cpu().numpy()
    assert qa.sum() == pytest.approx(-1.0, abs=1e-4)
    dfa = out["dFa"].detach().cpu().numpy()
    dfd = _finite_difference_forces(cpcm, charged, device, dielectric=80.0, energy_key="dE")
    np.testing.assert_allclose(dfa, dfd, rtol=FD_RTOL, atol=FD_ATOL)
