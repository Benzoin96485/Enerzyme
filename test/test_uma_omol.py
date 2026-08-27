"""UMA-s-1p2 OMol wrapper: pin, registry, official calculator parity.

Live tests load ``uma-s-1p2`` and compare against
``FAIRChemCalculator(task_name="omol", inference_settings="batch")``.
They require the pinned fairchem source on ``PYTHONPATH`` and
``UMA_CHECKPOINT``. GPU is used when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "enerzyme" / "config" / "uma_omol_layers_example.yaml"
PLACEHOLDER = "UMA_CHECKPOINT"

ENERGY_ATOL = 1e-5
ENERGY_RTOL = 1e-5
FORCE_ATOL = 1e-5
FORCE_RTOL = 1e-5


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
    reason="need pinned fairchem PYTHONPATH and UMA_CHECKPOINT for live OMol tests",
)


def test_uma_omol_config_shape():
    with open(CFG) as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["build_params"]["checkpoint_path"] == PLACEHOLDER
    assert cfg["build_params"]["task_name"] == "omol"
    assert cfg["build_params"]["frozen"] is True
    assert cfg["build_params"]["use_ema"] is True
    assert cfg["build_params"]["inference_settings"] == "batch"
    assert [layer["name"] for layer in cfg["layers"]] == ["Core"]
    assert CFG.read_text().count("/home/") == 0
    assert CFG.read_text().count("/global/") == 0


def test_get_ff_core_registers_uma_omol():
    from enerzyme.models.ff import get_ff_core

    core, build_params, layer_params = get_ff_core("uma_omol")
    assert core.__name__ == "UMAOmolWrapper"
    assert build_params["task_name"] == "omol"
    assert build_params["inference_settings"] == "batch"
    assert layer_params == [{"name": "Core"}]


def test_runtime_pin_constants():
    from enerzyme.models.esen.runtime import (
        FAIRCHEM_CACHE_DIR,
        PINNED_FAIRCHEM_COMMIT,
        PINNED_FAIRCHEM_SRC,
        UMA_MODEL_NAME,
    )

    assert PINNED_FAIRCHEM_COMMIT == "de5db01588da57665bde96966091d268a9b6b8f7"
    assert PINNED_FAIRCHEM_SRC.name == "src"
    assert FAIRCHEM_CACHE_DIR.as_posix().endswith("models/fairchem")
    assert UMA_MODEL_NAME == "uma-s-1p2"


def test_fields_to_atomic_data_spin_and_ase_molecule_cell():
    import torch

    from enerzyme.models.esen.core import _fields_to_atomic_data

    ra = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], dtype=torch.float32)
    za = torch.tensor([8, 1], dtype=torch.long)
    batch = torch.tensor([0, 0], dtype=torch.long)
    charge = torch.tensor([-1], dtype=torch.long)
    spin = torch.tensor([0], dtype=torch.long)  # Enerzyme S = multiplicity - 1
    data = _fields_to_atomic_data(
        ra, za, batch, charge=charge, spin=spin, vacuum_pbc=False, task_name="omol"
    )
    assert int(data.charge.item()) == -1
    assert int(data.spin.item()) == 1  # UMA multiplicity
    assert not bool(data.pbc.any())
    assert data.dataset == ["omol"]
    assert data.cell.shape == (1, 3, 3)


@requires_live
def test_pinned_fairchem_commit_and_checkpoint_hash():
    from enerzyme.models.esen.runtime import (
        PINNED_FAIRCHEM_COMMIT,
        collect_runtime_versions,
        ensure_pinned_fairchem,
        sha256_file,
        resolve_uma_checkpoint,
    )

    fairchem = ensure_pinned_fairchem()
    versions = collect_runtime_versions()
    assert versions["fairchem_git_commit"].startswith(PINNED_FAIRCHEM_COMMIT[:12])
    ckpt = resolve_uma_checkpoint()
    assert versions["uma_checkpoint_sha256"] == sha256_file(ckpt)
    assert "fairchem" in versions["fairchem_file"]
    assert Path(fairchem.core.__file__).resolve().as_posix().endswith(
        "fairchem/core/__init__.py"
    )


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _molecules():
    from ase.build import molecule
    from ase import Atoms

    water = molecule("H2O")
    water.info.update({"charge": 0, "spin": 1, "name": "H2O"})
    water.pbc = False

    oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
    oh.info.update({"charge": -1, "spin": 1, "name": "OH-"})
    oh.pbc = False

    h3o = Atoms(
        "OHHH",
        positions=[
            [0.0000, 0.0000, 0.0000],
            [0.9584, 0.0000, 0.0000],
            [-0.2393, 0.9280, 0.0000],
            [-0.2393, -0.3570, 0.8560],
        ],
    )
    h3o.info.update({"charge": 1, "spin": 1, "name": "H3O+"})
    h3o.pbc = False

    o2 = molecule("O2")
    o2.info.update({"charge": 0, "spin": 3, "name": "O2-triplet"})
    o2.pbc = False
    return [water, oh, h3o, o2]


@pytest.fixture(scope="module")
def live_models():
    if not (_checkpoint_available() and _pinned_fairchem_ok()):
        pytest.skip("need pinned fairchem and UMA_CHECKPOINT")
    import torch
    from fairchem.core import FAIRChemCalculator
    from enerzyme.models.esen.core import UMAOmolWrapper
    from enerzyme.models.esen.runtime import OMOL_TASK_NAME, resolve_uma_checkpoint

    device = _device()
    ckpt = str(resolve_uma_checkpoint())
    wrapper = UMAOmolWrapper(
        checkpoint_path=ckpt,
        task_name=OMOL_TASK_NAME,
        frozen=True,
        use_ema=True,
        inference_settings="batch",
        strict=True,
    ).to(device)
    calc = FAIRChemCalculator.from_model_checkpoint(
        ckpt,
        task_name=OMOL_TASK_NAME,
        inference_settings="batch",
        device=str(device),
    )
    return wrapper, calc, device


def _wrapper_energy_forces(wrapper, atoms, device):
    import torch

    ra = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    za = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
    charge = torch.tensor([int(atoms.info["charge"])], dtype=torch.long, device=device)
    spin = torch.tensor([int(atoms.info["spin"]) - 1], dtype=torch.long, device=device)
    out = wrapper.get_output(Ra=ra, Za=za, batch_seg=batch, Q=charge, S=spin)
    energy = out["E"].detach().cpu().numpy().reshape(-1)
    forces = out["Fa"].detach().cpu().numpy()
    return energy, forces


def _official_energy_forces(calc, atoms):
    sample = atoms.copy()
    sample.pbc = False
    sample.calc = calc
    energy = float(sample.get_potential_energy())
    forces = np.asarray(sample.get_forces(), dtype=float)
    return energy, forces


@requires_live
def test_wrapper_strict_load_keeps_omol_head(live_models):
    wrapper, _, _ = live_models
    assert wrapper.hydra.backbone.regress_config.direct_forces is False
    assert wrapper._energy_task_name in wrapper.hydra.tasks
    assert wrapper._forces_task_name in wrapper.hydra.tasks
    assert wrapper._efs_head_names
    for param in wrapper.uma.parameters():
        assert param.requires_grad is False


@requires_live
@pytest.mark.parametrize("index", range(4), ids=["neutral", "anion", "cation", "triplet"])
def test_official_calculator_parity_single(live_models, index):
    wrapper, calc, device = live_models
    atoms = _molecules()[index]
    off_e, off_f = _official_energy_forces(calc, atoms)
    wrap_e, wrap_f = _wrapper_energy_forces(wrapper, atoms, device)
    np.testing.assert_allclose(wrap_e, [off_e], rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
    np.testing.assert_allclose(wrap_f, off_f, rtol=FORCE_RTOL, atol=FORCE_ATOL)


@requires_live
def test_official_calculator_parity_mixed_batch(live_models):
    import torch

    wrapper, calc, device = live_models
    mols = _molecules()[:3]
    official = [_official_energy_forces(calc, atoms) for atoms in mols]

    ra = torch.cat(
        [
            torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
            for atoms in mols
        ]
    )
    za = torch.cat(
        [
            torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
            for atoms in mols
        ]
    )
    batch = torch.cat(
        [
            torch.full((len(atoms),), i, dtype=torch.long, device=device)
            for i, atoms in enumerate(mols)
        ]
    )
    charge = torch.tensor(
        [int(atoms.info["charge"]) for atoms in mols], dtype=torch.long, device=device
    )
    spin = torch.tensor(
        [int(atoms.info["spin"]) - 1 for atoms in mols], dtype=torch.long, device=device
    )
    out = wrapper.get_output(Ra=ra, Za=za, batch_seg=batch, Q=charge, S=spin)
    wrap_e = out["E"].detach().cpu().numpy().reshape(-1)
    wrap_f = out["Fa"].detach().cpu().numpy()

    offset = 0
    for i, atoms in enumerate(mols):
        n = len(atoms)
        np.testing.assert_allclose(
            wrap_e[i], official[i][0], rtol=ENERGY_RTOL, atol=ENERGY_ATOL
        )
        np.testing.assert_allclose(
            wrap_f[offset : offset + n],
            official[i][1],
            rtol=FORCE_RTOL,
            atol=FORCE_ATOL,
        )
        # Mixed-batch prediction must match the isolated single-structure call.
        single_e, single_f = _wrapper_energy_forces(wrapper, atoms, device)
        np.testing.assert_allclose(wrap_e[i], single_e[0], rtol=ENERGY_RTOL, atol=ENERGY_ATOL)
        np.testing.assert_allclose(
            wrap_f[offset : offset + n], single_f, rtol=FORCE_RTOL, atol=FORCE_ATOL
        )
        offset += n
