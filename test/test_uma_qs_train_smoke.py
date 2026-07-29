"""Smoke for uma_qs training config + optional 1-epoch live train (PR #81).

Config shape checks always run. Live training runs only when:
- ``UMA_CHECKPOINT`` points at a UMA ``.pt`` file, and
- ``fairchem`` imports successfully.

Host-absolute checkpoint / cache paths stay on the CLI / env — never in the
committed YAML (placeholder ``UMA_CHECKPOINT``).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "L3-COMT-aselmdb-smoke"
CFG = EXAMPLE / "config" / "train_uma_qs.yaml"
PLACEHOLDER = "UMA_CHECKPOINT"


def test_train_uma_qs_config_shape():
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["Datahub"]["data_format"] == "pickle"
    assert cfg["Datahub"]["data_path"] == "fixtures/fragments_tiny.pkl"
    assert "S" in cfg["Datahub"]["features"]
    assert "Q" in cfg["Datahub"]["targets"] and "S" in cfg["Datahub"]["targets"]

    ff = cfg["Modelhub"]["internal_FFs"]["FF_UMA_QS"]
    assert ff["architecture"] == "uma_qs"
    assert ff["build_params"]["checkpoint_path"] == PLACEHOLDER
    assert ff["build_params"]["frozen_backbone"] is True
    names = [layer["name"] for layer in ff["layers"]]
    assert names == ["Core", "SimpleReadout", "ChargeConservation", "SpinConservation"]
    assert CFG.read_text().count("/home/") == 0
    assert CFG.read_text().count("/gridsan/") == 0


def test_train_uma_qs_fixture_exists():
    assert (EXAMPLE / "fixtures" / "fragments_tiny.pkl").is_file()


@pytest.mark.skipif(
    not os.environ.get("UMA_CHECKPOINT"),
    reason="set UMA_CHECKPOINT to a local uma-*.pt for live train smoke",
)
def test_uma_qs_one_epoch_train(tmp_path: Path):
    ckpt = Path(os.environ["UMA_CHECKPOINT"]).expanduser()
    assert ckpt.is_file(), f"UMA_CHECKPOINT not a file: {ckpt}"

    fairchem = pytest.importorskip("fairchem.core")
    _ = fairchem  # silence unused

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    # Repo-relative paths → absolute only in the ephemeral resolved config
    cfg["Datahub"]["data_path"] = str(EXAMPLE / "fixtures" / "fragments_tiny.pkl")
    cfg["Modelhub"]["internal_FFs"]["FF_UMA_QS"]["build_params"]["checkpoint_path"] = str(
        ckpt
    )
    # Prefer GPU when available; allow CPU fallback for CI-like environments
    import torch

    cfg["Trainer"]["cuda"] = bool(torch.cuda.is_available())
    cfg["Trainer"]["num_workers"] = 1

    # UMA checkpoints are large; prefer a roomy local scratch over pytest's /tmp nest.
    scratch = Path(os.environ.get("UMA_SMOKE_OUT", "/tmp/uma_qs_train_smoke"))
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / f"run_{os.getpid()}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    resolved = work / "train_uma_qs_resolved.yaml"
    with open(resolved, "w") as f:
        yaml.safe_dump(cfg, f)

    out_dir = work / "out"
    out_dir.mkdir()

    from enerzyme.train import FFTrain

    try:
        trainer = FFTrain(config_path=str(resolved), out_dir=str(out_dir))
        trainer.train_all()

        model_dirs = [
            p for p in out_dir.iterdir() if p.is_dir() and "uma_qs" in p.name.lower()
        ]
        assert model_dirs, f"no uma_qs output under {out_dir}: {list(out_dir.iterdir())}"
        ckpts = list(model_dirs[0].glob("model*.pth"))
        assert ckpts, f"expected model*.pth under {model_dirs[0]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
