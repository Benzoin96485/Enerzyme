"""Smoke for uma_flow_qs training config + optional 1-epoch live train (PR #82).

Config shape checks always run. Live training runs only when:
- ``UMA_CHECKPOINT`` points at a UMA ``.pt`` file, and
- ``fairchem`` and ``torchdiffeq`` import successfully.

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
CFG = EXAMPLE / "config" / "train_uma_flow_qs.yaml"
FIXTURE = EXAMPLE / "fixtures" / "fragments_flow_tiny.pkl"
PLACEHOLDER = "UMA_CHECKPOINT"


def test_train_uma_flow_qs_config_shape():
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["Datahub"]["data_format"] == "pickle"
    assert cfg["Datahub"]["data_path"] == "fixtures/fragments_flow_tiny.pkl"
    assert cfg["Datahub"]["global_transforms"]["uniform_qs_init"] is True
    assert "Qa" in cfg["Datahub"]["targets"] and "Sa" in cfg["Datahub"]["targets"]

    ff = cfg["Modelhub"]["internal_FFs"]["FF_UMA_FLOW"]
    assert ff["architecture"] == "uma_flow_qs"
    assert ff["build_params"]["checkpoint_path"] == PLACEHOLDER
    assert ff["build_params"]["frozen_backbone"] is True
    names = [layer["name"] for layer in ff["layers"]]
    assert names == [
        "ScalarDenseEmbedding",
        "ScalarDenseEmbedding",
        "GraphScalarBroadcastEmbedding",
        "GatherAtomEmbedding",
        "Core",
        "VelocityReadout",
        "VelocityConservation",
    ]
    assert "cfm" in ff["loss"]
    gen = cfg["Trainer"]["Generator"]
    assert gen["enabled"] is True
    assert gen["ode_predict"] is True
    text = CFG.read_text()
    assert "/home/" not in text
    assert "/gridsan/" not in text


def test_train_uma_flow_qs_fixture_exists():
    assert FIXTURE.is_file()


def test_torchdiffeq_is_optional_extra():
    setup = (ROOT / "setup.py").read_text()
    assert '"torchdiffeq"' not in setup.split("install_requires")[1].split("]")[0]
    assert '"flow"' in setup and "torchdiffeq" in setup


@pytest.mark.skipif(
    not os.environ.get("UMA_CHECKPOINT"),
    reason="set UMA_CHECKPOINT to a local uma-*.pt for live flow train smoke",
)
def test_uma_flow_qs_one_epoch_train():
    ckpt = Path(os.environ["UMA_CHECKPOINT"]).expanduser()
    assert ckpt.is_file(), f"UMA_CHECKPOINT not a file: {ckpt}"

    pytest.importorskip("fairchem.core")
    pytest.importorskip("torchdiffeq")

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    cfg["Datahub"]["data_path"] = str(FIXTURE)
    cfg["Modelhub"]["internal_FFs"]["FF_UMA_FLOW"]["build_params"]["checkpoint_path"] = str(
        ckpt
    )
    import torch

    cfg["Trainer"]["cuda"] = bool(torch.cuda.is_available())
    cfg["Trainer"]["num_workers"] = 1

    scratch = Path(os.environ.get("UMA_SMOKE_OUT", "/tmp/uma_flow_qs_train_smoke"))
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / f"flow_run_{os.getpid()}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    resolved = work / "train_uma_flow_qs_resolved.yaml"
    with open(resolved, "w") as f:
        yaml.safe_dump(cfg, f)

    out_dir = work / "out"
    out_dir.mkdir()

    from enerzyme.train import FFTrain

    try:
        trainer = FFTrain(config_path=str(resolved), out_dir=str(out_dir))
        assert trainer.trainer.generator_config.get("enabled") is True
        trainer.train_all()

        model_dirs = [
            p for p in out_dir.iterdir() if p.is_dir() and "uma_flow" in p.name.lower()
        ]
        assert model_dirs, f"no uma_flow output under {out_dir}: {list(out_dir.iterdir())}"
        ckpts = list(model_dirs[0].glob("model*.pth"))
        assert ckpts, f"expected model*.pth under {model_dirs[0]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
