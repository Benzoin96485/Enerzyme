"""Smoke for uma_qs training config + optional 1-epoch live train (PR #81).

Config shape checks always run. Live training runs only when:
- ``UMA_CHECKPOINT`` points at a UMA ``.pt`` file, and
- ``fairchem`` imports successfully.

Host-absolute checkpoint / cache paths stay on the CLI / env — never in the
committed YAML (placeholder ``UMA_CHECKPOINT``).
"""
from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "L3-COMT-aselmdb-smoke"
CFG = EXAMPLE / "config" / "train_uma_qs.yaml"
FIXTURE = EXAMPLE / "fixtures" / "fragments_tiny.pkl"
PLACEHOLDER = "UMA_CHECKPOINT"


def test_train_uma_qs_config_shape():
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    assert cfg["Datahub"]["data_format"] == "pickle"
    assert cfg["Datahub"]["data_path"] == "fixtures/fragments_tiny.pkl"
    assert cfg["Datahub"]["neighbor_list"] in ("", None)
    assert cfg["Trainer"]["batch_size"] > 1
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
    assert FIXTURE.is_file()


def _feature_from_frame(frame: dict) -> dict:
    from enerzyme.data.transform import parse_Za

    za = parse_Za(frame["atom_type"])
    ra = np.asarray(frame["coord"], dtype=float)
    return {
        "N": int(len(za)),
        "Za": np.asarray(za, dtype=int),
        "Ra": ra,
        "Q": int(frame["total_chrg"]),
        "S": int(frame["total_spin"]),
    }


def _assert_full_batch_graph(batch_features: dict, sizes: list[int]) -> None:
    """Every atom in every structure must participate in a full neighbor graph."""
    assert "idx_i" in batch_features and "idx_j" in batch_features
    idx_i = batch_features["idx_i"].detach().cpu().numpy()
    idx_j = batch_features["idx_j"].detach().cpu().numpy()
    expected_edges = sum(n * (n - 1) for n in sizes)
    assert len(idx_i) == expected_edges == len(idx_j)

    offset = 0
    for n in sizes:
        atom_ids = set(range(offset, offset + n))
        mask = (idx_i >= offset) & (idx_i < offset + n)
        assert mask.sum() == n * (n - 1)
        assert set(idx_i[mask].tolist()) == atom_ids
        assert set(idx_j[mask].tolist()) == atom_ids
        # No cross-molecule edges
        assert np.all((idx_j[mask] >= offset) & (idx_j[mask] < offset + n))
        offset += n
    assert offset == sum(sizes)
    assert set(idx_i.tolist()) | set(idx_j.tolist()) == set(range(sum(sizes)))


def test_decorate_batch_input_otf_graph_covers_all_uma_fixture_structures():
    """Regression: otf_graph must build idx_i/idx_j for *every* molecule in a batch.

    uma_qs smoke uses neighbor_list='' (OTF) and batch_size>1; a batch-wide
    ``built_graph`` flag previously stopped after the first structure.
    """
    torch = pytest.importorskip("torch")
    from enerzyme.data.neighbor_list import full_neighbor_list
    from enerzyme.tasks.batch import _decorate_batch_input

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    assert len(frames) >= 2

    features = [_feature_from_frame(frame) for frame in frames[:2]]
    sizes = [feat["N"] for feat in features]
    batch = [
        (features[0], {"Q": features[0]["Q"], "S": features[0]["S"]}, 0),
        (features[1], {"Q": features[1]["Q"], "S": features[1]["S"]}, 1),
    ]
    batch_features, _ = _decorate_batch_input(
        batch, dtype=torch.float32, device=torch.device("cpu"), otf_graph=True
    )
    _assert_full_batch_graph(batch_features, sizes)

    # Mixed: precomputed edges on the first structure, OTF on the second
    idx_i, idx_j = full_neighbor_list(features[0]["N"])
    features[0] = {
        **features[0],
        "idx_i": idx_i,
        "idx_j": idx_j,
        "N_pair": len(idx_i),
    }
    batch_mixed = [
        (features[0], {"Q": features[0]["Q"], "S": features[0]["S"]}, 0),
        (features[1], {"Q": features[1]["Q"], "S": features[1]["S"]}, 1),
    ]
    mixed_features, _ = _decorate_batch_input(
        batch_mixed, dtype=torch.float32, device=torch.device("cpu"), otf_graph=True
    )
    _assert_full_batch_graph(mixed_features, sizes)


def test_decorate_batch_input_rejects_incomplete_neighbor_lists():
    torch = pytest.importorskip("torch")
    from enerzyme.data.neighbor_list import full_neighbor_list
    from enerzyme.tasks.batch import _decorate_batch_input

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    feat0 = _feature_from_frame(frames[0])
    feat1 = _feature_from_frame(frames[1])
    idx_i, idx_j = full_neighbor_list(feat0["N"])
    feat0.update({"idx_i": idx_i, "idx_j": idx_j, "N_pair": len(idx_i)})
    batch = [
        (feat0, {"Q": feat0["Q"], "S": feat0["S"]}, 0),
        (feat1, {"Q": feat1["Q"], "S": feat1["S"]}, 1),
    ]
    with pytest.raises(ValueError, match="Incomplete neighbor lists"):
        _decorate_batch_input(
            batch, dtype=torch.float32, device=torch.device("cpu"), otf_graph=False
        )


def test_decorate_pyg_batch_input_keeps_samples_without_edges():
    """Regression: otf_graph=False with omitted edges must still batch every sample.

    UMA-style cores build the graph internally; dropping samples from feature_list
    desyncs PyG features from targets.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from enerzyme.tasks.batch import _decorate_pyg_batch_input

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    features = [_feature_from_frame(frame) for frame in frames[:2]]
    sizes = [feat["N"] for feat in features]
    batch = [
        (features[0], {"Q": features[0]["Q"], "S": features[0]["S"]}, 0),
        (features[1], {"Q": features[1]["Q"], "S": features[1]["S"]}, 1),
    ]
    batch_features, batch_targets = _decorate_pyg_batch_input(
        batch, dtype=torch.float32, device=torch.device("cpu"), otf_graph=False
    )
    assert batch_features.num_graphs == len(features) == batch_targets.num_graphs
    assert batch_features.edge_index.numel() == 0
    assert batch_features.N.tolist() == sizes
    assert int(batch_features.Za.numel()) == sum(sizes)


def test_decorate_pyg_batch_input_rejects_incomplete_neighbor_lists():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from enerzyme.data.neighbor_list import full_neighbor_list
    from enerzyme.tasks.batch import _decorate_pyg_batch_input

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    feat0 = _feature_from_frame(frames[0])
    feat1 = _feature_from_frame(frames[1])
    idx_i, idx_j = full_neighbor_list(feat0["N"])
    feat0.update({"idx_i": idx_i, "idx_j": idx_j, "N_pair": len(idx_i)})
    batch = [
        (feat0, {"Q": feat0["Q"], "S": feat0["S"]}, 0),
        (feat1, {"Q": feat1["Q"], "S": feat1["S"]}, 1),
    ]
    with pytest.raises(ValueError, match="Incomplete neighbor lists"):
        _decorate_pyg_batch_input(
            batch, dtype=torch.float32, device=torch.device("cpu"), otf_graph=False
        )


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
    assert cfg["Trainer"]["batch_size"] > 1
    assert not cfg["Datahub"].get("neighbor_list")

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
        assert trainer.trainer.batch_size > 1
        assert trainer.trainer.otf_graph is True

        # Collate the two training frames before the epoch so a broken OTF batch
        # graph fails fast (even if the backbone rebuilds edges internally).
        with open(FIXTURE, "rb") as f:
            frames = pickle.load(f)
        collated, _ = trainer.trainer.decorate_batch_input(
            [
                (_feature_from_frame(frames[0]), {"Q": frames[0]["total_chrg"], "S": frames[0]["total_spin"]}, 0),
                (_feature_from_frame(frames[1]), {"Q": frames[1]["total_chrg"], "S": frames[1]["total_spin"]}, 1),
            ]
        )
        _assert_full_batch_graph(
            collated,
            [len(frames[0]["atom_type"]), len(frames[1]["atom_type"])],
        )

        trainer.train_all()

        model_dirs = [
            p for p in out_dir.iterdir() if p.is_dir() and "uma_qs" in p.name.lower()
        ]
        assert model_dirs, f"no uma_qs output under {out_dir}: {list(out_dir.iterdir())}"
        ckpts = list(model_dirs[0].glob("model*.pth"))
        assert ckpts, f"expected model*.pth under {model_dirs[0]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
