"""OTF multi-molecule batch smoke for SpookyNet / PhysNet / SchNet.

These architectures consume Enerzyme ``idx_i`` / ``idx_j`` via RangeSeparation.
Unlike uma_qs (fairchem builds its own graph), a broken ``_decorate_batch_input``
OTF path with ``batch_size>1`` corrupts their message passing.
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "L3-COMT-aselmdb-smoke"
FIXTURE = EXAMPLE / "fixtures" / "molecules_otf_tiny.pkl"

ARCH_CFGS = {
    "SpookyNet": EXAMPLE / "config" / "train_otf_batch_spookynet.yaml",
    "PhysNet": EXAMPLE / "config" / "train_otf_batch_physnet.yaml",
    "SchNet": EXAMPLE / "config" / "train_otf_batch_schnet.yaml",
}


@pytest.mark.parametrize("architecture", list(ARCH_CFGS))
def test_otf_batch_config_shape(architecture: str):
    cfg_path = ARCH_CFGS[architecture]
    assert cfg_path.is_file()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["Datahub"]["neighbor_list"] in ("", None)
    assert cfg["Datahub"]["data_path"] == "fixtures/molecules_otf_tiny.pkl"
    assert cfg["Trainer"]["batch_size"] > 1
    assert cfg["Trainer"].get("otf_graph", True) is True
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    assert ff["architecture"] == architecture
    assert FIXTURE.is_file()


def _feature_from_frame(frame: dict, with_spin: bool = False) -> dict:
    from enerzyme.data.transform import parse_Za

    za = parse_Za(frame["atom_type"])
    feat = {
        "N": int(len(za)),
        "Za": np.asarray(za, dtype=int),
        "Ra": np.asarray(frame["coord"], dtype=float),
        "Q": int(frame["total_chrg"]),
    }
    if with_spin:
        feat["S"] = int(frame["total_spin"])
    return feat


def _assert_full_batch_graph(batch_features: dict, sizes: list[int]) -> None:
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
        assert np.all((idx_j[mask] >= offset) & (idx_j[mask] < offset + n))
        offset += n
    assert offset == sum(sizes)
    assert set(idx_i.tolist()) | set(idx_j.tolist()) == set(range(sum(sizes)))


def _load_cfg(architecture: str) -> dict:
    with open(ARCH_CFGS[architecture]) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("architecture", list(ARCH_CFGS))
def test_otf_collate_and_forward_finite(architecture: str):
    torch = pytest.importorskip("torch")
    from enerzyme.models.ff import build_model
    from enerzyme.tasks.batch import _decorate_batch_input

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    assert len(frames) >= 2
    with_spin = architecture == "SpookyNet"
    features = [_feature_from_frame(frame, with_spin=with_spin) for frame in frames[:2]]
    sizes = [feat["N"] for feat in features]
    assert sizes[0] != sizes[1]

    batch = [
        (features[0], {"E": frames[0]["energy"], "Fa": frames[0]["grad"]}, 0),
        (features[1], {"E": frames[1]["energy"], "Fa": frames[1]["grad"]}, 1),
    ]
    # Match SpookyNet/PhysNet float64 buffers (electron_config / RBF params)
    dtype = torch.float64
    batch_features, _ = _decorate_batch_input(
        batch, dtype=dtype, device=torch.device("cpu"), otf_graph=True
    )
    _assert_full_batch_graph(batch_features, sizes)

    cfg = _load_cfg(architecture)
    ff = next(iter(cfg["Modelhub"]["internal_FFs"].values()))
    model = build_model(
        architecture=ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    model = model.to(dtype=dtype)
    model.eval()
    # Force needs grad-enabled Ra
    batch_features["Ra"] = batch_features["Ra"].detach().requires_grad_(True)
    output = model(batch_features)
    assert "E" in output and "Fa" in output
    assert torch.isfinite(output["E"]).all()
    assert torch.isfinite(output["Fa"]).all()
    assert output["E"].shape[0] == 2
    assert output["Fa"].shape[0] == sum(sizes)


@pytest.mark.parametrize("architecture", list(ARCH_CFGS))
def test_otf_one_epoch_fftrain(architecture: str, tmp_path: Path):
    pytest.importorskip("torch")
    from enerzyme.train import FFTrain

    cfg = _load_cfg(architecture)
    cfg["Datahub"]["data_path"] = str(FIXTURE)
    cfg["Trainer"]["cuda"] = False
    cfg["Trainer"]["num_workers"] = 1

    work = tmp_path / f"otf_{architecture.lower()}"
    work.mkdir()
    resolved = work / "train_resolved.yaml"
    with open(resolved, "w") as f:
        yaml.safe_dump(cfg, f)
    out_dir = work / "out"
    out_dir.mkdir()

    with open(FIXTURE, "rb") as f:
        frames = pickle.load(f)
    with_spin = architecture == "SpookyNet"

    try:
        trainer = FFTrain(config_path=str(resolved), out_dir=str(out_dir))
        assert trainer.trainer.batch_size > 1
        assert trainer.trainer.otf_graph is True

        collated, _ = trainer.trainer.decorate_batch_input(
            [
                (
                    _feature_from_frame(frames[0], with_spin=with_spin),
                    {"E": frames[0]["energy"], "Fa": frames[0]["grad"]},
                    0,
                ),
                (
                    _feature_from_frame(frames[1], with_spin=with_spin),
                    {"E": frames[1]["energy"], "Fa": frames[1]["grad"]},
                    1,
                ),
            ]
        )
        _assert_full_batch_graph(
            collated,
            [len(frames[0]["atom_type"]), len(frames[1]["atom_type"])],
        )

        trainer.train_all()

        arch_token = architecture.lower()
        model_dirs = [
            p
            for p in out_dir.iterdir()
            if p.is_dir() and arch_token in p.name.lower()
        ]
        assert model_dirs, f"no {architecture} output under {out_dir}: {list(out_dir.iterdir())}"
        ckpts = list(model_dirs[0].glob("model*.pth"))
        assert ckpts, f"expected model*.pth under {model_dirs[0]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)
