"""The performance gate requires complete measurements for matching workloads."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_cores import _fixture_configuration, compare


def _timings():
    return {"device": "cpu", "mol_batch": {"infer_ms": 10.0, "train_ms": 20.0}}


@pytest.mark.parametrize("entry", [
    {}, {"error": "timeout"},
    {"device": "cpu", "mol_large": {"infer_ms": 10.0, "train_ms": 20.0}},
    {"device": "cpu", "mol_batch": {"infer_ms": 10.0}},
    {"device": "cpu", "mol_batch": {"infer_ms": float("nan"), "train_ms": 20.0}},
    dict(_timings(), precision="float64"), dict(_timings(), compile=True),
])
def test_compare_rejects_incomplete_or_different_workloads(tmp_path, entry):
    old, new = tmp_path / "before.json", tmp_path / "after.json"
    old.write_text(json.dumps({"schnet": _timings()}))
    new.write_text(json.dumps({"schnet": entry}))
    with pytest.raises(ValueError):
        compare(old, new, 0.03)


def test_compare_checks_each_measurement(tmp_path):
    old, new = tmp_path / "before.json", tmp_path / "after.json"
    old.write_text(json.dumps({"schnet": _timings()}))
    new.write_text(old.read_text())
    assert compare(old, new, 0.03) == []
    entry = _timings()
    entry["mol_batch"]["train_ms"] = 20.8
    new.write_text(json.dumps({"schnet": entry}))
    failures = compare(old, new, 0.03)
    assert [row[:3] for row in failures] == [("schnet", "mol_batch", "train_ms")]


def test_compare_distinguishes_native_and_reference_compilation(tmp_path):
    old, new = tmp_path / "before.json", tmp_path / "after.json"
    record = dict(_timings(), compile=True)
    old.write_text(json.dumps({"schnet": record}))
    record["compile_reference"] = True
    new.write_text(json.dumps({"schnet": record}))
    with pytest.raises(ValueError, match="compile_reference"):
        compare(old, new, 0.03)
    old.write_text(new.read_text())
    assert compare(old, new, 0.03) == []


@pytest.mark.parametrize("before, after", [(17, 18), (17, None), (None, 17)])
def test_compare_requires_matching_parameter_counts(tmp_path, before, after):
    paths = []
    for label, params in (("before", before), ("after", after)):
        path = tmp_path / f"{label}.json"
        record = _timings()
        if params is not None:
            record["params"] = params
        path.write_text(json.dumps({"schnet": record}))
        paths.append(path)
    with pytest.raises(ValueError, match="params"):
        compare(*paths, 0.03)


def test_fixture_retains_shared_architecture_configuration():
    fixture = Path(__file__).parent / "fixtures" / "main_legacy" / "mace_ff04.pt.gz"
    config, digest = _fixture_configuration(fixture)
    assert config["architecture"] == "mace"
    assert len(config["layers"]) > 1
    assert config["build_params"]["cutoff_sr"] > 0
    assert len(digest) == 64


def test_compare_rejects_different_fixture_configs_at_same_parameter_count(tmp_path):
    paths = []
    for label in ("left", "right"):
        path = tmp_path / f"{label}.json"
        record = dict(_timings(), params=7, config_sha256=label)
        path.write_text(json.dumps({"mace": record}))
        paths.append(path)
    with pytest.raises(ValueError, match="config_sha256"):
        compare(*paths, 0.03)
