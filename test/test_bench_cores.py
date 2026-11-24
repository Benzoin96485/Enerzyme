"""The performance gate requires complete measurements for matching workloads."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_cores import compare


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
