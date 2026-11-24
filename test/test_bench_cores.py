"""The performance gate requires complete measurements for matching workloads."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_cores import _add_fixture_features, _build_fixture_model, _fixture_configuration, compare, run_all


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
    config, digest, state, _ = _fixture_configuration(fixture)
    assert config["architecture"] == "mace"
    assert len(config["layers"]) > 1
    assert config["build_params"]["cutoff_sr"] > 0
    assert len(digest) == 64
    assert "interactions.0.linear_up.weight" in state


def test_fixture_includes_external_atomic_embeddings():
    import torch

    fixture = Path(__file__).parent / "fixtures" / "main_legacy" / "physnet_ff01.pt.gz"
    _, _, _, features = _fixture_configuration(fixture)
    assert set(features) == {"Qa_xTB"}
    assert features["Qa_xTB"].shape == (21,)
    batch = _add_fixture_features({"Za": torch.ones(44)}, features, torch.float64)
    torch.testing.assert_close(batch["Qa_xTB"], features["Qa_xTB"].repeat(3)[:44].double())
    explicit = torch.zeros(44)
    batch["Qa_xTB"] = explicit
    assert _add_fixture_features(batch, features, torch.float32)["Qa_xTB"] is explicit


@pytest.mark.parametrize("name", ["schnet_default", "physnet_ff01", "spookynet_ff02", "mace_ff04"])
def test_trained_benchmark_keeps_original_parameter_names_and_weights(name):
    import torch

    fixture = Path(__file__).parent / "fixtures" / "main_legacy" / f"{name}.pt.gz"
    config, _, state, _ = _fixture_configuration(fixture)
    model, loaded_state, names = _build_fixture_model(config, state, torch.float64)
    assert names.keys() == dict(model.named_parameters()).keys()
    assert len(set(names.values())) == len(names)
    assert loaded_state.keys() == model.state_dict().keys()
    for key, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, state[names[key]].double(), rtol=0, atol=0)


def test_compare_rejects_different_fixture_configs_at_same_parameter_count(tmp_path):
    paths = []
    for label in ("left", "right"):
        path = tmp_path / f"{label}.json"
        record = dict(_timings(), params=7, config_sha256=label)
        path.write_text(json.dumps({"mace": record}))
        paths.append(path)
    with pytest.raises(ValueError, match="config_sha256"):
        compare(*paths, 0.03)


def test_compiled_checkpoint_comparison_requires_cross_branch_weight_gradients(tmp_path):
    import torch
    from bench_frozen_compile import compare_main_runs

    measurement = {
        "fixture_kind": "main_trained", "snapshot": "fixture.pt.gz", "snapshot_sha256": "fixture",
        "config_sha256": "config", "compile": True, "torch": "2.13", "precision": "float32",
        "device": "cuda", "device_uuid": "test-card", "params": 1,
        "requested_systems": ["large_2000"], "modes": ["train"], "reps": 5,
        "large_2000": {"train_ms": 10.0, "train_peak_mb": 64.0},
        "numerical": {"large_2000": {"train": {"passed": True}}},
    }
    paths = [tmp_path / (side + ".json") for side in ("base", "new")]
    refs = [tmp_path / (side + ".pt") for side in ("base", "new")]
    values = {"E": torch.tensor([1.0]), "Fa": torch.zeros(1, 3)}
    for path, ref in zip(paths, refs):
        path.write_text(json.dumps({"schnet": measurement}))
        torch.save({("large_2000", "train"): (values, {"w": torch.ones(1)})}, ref)

    def cell():
        return compare_main_runs(paths, refs)["comparisons"]["large_2000"]["train"]

    assert cell()["delta_percent"] == {"ms": 0.0, "peak_MiB": 0.0}
    paths[0].write_text(json.dumps({"schnet": dict(measurement, large_2000={"train_error": "compiler error"})}))
    assert cell()["status"] == "candidate_only_validated"
    assert "delta_percent" not in cell()
    torch.save({("large_2000", "train"): (values, {"other": torch.ones(1)})}, refs[1])
    mismatch = cell()
    assert mismatch["status"] == "cross_native_mismatch"
    assert mismatch["cross_native"]["different_gradient_names"] == ["other", "w"]
    # Equal values must not pass through allclose's broadcasting or dtype conversions.
    torch.save({("large_2000", "train"): (dict(values, E=torch.ones(1, 1)), {"w": torch.ones(1).double()})}, refs[1])
    incompatible = cell()["cross_native"]
    assert not incompatible["passed"]
    assert set(incompatible["incompatible_tensors"]) == {"E", "w"}
    torch.save({("large_2000", "train"): "missing external field"}, refs[1])
    missing = cell()
    assert missing["status"] == "cross_native_unavailable"
    assert missing["native_reference_errors"]["candidate"] == "missing external field"
    paths[1].write_text(json.dumps({"schnet": dict(measurement, torch="other")}))
    with pytest.raises(ValueError, match="different torch"):
        cell()
    paths[1].write_text(json.dumps({"schnet": dict(measurement, device_uuid="second-card")}))
    with pytest.raises(ValueError, match="different device_uuid"):
        cell()


def test_worker_failure_preserves_completed_measurements(tmp_path, monkeypatch):
    output = 'BENCH_PROGRESS {"bulk_256":{"infer_ms":1.2}}\nBENCH_PROGRESS {"bulk_'

    def run(*args, **kwargs):
        return subprocess.CompletedProcess([], -9, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    path = tmp_path / "partial.json"
    assert not run_all(["schnet"], path, 5, 2, False, timeout=60, systems="acceptance")
    result = json.loads(path.read_text())["schnet"]
    assert result["bulk_256"] == {"infer_ms": 1.2}
    assert result["error"] == "killed by signal 9"


def test_compare_checks_declared_workload_against_historical_measurements(tmp_path):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps({"schnet": _timings()}))
    new.write_text(json.dumps({"schnet": dict(_timings(), system_set="default")}))
    with pytest.raises(ValueError, match="incomplete default"):
        compare(old, new, 0.03)
