"""Measure ordinary torch.compile against a pre-refactor snapshot on one checkout.

Run the same script, snapshot, precision and systems on both checkouts. The
first compiled warmup is compared with eager energy, force and every active
weight derivative before subsequent timed repetitions. Progress is persisted
after each mode so later compiler timeouts and device OOMs retain earlier data.

Use --main-fixture in place of --snapshot to run a trained main-branch
checkpoint. Save --eager-grad-out on each checkout, then compare the runs
with --compare-main BASE.json CANDIDATE.json --reference-grads BASE.pt
CANDIDATE.pt --out comparison.json.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from importlib import metadata

import numpy as np
import torch

from bench_cores import SYSTEM_SETS, _add_fixture_features, _batch, _build_fixture_model, _fixture_configuration
from core_harness import build, forward


def seed_run():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)


def step(model, batch, dtype, training):
    out = forward(model, batch, dtype, train=training)
    if training:
        (out["E"].square().sum() + out["Fa"].square().sum()).backward()
    return out


def snapshot_step(model, outputs, training):
    fields = {key: outputs[key].detach().cpu().clone() for key in ("E", "Fa")}
    grads = {
        name: p.grad.detach().cpu().clone()
        for name, p in model.named_parameters()
        if training and p.grad is not None
    }
    return fields, grads


def compare_step(actual, reference, dtype):
    fields, gradients = actual
    ref_fields, ref_gradients = reference
    same = gradients.keys() == ref_gradients.keys()
    result = {
        "passed": same,
        "active_parameters": len(gradients),
        "reference_active_parameters": len(ref_gradients),
    }
    for quantity, pairs in (
        ("energy", [("E", fields["E"], ref_fields["E"])]),
        ("force", [("Fa", fields["Fa"], ref_fields["Fa"])]),
        ("gradient", [(k, gradients[k], ref_gradients[k]) for k in sorted(gradients.keys() & ref_gradients.keys())]),
    ):
        atol, rtol = (1e-10, 1e-8) if dtype == torch.float64 else (3e-4, 3e-4)
        if quantity == "gradient" and dtype == torch.float32:
            atol, rtol = 7e-4, 1e-3
        error = 0.0
        for name, a, b in pairs:
            if a.shape != b.shape or a.dtype != b.dtype:
                result["passed"] = False
                result.setdefault("incompatible_tensors", {})[name] = {
                    "actual_shape": list(a.shape), "reference_shape": list(b.shape),
                    "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
                }
                error = None
                continue
            if a.numel():
                diff = (a - b).abs().max().item()
                error = max(error, diff) if np.isfinite(diff) and error is not None else None
            result["passed"] &= bool(
                torch.isfinite(a).all()
                and torch.isfinite(b).all()
                and torch.allclose(a, b, atol=atol, rtol=rtol)
            )
        result[quantity + "_max_abs"] = error
    if not same:
        result["different_gradient_names"] = sorted(gradients.keys() ^ ref_gradients.keys())
    return result


def compare_main_runs(paths, reference_paths):
    """Classify each compiled main-checkpoint mode and require native parity."""
    return compare_frozen_runs(paths, reference_paths, compiled_main=True)


def compare_frozen_runs(paths, reference_paths, *, compiled_main=False):
    """Compare matched measurements from the same fixture and physical GPU."""
    raw = [json.loads(path.read_text()) for path in paths]
    if len(raw[0]) != 1 or raw[0].keys() != raw[1].keys():
        raise ValueError("expected both files to contain the same single architecture")
    arch = next(iter(raw[0]))
    base, candidate = (result[arch] for result in raw)
    fields = ("fixture_kind", "snapshot", "snapshot_sha256", "config_sha256", "checkpoint_state", "torch",
              "tiled_atom_features", "compile", "device", "device_name", "device_uuid", "precision", "params",
              "reps", "threads", "requested_systems", "modes", "snapshot_revision", "cuda_environment")
    if not base.get("snapshot_sha256"):
        raise ValueError("expected a frozen fixture hash")
    if (compiled_main or base.get("fixture_kind") == "main_trained") and (
        base.get("fixture_kind") != "main_trained" or not base.get("config_sha256")
    ):
        raise ValueError("expected a trained main checkpoint with configuration and fixture hashes")
    for field in fields:
        if field == "params" and (field not in base or field not in candidate):
            # Construction may fail before a model can report its parameter count.
            continue
        if base.get(field) != candidate.get(field):
            raise ValueError(f"cannot compare different {field}")
    if (compiled_main and not base.get("compile")) or not base.get("torch"):
        raise ValueError("expected ordinary compiled measurements with a PyTorch version")
    if base.get("device") == "cuda" and not base.get("device_uuid"):
        raise ValueError("a paired GPU comparison requires a physical device UUID; on older PyTorch, set CUDA_VISIBLE_DEVICES to the full GPU UUID")

    refs = [torch.load(path, map_location="cpu", weights_only=True) if path.is_file() else {}
            for path in reference_paths]
    result = {key: value for key, value in base.items() if key in fields}
    result.update(architecture=arch, comparisons={}, counts={})
    result["source"] = {
        name: {key: row.get(key) for key in ("model_revision", "driver_revision", "driver_sha256", "torch",
                                             "slurm_job_id", "node", "started_unix_ns", "finished_unix_ns")}
        for name, row in (("baseline", base), ("candidate", candidate))
    }
    if all(side["started_unix_ns"] is not None for side in result["source"].values()):
        result["measurement_order"] = sorted(result["source"], key=lambda side: result["source"][side]["started_unix_ns"])
    if base.get("driver_sha256") != candidate.get("driver_sha256"):
        raise ValueError("cannot compare different benchmark drivers")
    dtype = getattr(torch, base["precision"])
    for system in base["requested_systems"]:
        result["comparisons"][system] = modes = {}
        for mode in base["modes"]:
            cell = modes[mode] = {}
            valid = []
            for side, run in (("baseline", base), ("candidate", candidate)):
                measurements = run.get(system, {})
                timing = measurements.get(mode + "_ms")
                peak = measurements.get(mode + "_peak_mb")
                numeric = run.get("numerical", {}).get(system, {}).get(mode)
                error = measurements.get(mode + "_error") or run.get("worker_error")
                cell[side] = {"ms": timing, "peak_MiB": peak, "numerical": numeric, "error": error,
                              "capture": run.get("compile_capture", {}).get(system, {}).get(mode)}
                valid.append(bool(numeric and numeric.get("passed") and timing is not None
                                  and math.isfinite(timing) and timing > 0 and
                                  (run["device"] != "cuda" or (peak is not None and math.isfinite(peak) and peak > 0))))
            gold, trial = (saved.get((system, mode)) for saved in refs)
            cross = None
            if gold is not None and trial is not None and not isinstance(gold, str) and not isinstance(trial, str):
                cross = compare_step(trial, gold, dtype)
            cell["cross_native"] = cross
            for side, value in (("baseline", gold), ("candidate", trial)):
                if isinstance(value, str):
                    cell.setdefault("native_reference_errors", {})[side] = value
            if not any(valid):
                status = "neither_validated"
            elif cross is None:
                status = "cross_native_unavailable"
            elif not cross["passed"]:
                status = "cross_native_mismatch"
            elif all(valid):
                status = "both_validated"
                cell["delta_percent"] = {
                    key: 100 * (cell["candidate"][key] / cell["baseline"][key] - 1)
                    for key in ("ms", "peak_MiB") if cell["baseline"][key] is not None
                }
            else:
                status = "baseline_only_validated" if valid[0] else "candidate_only_validated"
            cell["status"] = status
            result["counts"][status] = result["counts"].get(status, 0) + 1
    return result


def _cuda_provenance(device):
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    # Older torch builds omit the UUID. CUDA_VISIBLE_DEVICES supports full physical
    # UUIDs, which remain unambiguous even when the runtime remaps card ordinals.
    if uuid is None:
        names = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        index = torch.cuda.current_device()
        if index < len(names) and re.fullmatch(r"(?:GPU|MIG)-[0-9a-fA-F-]{36}", names[index].strip()):
            uuid = names[index].strip()
    environment = {"runtime": torch.version.cuda, "compute_capability": list(torch.cuda.get_device_capability(device)),
                   "total_memory_bytes": properties.total_memory}
    try:
        environment["triton_version"] = metadata.version("triton")
    except metadata.PackageNotFoundError:
        environment["triton_version"] = None
    # Every visible GPU uses the same installed CUDA driver.
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=False)
        versions = set(driver.stdout.splitlines())
        if driver.returncode == 0 and len(versions) == 1:
            environment["driver_version"] = versions.pop().strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    result = {"device_name": properties.name, "cuda_environment": environment}
    if uuid is not None:
        result["device_uuid"] = str(uuid)
    return result


def _git_revision(directory):
    try:
        git = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return git.stdout.strip() if git.returncode == 0 else None


def run(args):
    started = time.time_ns()
    model_tree = args.model_tree.resolve()
    sys.path.insert(0, str(model_tree))
    from enerzyme.models import ff

    if Path(ff.__file__).resolve() != (model_tree / "enerzyme/models/ff.py").resolve():
        raise ValueError(f"Expected to load Enerzyme from {model_tree}, loaded {ff.__file__}")

    torch.set_num_threads(args.threads)
    dtype, device = getattr(torch, args.precision), torch.device(args.device)
    main_fixture = getattr(args, "main_fixture", None)
    source = main_fixture or args.snapshot
    atom_features = {}
    driver_path = Path(__file__).resolve()
    provenance = {
        "model_revision": _git_revision(args.model_tree), "driver_revision": _git_revision(driver_path.parent),
        "driver_sha256": hashlib.sha256(driver_path.read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname(), "started_unix_ns": started,
    }
    if main_fixture:
        config, digest, weights, atom_features = _fixture_configuration(source)
        arch, params = config["architecture"], config["build_params"]
        provenance.update(fixture_kind="main_trained", config_sha256=digest,
                          checkpoint_state="model_state_dict", tiled_atom_features=sorted(atom_features))
    else:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        arch = payload["architecture"]
        _, params, _ = ff.get_ff_core(arch)
        provenance.update(fixture_kind="presnapshot", snapshot_revision=payload["commit"])
    cutoff = max(float(params.get("cutoff_sr") or 0), float(params.get("cutoff_lr") or 0), 6.0)

    def load_model():
        seed_run()
        if main_fixture:
            model, state, names = _build_fixture_model(config, weights, dtype)
            return model.to(device=device), state, names
        model = build(arch).float()
        state = payload["state_dict"]
        if model.state_dict().keys() != state.keys():
            sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "test"))
            from presnapshot_maps import remap_snapshot

            state = remap_snapshot(arch, state, payload["param_names"], model)
        original_names = {}
        if getattr(args, "eager_grad_out", None) is not None:
            source_names = {id(payload["state_dict"][key]): key for key in payload["param_names"]}
            original_names = {key: source_names[id(state[key])] for key, _ in model.named_parameters()}
            if set(original_names.values()) != set(payload["param_names"]):
                raise ValueError("missing original parameter names in frozen gradient comparison")
        model.load_state_dict(state, strict=True)
        return model.to(device=device, dtype=dtype), state, original_names

    def write():
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps({arch: result}, allow_nan=False))
        temp.replace(args.out)

    result = {
        "device": device.type,
        "torch": str(torch.__version__),
        "precision": args.precision,
        "compile": not getattr(args, "eager", False),
        "reps": args.reps,
        "threads": args.threads,
        "snapshot": source.name,
        "snapshot_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        **provenance,
        "compile_capture": {},
        "numerical": {},
        "requested_systems": args.systems,
        "modes": args.modes,
    }
    if device.type == "cuda":
        result.update(_cuda_provenance(device))
    write()
    try:
        model, initial_state, original_names = load_model()
    except Exception as exc:
        result["worker_error"] = f"{type(exc).__name__}: {exc}"[:2500]
        write()
        raise
    result["params"] = sum(p.numel() for p in model.parameters())
    write()
    host_batches = {
        name: _add_fixture_features(_batch(name, SYSTEM_SETS["acceptance"][name], dtype, cutoff), atom_features, dtype)
        for name in args.systems
    }

    def device_batch(name):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in host_batches[name].items()}

    def configure_system(net, name):
        if arch == "allscaip":
            getattr(getattr(net, "embed", None), "graph_params", net).pbc = name == "bulk_256"

    def flush():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def clean():
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    references = {}
    for name in args.systems:
        configure_system(model, name)
        for mode in args.modes:
            try:
                seed_run()
                model.load_state_dict(initial_state, strict=True)
                out = step(model, device_batch(name), dtype, mode == "train")
                references[name, mode] = snapshot_step(model, out, mode == "train")
                del out
            except (RuntimeError, ValueError, TypeError) as exc:
                references[name, mode] = f"{type(exc).__name__}: {exc}"[:600]
            finally:
                model.zero_grad(set_to_none=True)
    if args.eager_out is not None:
        torch.save(
            {key: value[0] for key, value in references.items() if not isinstance(value, str)}, args.eager_out
        )
    if getattr(args, "eager_grad_out", None) is not None:
        saved = {}
        for key, value in references.items():
            if isinstance(value, str):
                saved[key] = value
            else:
                fields, grads = value
                saved[key] = (fields, {original_names.get(name, name): v for name, v in grads.items()})
        torch.save(saved, args.eager_grad_out)
    del model, initial_state
    clean()
    model, initial_state, _ = load_model()
    runner = torch.compile(model) if result["compile"] else model
    from torch._dynamo.utils import counters

    for name in args.systems:
        configure_system(model, name)
        batch = device_batch(name)
        entry = result[name] = {}
        result["numerical"][name] = {}
        result["compile_capture"][name] = {}
        for mode in args.modes:
            stats = counters["stats"]
            old_graphs, old_calls = stats["unique_graphs"], stats["calls_captured"]
            times, peaks = [], []
            try:
                seed_run()
                model.load_state_dict(initial_state, strict=True)
                for i in range(args.reps + 2):
                    flush()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    out = step(runner, batch, dtype, mode == "train")
                    if i == 0:
                        ref = references[name, mode]
                        proof = (
                            {"reference_error": ref}
                            if isinstance(ref, str)
                            else compare_step(snapshot_step(model, out, mode == "train"), ref, dtype)
                        )
                        if main_fixture:
                            proof["output_grad_functions"] = {
                                key: type(out[key].grad_fn).__name__ for key in ("E", "Fa")
                            }
                        result["numerical"][name][mode] = proof
                    if mode == "train":
                        model.zero_grad(set_to_none=True)
                    flush()
                    if i >= 2:
                        times.append(1000 * (time.perf_counter() - start))
                        if device.type == "cuda":
                            peaks.append(torch.cuda.max_memory_allocated(device) / 2**20)
                    del out
                entry[mode + "_ms"] = statistics.median(times)
                if peaks:
                    entry[mode + "_peak_mb"] = statistics.median(peaks)
            except (RuntimeError, ValueError, TypeError) as exc:
                entry[mode + "_error"] = f"{type(exc).__name__}: {exc}"[:2500]
                model.zero_grad(set_to_none=True)
                clean()
            if result["compile"]:
                result["compile_capture"][name][mode] = {
                    "new_graphs": stats["unique_graphs"] - old_graphs,
                    "new_calls": stats["calls_captured"] - old_calls,
                }
            write()
            print(
                arch,
                args.precision,
                name,
                mode,
                entry.get(mode + "_ms", entry.get(mode + "_error", "")),
                flush=True,
            )
        del batch
    result["finished"] = True
    result["finished_unix_ns"] = time.time_ns()
    write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tree", type=Path)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--snapshot", type=Path, help="use a Phase 0 pre-refactor snapshot")
    sources.add_argument("--main-fixture", type=Path, help="use the layers, trained weights and external features from a main checkpoint fixture")
    sources.add_argument("--compare-main", nargs=2, type=Path, metavar=("BASE", "CANDIDATE"),
                         help="compare two measured main-checkpoint JSON files")
    parser.add_argument("--reference-grads", nargs=2, type=Path, metavar=("BASE", "CANDIDATE"),
                        help="native reference files saved by each run with --eager-grad-out")
    parser.add_argument("--precision", choices=("float32", "float64"))
    parser.add_argument(
        "--systems", nargs="+", choices=SYSTEM_SETS["acceptance"], default=list(SYSTEM_SETS["acceptance"])
    )
    parser.add_argument("--modes", nargs="+", choices=("infer", "train"), default=["infer", "train"])
    parser.add_argument("--eager", action="store_true", help="measure ordinary eager execution against the frozen native reference")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eager-out", type=Path, help="save native E and Fa for cross-checkout comparison")
    parser.add_argument("--eager-grad-out", type=Path, help="save native E, Fa and active gradients; main fixture parameters use their original names")
    args = parser.parse_args()
    if min(args.reps, args.threads) < 1:
        parser.error("repetitions and threads must be positive")
    if args.compare_main:
        if not args.reference_grads:
            parser.error("--compare-main requires the two --reference-grads paths")
        comparison = compare_main_runs(args.compare_main, args.reference_grads)
        args.out.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    else:
        if args.model_tree is None or args.precision is None:
            parser.error("measurement requires --model-tree and --precision")
        run(args)
