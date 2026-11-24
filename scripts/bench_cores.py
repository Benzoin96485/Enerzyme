"""Time inference (energy + forces) and one training step for every architecture.

Each architecture runs in its own subprocess so threads, caches and peak memory
do not leak between models. Results are written as JSON; compare two runs with
``--compare``.

    PYTHONPATH=. python scripts/bench_cores.py --out bench.json [--only a,b] [--compile]
    PYTHONPATH=. python scripts/bench_cores.py --out normalized.json --compile-reference
    python scripts/bench_cores.py --out mace.json --fixture test/fixtures/main_legacy/mace_ff04.pt.gz
    python scripts/bench_cores.py --compare base.json new.json [--threshold 0.03]
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import resource
import statistics
import subprocess
import tempfile
import sys
import time
from functools import lru_cache
from pathlib import Path
from types import MethodType

# ``small`` and ``tiny`` are for Cores whose default systems exceed the memory
# of the machine; their keys differ so ``--compare`` never mixes sets.
SYSTEM_SETS = {
    "default": {"mol_batch": [16] * 8, "mol_large": [64]},
    "small": {"mol_batch_small": [16] * 2, "mol_large_small": [32]},
    "tiny": {"mol_batch_tiny": [8] * 4, "mol_large_tiny": [12]},
    "acceptance": {"mol_batch_32x20": [20] * 32, "bulk_256": [256], "large_2000": [2000]},
}


def _batch(name, sizes, dtype, cutoff):
    from core_harness import make_batch, random_molecules

    if name not in ("bulk_256", "large_2000"):
        return make_batch(random_molecules(11, sizes), dtype=dtype)

    import numpy as np
    import torch
    from ase.neighborlist import primitive_neighbor_list

    periodic = name == "bulk_256"
    dims = (8, 8, 4) if periodic else (10, 10, 20)
    rng = np.random.default_rng(11)
    positions = (np.indices(dims).reshape(3, -1).T + 0.5) * 2.6
    positions += rng.uniform(-0.25, 0.25, positions.shape)
    cell = np.diag(np.asarray(dims) * 2.6)
    receiver, sender, shifts = primitive_neighbor_list("ijS", [periodic] * 3, cell, positions, cutoff)
    n = len(positions)
    sample = dict(Ra=positions, Za=rng.choice([1, 6, 7, 8], n), N=n, Q=0.0, S=0.0,
                  idx_i=receiver, idx_j=sender, N_pair=len(receiver))
    batch = make_batch([sample], dtype=dtype)
    if periodic:
        batch["offsets"] = torch.as_tensor(shifts @ cell, dtype=dtype)
        batch["cell"] = torch.as_tensor(cell, dtype=dtype).unsqueeze(0)
    return batch


def _fixture_configuration(path):
    """Read the construction settings and the trained parameters from one main fixture."""
    import torch
    from torch.torch_version import TorchVersion

    with gzip.open(path, "rb") as archive, torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(io.BytesIO(archive.read()), map_location="cpu", weights_only=True)
    config = {key: payload[key] for key in ("architecture", "layers", "build_params")}
    config["architecture"] = config["architecture"].lower()
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return config, hashlib.sha256(encoded).hexdigest(), payload["checkpoint"]["model_state_dict"]


def _load_reference_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=None)
def _reference_forward(gather_nd):
    import torch

    root = Path(__file__).resolve().parents[1] / "enerzyme" / "models"
    forces = _load_reference_module(
        "enerzyme.models.layers._benchmark_compiled_grad", root / "layers" / "compiled_grad.py"
    )
    table = _load_reference_module("enerzyme.models._benchmark_table", root / "functional.py")
    torch.compiler.substitute_in_graph(gather_nd)(table.gather_nd)
    return forces.forward_with_coordinate_gradients


def compile_reference_model(model):
    """Compile the same functional force and tensor table gather on both checkouts.

    Each checkout retains its model, layer classes, parameters and eager path.
    Both compiler representations come from this benchmark driver's tree.
    """
    import torch
    from enerzyme.models.functional import gather_nd

    model.forward = MethodType(_reference_forward(gather_nd), model)
    return torch.compile(model, fullgraph=True)


def _worker(arch, reps, threads, use_compile, systems="default", precision="float32",
            device_name="auto", model_tree=None, compile_reference=False, fixture=None):
    import numpy as np
    import torch
    from core_harness import build, forward

    if model_tree:
        model_tree = os.path.realpath(model_tree)
        if not os.path.isfile(os.path.join(model_tree, "enerzyme", "models", "ff.py")):
            raise ValueError(f"no Enerzyme source tree at {model_tree!r}")
        sys.path.insert(0, model_tree)
    from enerzyme.models.ff import get_ff_core

    torch.set_num_threads(threads)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = getattr(torch, precision)
    metadata = {}
    if fixture is None:
        model = build(arch)
        _, build_params, _ = get_ff_core(arch)
    else:
        config, digest, state = _fixture_configuration(fixture)
        if config["architecture"] != arch.lower():
            raise ValueError(f"Fixture builds {config['architecture']!r}, worker requested {arch!r}")
        build_params = config["build_params"]
        model = build(arch, layer_params=config["layers"], build_params=build_params)
        model = model.to(dtype=dtype)
        if model.state_dict().keys() != state.keys():
            from enerzyme.models.compat.main_branch import convert_state_dict
            state, _ = convert_state_dict(arch, state)
        model.load_state_dict(state, strict=True)
        metadata = {"fixture": Path(fixture).name, "config_sha256": digest,
                    "fixture_sha256": hashlib.sha256(Path(fixture).read_bytes()).hexdigest()}
    model = model.to(device=device, dtype=dtype)
    cutoff = max(float(build_params.get("cutoff_sr") or 0), float(build_params.get("cutoff_lr") or 0), 6.0)
    # AllScAIP builds its own neighbors. Its native option selects periodic images.
    graph_options = None
    if arch == "allscaip":
        graph_options = getattr(getattr(model, "embed", None), "graph_params", model)
    if compile_reference:
        model = compile_reference_model(model)
    elif use_compile:
        model = torch.compile(model)
    result = {"params": sum(p.numel() for p in model.parameters()), "device": device.type,
              "system_set": systems, "threads": threads, "reps": reps,
              "precision": precision, "compile": use_compile or compile_reference,
              "compile_reference": compile_reference, **metadata}
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
    if result["compile"]:
        from torch._dynamo.utils import counters
        result["compile_capture"] = {}

    def captured():
        stats = counters["stats"]
        return stats["unique_graphs"], stats["calls_captured"]

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(batch, mode):
        times = []
        peaks = []
        for step in range(reps + 2):
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            out = forward(model, batch, dtype, train=(mode == "train"))
            if mode == "train":
                loss = out["E"].pow(2).sum() + out["Fa"].pow(2).sum()
                loss.backward()
                model.zero_grad(set_to_none=True)
                del loss
            sync()
            if step >= 2:
                times.append((time.perf_counter() - t) * 1e3)
                if device.type == "cuda":
                    peaks.append(torch.cuda.max_memory_allocated() / 2**20)
            del out
        metrics = {f"{mode}_ms": statistics.median(times)}
        if peaks:
            metrics[f"{mode}_peak_mb"] = statistics.median(peaks)
        return metrics

    for name, sizes in SYSTEM_SETS[systems].items():
        if graph_options is not None:
            graph_options.pbc = name == "bulk_256"
        batch = _batch(name, sizes, dtype, cutoff)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        entry = result[name] = {}
        for mode in ("infer", "train"):
            if result["compile"]:
                before_graphs, before_calls = captured()
            try:
                entry.update(measure(batch, mode))
            except (RuntimeError, ValueError, TypeError) as exc:
                entry[f"{mode}_error"] = f"{type(exc).__name__}: {exc}"[:2500]
            if f"{mode}_error" in entry:
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if result["compile"]:
                graphs, calls = captured()
                # A zero delta can also mean that a previous graph was reused.
                result["compile_capture"].setdefault(name, {})[mode] = {
                    "new_graphs": graphs - before_graphs, "new_calls": calls - before_calls,
                }
            print("BENCH_PROGRESS " + json.dumps(result), flush=True)
    result["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def run_all(archs, out_path, reps, threads, use_compile, timeout, systems="default",
            precision="float32", device_name="auto", model_tree=None, compile_reference=False, fixture=None):
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    here = os.path.dirname(os.path.abspath(__file__))
    for arch in archs:
        cmd = [sys.executable, os.path.join(here, "bench_cores.py"), "--worker", arch,
               "--reps", str(reps), "--threads", str(threads)]
        if compile_reference:
            cmd.append("--compile-reference")
        elif use_compile:
            cmd.append("--compile")
        cmd += ["--systems", systems]
        cmd += ["--dtype", precision, "--device", device_name]
        if fixture:
            cmd += ["--fixture", fixture]
        if model_tree:
            cmd += ["--model-tree", model_tree]
        error = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            error = "timeout"
        else:
            stdout = proc.stdout
            if proc.returncode < 0:
                error = f"killed by signal {-proc.returncode}"
            elif proc.returncode:
                error = proc.stderr.strip()[-1500:] or f"worker exited with code {proc.returncode}"
        results[arch] = {}
        completed = False
        for line in stdout.splitlines():
            label, _, payload = line.partition(" ")
            if label in ("BENCH_PROGRESS", "BENCH"):
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError:
                    if error:  # A killed worker can leave its final line truncated.
                        continue
                    raise
                results[arch] = record
                completed = label == "BENCH"
        if error or not completed:
            results[arch]["error"] = error or "worker ended without a final result"
        print(arch, results[arch], flush=True)
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(os.path.abspath(out_path)),
                                         delete=False) as f:
            json.dump(results, f, indent=1, sort_keys=True)
        os.replace(f.name, out_path)
    return all("error" not in results[arch] and all(
        not any(k.endswith("_error") for k in results[arch].get(system, {}))
        for system in SYSTEM_SETS[systems]) for arch in archs)


def compare(base_path, new_path, threshold):
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    worse = []
    systems = {name for group in SYSTEM_SETS.values() for name in group}
    archs = sorted((set(base) | set(new)) - {"_meta"})
    if not archs:
        raise ValueError("no architectures to compare")
    for arch in archs:
        old = base.get(arch, {})
        current = new.get(arch, {})
        if "error" in old or "error" in current:
            raise ValueError(f"{arch}: incomplete benchmark: {old.get('error') or current.get('error')}")
        for field, default in (("params", None), ("config_sha256", None), ("fixture_sha256", None), ("device", None),
                               ("device_name", None), ("precision", "float32"), ("compile", False),
                               ("compile_reference", False)):
            if old.get(field, default) != current.get(field, default):
                raise ValueError(f"{arch}: different {field} in the two runs")
        for field in ("system_set", "threads", "reps"):
            if field in old and field in current and old[field] != current[field]:
                raise ValueError(f"{arch}: different {field} in the two runs")
        measured = systems.intersection(old)
        if not measured or measured != systems.intersection(current):
            raise ValueError(f"{arch}: missing or mismatched benchmark systems")
        system_set = old.get("system_set", current.get("system_set"))
        if system_set is not None and measured != set(SYSTEM_SETS[system_set]):
            raise ValueError(f"{arch}: incomplete {system_set} benchmark")
        for system in sorted(measured):
            b, n = old[system], current[system]
            for side, record in (("baseline", b), ("candidate", n)):
                for key in ("infer_error", "train_error"):
                    if key in record:
                        raise ValueError(f"{arch}/{system}/{side}/{key}: {record[key]}")
            required = {"infer_ms", "train_ms"}
            if old.get("device") == "cuda":
                required.update(("infer_peak_mb", "train_peak_mb"))
            if not required.issubset(b) or set(b) != set(n):
                raise ValueError(f"{arch}/{system}: missing or mismatched metrics")
            for key in b:
                if not all(math.isfinite(v) and v > 0 for v in (b[key], n[key])):
                    raise ValueError(f"{arch}/{system}/{key}: invalid measurement")
                ratio = n[key] / b[key] - 1.0
                flag = "SLOWER" if ratio > threshold else ""
                print(f"{arch:14s} {system:10s} {key:14s} {b[key]:10.2f} -> {n[key]:10.2f} {ratio:+7.1%} {flag}")
                if flag:
                    worse.append((arch, system, key, ratio))
    return worse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--only", default=None)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--compile", action="store_true", help="compile the selected checkout directly")
    modes.add_argument("--compile-reference", action="store_true",
                       help="use the same full-graph force derivative and tensor table gather for both checkouts")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fixture", help="load the settings and trained weights from a main_legacy .pt.gz fixture")
    parser.add_argument("--model-tree", help="load Enerzyme from this checkout using the same benchmark driver")
    parser.add_argument("--systems", choices=sorted(SYSTEM_SETS), default="default")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--compare", nargs=2, default=None)
    parser.add_argument("--threshold", type=float, default=0.03)
    args = parser.parse_args()
    if min(args.reps, args.threads, args.timeout) < 1:
        parser.error("--reps, --threads and --timeout must be positive")
    if args.worker:
        print("BENCH " + json.dumps(_worker(args.worker, args.reps, args.threads, args.compile, args.systems,
                                             args.dtype, args.device, args.model_tree, args.compile_reference, args.fixture)),
              flush=True)
        return
    if args.compare:
        worse = compare(*args.compare, args.threshold)
        sys.exit(1 if worse else 0)
    from core_harness import ARCHITECTURES

    if args.fixture:
        config, _, _ = _fixture_configuration(args.fixture)
        archs = [config["architecture"]]
        if args.only and args.only.lower().split(",") != archs:
            parser.error(f"--fixture builds {archs[0]!r}, incompatible with --only {args.only!r}")
    else:
        archs = args.only.split(",") if args.only else ARCHITECTURES
    ok = run_all(archs, args.out, args.reps, args.threads, args.compile, args.timeout, args.systems,
                 args.dtype, args.device, args.model_tree, args.compile_reference, args.fixture)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
