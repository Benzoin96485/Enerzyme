"""Time inference (energy + forces) and one training step for every architecture.

Each architecture runs in its own subprocess so threads, caches and peak memory
do not leak between models. Results are written as JSON; compare two runs with
``--compare``.

    PYTHONPATH=. python scripts/bench_cores.py --out bench.json [--only a,b] [--compile]
    python scripts/bench_cores.py --compare base.json new.json [--threshold 0.03]
"""
import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import tempfile
import sys
import time

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
                  idx_i=receiver, idx_j=sender)
    batch = make_batch([sample], dtype=dtype)
    if periodic:
        batch["offsets"] = torch.as_tensor(shifts @ cell, dtype=dtype)
    return batch


def _worker(arch, reps, threads, use_compile, systems="default", precision="float32",
            device_name="auto", model_tree=None):
    import torch
    from core_harness import build, forward

    if model_tree:
        model_tree = os.path.realpath(model_tree)
        if not os.path.isfile(os.path.join(model_tree, "enerzyme", "models", "ff.py")):
            raise ValueError(f"no Enerzyme source tree at {model_tree!r}")
        sys.path.insert(0, model_tree)
    from enerzyme.models.ff import get_ff_core

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = getattr(torch, precision)
    model = build(arch).to(device=device, dtype=dtype)
    _, defaults, _ = get_ff_core(arch)
    cutoff = max(float(defaults.get("cutoff_sr") or 0), float(defaults.get("cutoff_lr") or 0), 6.0)
    if use_compile:
        model = torch.compile(model)
    result = {"params": sum(p.numel() for p in model.parameters()), "device": device.type,
              "precision": precision, "compile": use_compile}
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for name, sizes in SYSTEM_SETS[systems].items():
        batch = _batch(name, sizes, dtype, cutoff)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        entry = {}
        for mode in ("infer", "train"):
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
            entry[f"{mode}_ms"] = statistics.median(times)
            if device.type == "cuda":
                entry[f"{mode}_peak_mb"] = statistics.median(peaks)
        result[name] = entry
    result["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def run_all(archs, out_path, reps, threads, use_compile, timeout, systems="default",
            precision="float32", device_name="auto", model_tree=None):
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    here = os.path.dirname(os.path.abspath(__file__))
    for arch in archs:
        cmd = [sys.executable, os.path.join(here, "bench_cores.py"), "--worker", arch,
               "--reps", str(reps), "--threads", str(threads)]
        cmd += (["--compile"] if use_compile else []) + ["--systems", systems]
        cmd += ["--dtype", precision, "--device", device_name]
        if model_tree:
            cmd += ["--model-tree", model_tree]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            results[arch] = {"error": "timeout"}
        else:
            line = [l for l in proc.stdout.splitlines() if l.startswith("BENCH ")]
            start = proc.stderr.rfind("Traceback")
            if line:
                results[arch] = json.loads(line[-1][6:])
            elif proc.returncode < 0:
                results[arch] = {"error": f"killed by signal {-proc.returncode} (out of memory?)"}
            else:
                results[arch] = {"error": proc.stderr[start:][-1500:] if start >= 0 else proc.stderr[-800:]}
        print(arch, results[arch], flush=True)
        with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(os.path.abspath(out_path)),
                                         delete=False) as f:
            json.dump(results, f, indent=1, sort_keys=True)
        os.replace(f.name, out_path)
    return all("error" not in results[arch] for arch in archs)


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
        for field, default in (("device", None), ("device_name", None), ("precision", "float32"), ("compile", False)):
            if old.get(field, default) != current.get(field, default):
                raise ValueError(f"{arch}: different {field} in the two runs")
        measured = systems.intersection(old)
        if not measured or measured != systems.intersection(current):
            raise ValueError(f"{arch}: missing or mismatched benchmark systems")
        for system in sorted(measured):
            b, n = old[system], current[system]
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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
                                             args.dtype, args.device, args.model_tree)),
              flush=True)
        return
    if args.compare:
        worse = compare(*args.compare, args.threshold)
        sys.exit(1 if worse else 0)
    from core_harness import ARCHITECTURES

    archs = args.only.split(",") if args.only else ARCHITECTURES
    ok = run_all(archs, args.out, args.reps, args.threads, args.compile, args.timeout, args.systems,
                 args.dtype, args.device, args.model_tree)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
