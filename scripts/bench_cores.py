"""Time inference (energy + forces) and one training step for every architecture.

Each architecture runs in its own subprocess so threads, caches and peak memory
do not leak between models. Results are written as JSON; compare two runs with
``--compare``.

    PYTHONPATH=. python scripts/bench_cores.py --out bench.json [--only a,b] [--compile]
    python scripts/bench_cores.py --compare base.json new.json [--threshold 0.03]
"""
import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time

SYSTEMS = {"mol_batch": [16] * 8, "mol_large": [64]}


def _worker(arch: str, reps: int, threads: int, use_compile: bool) -> dict:
    import torch
    from core_harness import build, forward, make_batch, random_molecules

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(arch).float().to(device)
    if use_compile:
        model = torch.compile(model)
    result = {"params": sum(p.numel() for p in model.parameters()), "device": device.type}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for name, sizes in SYSTEMS.items():
        batch = make_batch(random_molecules(11, sizes), dtype=torch.float32)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        entry = {}
        for mode in ("infer", "train"):
            times = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for step in range(reps + 2):
                sync()
                t = time.perf_counter()
                out = forward(model, batch, torch.float32, train=(mode == "train"))
                if mode == "train":
                    loss = out["E"].pow(2).sum() + out["Fa"].pow(2).sum()
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                sync()
                if step >= 2:
                    times.append((time.perf_counter() - t) * 1e3)
            entry[f"{mode}_ms"] = statistics.median(times)
            if device.type == "cuda":
                entry[f"{mode}_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        result[name] = entry
    result["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def run_all(archs, out_path, reps, threads, use_compile, timeout):
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    here = os.path.dirname(os.path.abspath(__file__))
    for arch in archs:
        cmd = [sys.executable, os.path.join(here, "bench_cores.py"), "--worker", arch,
               "--reps", str(reps), "--threads", str(threads)] + (["--compile"] if use_compile else [])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            line = [l for l in proc.stdout.splitlines() if l.startswith("BENCH ")]
            results[arch] = json.loads(line[-1][6:]) if line else {"error": proc.stderr[-800:]}
        except subprocess.TimeoutExpired:
            results[arch] = {"error": "timeout"}
        print(arch, results[arch], flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)


def compare(base_path, new_path, threshold):
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)
    worse = []
    for arch in sorted(set(base) & set(new)):
        for system in SYSTEMS:
            b, n = base[arch].get(system), new[arch].get(system)
            if not b or not n:
                continue
            for key in b:
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
    parser.add_argument("--worker", default=None)
    parser.add_argument("--compare", nargs=2, default=None)
    parser.add_argument("--threshold", type=float, default=0.03)
    args = parser.parse_args()
    if args.worker:
        print("BENCH " + json.dumps(_worker(args.worker, args.reps, args.threads, args.compile)), flush=True)
        return
    if args.compare:
        worse = compare(*args.compare, args.threshold)
        sys.exit(1 if worse else 0)
    from core_harness import ARCHITECTURES

    archs = args.only.split(",") if args.only else ARCHITECTURES
    run_all(archs, args.out, args.reps, args.threads, args.compile, args.timeout)


if __name__ == "__main__":
    main()
