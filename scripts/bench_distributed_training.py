"""Measure supervised force training through Trainer.fit_predict on a fixed global batch.

Run one process per GPU using srun or torchrun. Use the same driver, weights,
global batch, optimizer and allocation for both source checkouts. Each rank
writes its own measurements so the slowest rank determines throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from statistics import median
import subprocess
import sys
import time


def compare_runs(directory):
    """Summarize complete ABBA runs, checking that their workloads and GPUs match."""
    report = {}
    for system in ("molecules", "crystals"):
        legs = {}
        identity = None
        revisions = {}
        for leg in ("A1", "B1", "B2", "A2"):
            ranks = sorted((directory / f"{system}-{leg}").glob("rank-*.json"))
            if not ranks:
                raise ValueError(f"Missing measurements for {system}-{leg}")
            rows = [json.loads(path.read_text()) for path in ranks]
            if {row["rank"] for row in rows} != set(range(rows[0]["world_size"])):
                raise ValueError(f"Incomplete rank set for {system}-{leg}")
            fields = ("fixture_sha256", "config_sha256", "system", "atoms", "edges",
                      "world_size", "global_batch", "warmup", "steps", "torch", "cuda")
            hardware = sorted((r["host"], r["device"], r["device_uuid"]) for r in rows)
            if len(set(hardware)) != len(rows):
                raise ValueError(f"Ranks share a GPU in {system}-{leg}")
            source = {(row["revision"], row.get("package_tree"), row["lightning"]) for row in rows}
            if len(source) != 1:
                raise ValueError(f"Ranks disagree about the trainer or checkout in {system}-{leg}")
            sha, package, trainer = source.pop()
            prior_sha, prior_package, prior_trainer = revisions.setdefault(leg[0], (sha, package, trainer))
            if prior_trainer != trainer or (prior_sha != sha and (not package or prior_package != package)):
                raise ValueError(f"Trainer or package changed in {system}-{leg}")
            trial = (tuple(rows[0][k] for k in fields), hardware)
            if any(tuple(row[k] for k in fields) != trial[0] for row in rows):
                raise ValueError(f"Ranks disagree about the workload in {system}-{leg}")
            if identity is not None and trial != identity:
                raise ValueError(f"Hardware or workload changed in {system}-{leg}")
            identity = trial
            legs[leg] = {key: max(row[key] for row in rows) for key in ("step_ms", "peak_bytes")}
        changes = {}
        for metric in ("step_ms", "peak_bytes"):
            before = median(legs[leg][metric] for leg in ("A1", "A2"))
            after = median(legs[leg][metric] for leg in ("B1", "B2"))
            changes[metric] = dict(baseline=before, candidate=after, change_percent=100*(after/before-1))
        report[system] = dict(legs=legs, comparison=changes)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path, help="directory with molecules-A1 through crystals-A2")
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--system", choices=["molecules", "crystals"])
    parser.add_argument("--lightning", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare_runs(args.compare), indent=2))
        return
    if not (args.source_tree and args.output and args.system):
        parser.error("measurement requires --source-tree, --output and --system")
    if min(args.warmup, args.steps) < 1:
        parser.error("warmup and measured steps must both be positive")
    tree = args.source_tree.resolve()
    revision = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    package_tree = subprocess.check_output(["git", "-C", str(tree), "rev-parse", revision + ":enerzyme"],
                                           text=True).strip()
    sys.path[:0] = [str(tree), str(tree / "scripts")]

    import numpy as np
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import enerzyme
    from bench_cores import _batch, _build_fixture_model, _fixture_configuration
    from enerzyme.models.loss import MSELoss
    from enerzyme.tasks.trainer import Trainer

    assert Path(enerzyme.__file__).resolve().parent.parent == tree
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_STEP_NUM_TASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    fixture = tree / "test/fixtures/main_legacy/schnet_default.pt.gz"
    config, config_digest, state, _ = _fixture_configuration(fixture)
    torch.manual_seed(11)
    model, _, _ = _build_fixture_model(config, state, torch.float32)
    periodic = args.system == "crystals"
    global_batch = 4 if periodic else 32
    if global_batch % world:
        parser.error("number of ranks must divide the global batch size")
    atoms = 256 if periodic else 20
    batch = _batch("bulk_256" if periodic else "molecule", [atoms], torch.float32, 4.5)
    features = {k: v.detach().cpu().numpy() for k, v in batch.items()
                if k not in ("N", "batch_seg")}
    features.update(N=atoms, S=0.0, Q=0.0, N_pair=len(features["idx_i"]))
    if periodic:
        features.update(cell=features["cell"][0], pbc=np.ones(3, dtype=bool))
    sample = (features, {"E": np.float32(0), "Fa": np.zeros((atoms, 3), np.float32)}, "benchmark")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(out_dir=str(out), cuda=True, lightning=args.lightning, num_workers=1,
                      use_ema=False, seed=11, dtype="float32", warmup_ratio=0,
                      learning_rate=1e-5, max_epochs=1, batch_size=global_batch // world,
                      Splitter={"parts": ["training", "validation"], "ratios": [0.8, 0.2]},
                      metric_config={"E": {"rmse": 1}, "Fa": {"rmse": 1}})
    steps = 0
    start = elapsed = peak = gpu = None

    def record_step(optimizer, _args, _kwargs):
        nonlocal steps, start, elapsed, peak, gpu
        steps += 1
        if steps == args.warmup:
            torch.cuda.synchronize()
            gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        if steps == args.warmup + args.steps:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()

    count = (args.warmup + args.steps) * global_batch
    with register_optimizer_step_post_hook(record_step):
        trainer.fit_predict(model, None, [sample] * count, [sample] * global_batch,
                            {"mse": MSELoss(E=1., Fa=1.)}, str(out), None)
    if steps != args.warmup + args.steps:
        raise RuntimeError(f"Observed {steps} updates, expected {args.warmup + args.steps}")
    record = dict(revision=revision, package_tree=package_tree, rank=rank, world_size=world, host=socket.gethostname(),
                  device=gpu.name, device_uuid=str(gpu.uuid), capability=[gpu.major, gpu.minor],
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  lightning=args.lightning, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  config_sha256=config_digest, system=args.system, atoms=atoms, edges=features["N_pair"],
                  global_batch=global_batch, warmup=args.warmup, steps=args.steps,
                  seconds=elapsed, step_ms=1000 * elapsed / args.steps, peak_bytes=peak)
    (out / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
