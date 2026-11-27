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
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", choices=["molecules", "crystals"], required=True)
    parser.add_argument("--lightning", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if min(args.warmup, args.steps) < 1:
        parser.error("warmup and measured steps must both be positive")
    tree = args.source_tree.resolve()
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
    start = elapsed = peak = None

    def record_step(optimizer, _args, _kwargs):
        nonlocal steps, start, elapsed, peak
        steps += 1
        if steps == args.warmup:
            torch.cuda.synchronize()
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
    revision = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    record = dict(revision=revision, rank=rank, world_size=world, host=socket.gethostname(),
                  device=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  lightning=args.lightning, fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  config_sha256=config_digest, system=args.system, atoms=atoms, edges=features["N_pair"],
                  global_batch=global_batch, warmup=args.warmup, steps=args.steps,
                  seconds=elapsed, step_ms=1000 * elapsed / args.steps, peak_bytes=peak)
    (out / f"rank-{rank}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
