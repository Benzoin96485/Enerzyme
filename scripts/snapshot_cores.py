"""Freeze seeded weights, inputs and float64 outputs of every architecture.

Snapshots taken before a Core is rewritten let the rewritten Core be checked
against the old numerics: load the old weights through a key map and compare
outputs (see ``test/test_presnapshot_parity.py``). They are local artefacts and
are not committed.

    PYTHONPATH=. python scripts/snapshot_cores.py --out test/fixtures/presnapshot [--only a,b]
"""
import argparse
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from core_harness import ARCHITECTURES, build, forward, make_batch, random_molecules, targets


def snapshot(arch: str, out_dir: str, seed: int = 7) -> str:
    """Float64 outputs, or float32 when the Core cannot run in float64."""
    from enerzyme.models import ff

    # Record the checkout providing the model even when launched from elsewhere.
    commit = subprocess.check_output(
        ["git", "-C", str(Path(ff.__file__).resolve().parent), "rev-parse", "HEAD"], text=True
    ).strip()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build(arch).float()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batch = make_batch(random_molecules(seed, [9, 6, 11]), dtype=torch.float64)
    try:
        dtype = torch.float64
        out = targets(forward(model.double(), batch, dtype))
    except RuntimeError as err:
        print(f"{arch}: float64 failed ({err}); snapshotting float32", flush=True)
        dtype = torch.float32
        model.load_state_dict(state)
        out = targets(forward(model.float(), batch, dtype))
    payload = dict(
        architecture=arch, state_dict=state, batch=batch, outputs=out, dtype=dtype,
        param_names=[k for k, _ in model.named_parameters()], commit=commit,
    )
    path = os.path.join(out_dir, f"{arch.replace('+', 'p')}.pt")
    torch.save(payload, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test/fixtures/presnapshot")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    archs = args.only.split(",") if args.only else ARCHITECTURES
    failed = []
    for arch in archs:
        t = time.time()
        try:
            path = snapshot(arch, args.out)
            print(f"{arch}: {path} ({time.time() - t:.1f}s)", flush=True)
        except Exception as e:
            failed.append(arch)
            print(f"{arch}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    if failed:
        print("failed:", ",".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
