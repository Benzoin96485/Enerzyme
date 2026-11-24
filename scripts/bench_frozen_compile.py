"""Measure ordinary torch.compile against a pre-refactor snapshot on one checkout.

Run the same script, snapshot, precision and systems on both checkouts. The
first compiled warmup is compared with eager energy, force and every active
weight derivative before subsequent timed repetitions. Progress is persisted
after each mode so later compiler timeouts and device OOMs retain earlier data.
"""

import argparse
import gc
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bench_cores import SYSTEM_SETS, _batch
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
        ("energy", [(fields["E"], ref_fields["E"])]),
        ("force", [(fields["Fa"], ref_fields["Fa"])]),
        ("gradient", [(gradients[k], ref_gradients[k]) for k in gradients.keys() & ref_gradients.keys()]),
    ):
        atol, rtol = (1e-8, 1e-7) if dtype == torch.float64 else (3e-4, 3e-4)
        if quantity == "gradient" and dtype == torch.float32:
            atol, rtol = 7e-4, 1e-3
        error = 0.0
        for a, b in pairs:
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


def run(args):
    sys.path.insert(0, str(args.model_tree.resolve()))
    from enerzyme.models.ff import get_ff_core

    torch.set_num_threads(args.threads)
    dtype, device = getattr(torch, args.precision), torch.device(args.device)
    payload = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    arch = payload["architecture"]
    _, params, _ = get_ff_core(arch)
    cutoff = max(float(params.get("cutoff_sr") or 0), float(params.get("cutoff_lr") or 0), 6.0)

    def load_model():
        seed_run()
        model = build(arch).float()
        state = payload["state_dict"]
        if model.state_dict().keys() != state.keys():
            sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "test"))
            from presnapshot_maps import remap_snapshot

            state = remap_snapshot(arch, state, payload["param_names"], model)
        model.load_state_dict(state, strict=True)
        return model.to(device=device, dtype=dtype), state

    def write():
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps({arch: result}, allow_nan=False))
        temp.replace(args.out)

    result = {
        "device": device.type,
        "precision": args.precision,
        "compile": True,
        "reps": args.reps,
        "threads": args.threads,
        "snapshot": args.snapshot.name,
        "snapshot_revision": payload["commit"],
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "compile_capture": {},
        "numerical": {},
        "requested_systems": args.systems,
        "modes": args.modes,
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
    write()
    model, initial_state = load_model()
    result["params"] = sum(p.numel() for p in model.parameters())
    host_batches = {
        name: _batch(name, SYSTEM_SETS["acceptance"][name], dtype, cutoff) for name in args.systems
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
    del model, initial_state
    clean()
    model, initial_state = load_model()
    runner = torch.compile(model)
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
    write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tree", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--precision", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--systems", nargs="+", choices=SYSTEM_SETS["acceptance"], default=list(SYSTEM_SETS["acceptance"])
    )
    parser.add_argument("--modes", nargs="+", choices=("infer", "train"), default=["infer", "train"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eager-out", type=Path, help="save native E and Fa for cross-checkout comparison")
    args = parser.parse_args()
    if min(args.reps, args.threads) < 1:
        parser.error("repetitions and threads must be positive")
    run(args)
