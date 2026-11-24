"""Check one preset's compiled energy, forces and force-loss parameter gradients.

Run each preset in a separate process to isolate compiler caches and failures.
For example, from the checkout:

    PYTHONPATH=. python scripts/check_compiled_forces.py so3krates --dtype float64
"""

import argparse
import copy
import json
import math
import random
from contextlib import nullcontext

import numpy as np
import torch

from core_harness import build, forward, make_batch, random_molecules


def step(model, runner, batch, dtype):
    torch.manual_seed(22)
    np.random.seed(22)
    random.seed(22)
    model.zero_grad(set_to_none=True)
    out = forward(runner, batch, dtype, train=True)
    (out["E"].square().sum() + out["Fa"].square().sum()).backward()
    values = {key: out[key].detach().clone() for key in ("E", "Fa")}
    values.update({name: p.grad.detach().clone() if p.grad is not None else None
                   for name, p in model.named_parameters()})
    return values


def check(architecture, dtype, atol, rtol, device, *, e3nn_codegen="default", match_rng=False):
    from torch._dynamo.backends.registry import lookup_backend
    from enerzyme.models.ff import get_ff_core

    torch.manual_seed(142)
    np.random.seed(142)
    random.seed(142)
    _, defaults, _ = get_ff_core(architecture)
    model = build(architecture, build_params={**defaults, "e3nn_codegen": e3nn_codegen})
    model = model.float().to(device=device, dtype=dtype)
    batch = make_batch(random_molecules(107, [5, 4], density=0.12), torch.float32)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    weights = copy.deepcopy(model.state_dict())
    expected = step(model, model, batch, dtype)
    model.load_state_dict(weights, strict=True)

    graphs = []

    def lower(graph, inputs):
        ops = [getattr(n.target, "__name__", str(n.target)) for n in graph.graph.nodes
               if n.op in ("call_function", "call_method")]
        learned = {"embedding", "linear", "addmm", "mm", "matmul", "einsum", "bmm", "index_add_", "scatter_add_"}
        graphs.append({"operations": len(ops), "learned_operators": sorted(set(ops) & learned)})
        return lookup_backend("inductor")(graph, inputs)

    from torch._inductor import config
    with config.patch(fallback_random=True) if match_rng else nullcontext():
        actual = step(model, torch.compile(model, backend=lower), batch, dtype)
    errors, changed_participation, failed = {}, {}, []
    for name, reference in expected.items():
        value = actual[name]
        if (reference is None) != (value is None):
            available = reference if value is None else value
            changed_participation[name] = {
                "absent_in": "compiled" if value is None else "eager",
                "other_is_zero": bool(torch.count_nonzero(available) == 0),
            }
            continue
        if reference is None:
            continue
        delta = value.double() - reference.double()
        norm = reference.double().norm()
        max_abs = float(delta.abs().max()) if delta.numel() else 0.0
        rel_l2 = float(delta.norm() / norm) if norm > 0 else None
        errors[name] = {"max_abs": max_abs if math.isfinite(max_abs) else None,
                        "relative_l2": rel_l2 if rel_l2 is not None and math.isfinite(rel_l2) else None}
        if not torch.allclose(value, reference, rtol=rtol, atol=atol):
            failed.append(name)
    return {"architecture": architecture, "torch": str(torch.__version__), "device": str(device),
            "dtype": str(dtype), "atol": atol, "rtol": rtol, "graphs": graphs,
            "e3nn_codegen": e3nn_codegen, "matched_rng": match_rng,
            "parameter_gradients_compared": len(errors) - 2, "changed_gradient_participation": changed_participation,
            "outside_tolerance": failed, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("architecture")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--e3nn-codegen", choices=("default", "fx"), default="default",
                        help="Build the preset with the selected e3nn code generator.")
    parser.add_argument("--match-rng", action="store_true",
                        help="Use native random draws in Inductor to compare identical dropout masks.")
    args = parser.parse_args()
    double = args.dtype == "float64"
    atol = args.atol if args.atol is not None else (1e-10 if double else 7e-4)
    rtol = args.rtol if args.rtol is not None else (1e-8 if double else 1e-3)
    report = check(args.architecture, getattr(torch, args.dtype), atol, rtol, torch.device(args.device),
                   e3nn_codegen=args.e3nn_codegen, match_rng=args.match_rng)
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(bool(report["outside_tolerance"] or report["changed_gradient_participation"]))


if __name__ == "__main__":
    raise SystemExit(main())
