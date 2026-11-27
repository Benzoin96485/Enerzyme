"""Compare Wigner backends on identical differentiable Cartesian frames."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from enerzyme.models.mp.equivariant.rotation import edge_rotation
from enerzyme.models.mp.equivariant.rotation.so3_rotation import init_edge_rot_mat


def benchmark(device, dtype, edge_counts, degrees, reps):
    rng = torch.Generator(device=device).manual_seed(7)
    results = []
    for count in edge_counts:
        edges = torch.randn(count, 3, device=device, dtype=dtype, generator=rng, requires_grad=True)
        for lmax in degrees:
            # A shared frame removes gauge differences between DPA4, TECE and
            # eSCN. Truncation also exercises the fused order projection.
            mmax = min(2, lmax)
            reference = edge_rotation(edges, lmax, "jd", frame=init_edge_rot_mat(edges), mmax=mmax)
            weight = torch.randn(reference.wigner.shape, device=device, dtype=dtype, generator=rng)
            reference_grad, = torch.autograd.grad((reference.wigner * weight).sum(), edges)
            for backend in ("jd", "cg", "quaternion", "fused"):
                readings = []
                for step in range(reps + 2):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    rotations = edge_rotation(edges, lmax, backend, frame=init_edge_rot_mat(edges), mmax=mmax)
                    grad, = torch.autograd.grad((rotations.wigner * weight).sum(), edges)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    if step >= 2:
                        readings.append(1000 * (time.perf_counter() - start))
                results.append({
                    "edges": count, "lmax": lmax, "mmax": mmax, "backend": backend,
                    "forward_and_gradient_ms": statistics.median(readings),
                    "max_abs_rotation_diff": (reference.wigner.detach() - rotations.wigner.detach()).abs().max().item(),
                    "max_abs_edge_gradient_diff": (reference_grad - grad).abs().max().item(),
                })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    device = torch.device(args.device)
    records = {"_meta": {"device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                         "torch": str(torch.__version__), "threads": torch.get_num_threads(), "reps": args.reps,
                         "workload": "edge_rotation by backend name with an explicit frame, mmax <= 2; forward plus edge-vector gradient"}}
    for dtype in (torch.float32, torch.float64):
        records[str(dtype)] = benchmark(device, dtype, (64, 1024), (1, 2, 3, 4), args.reps)
    args.out.write_text(json.dumps(records, indent=1, allow_nan=False) + "\n")
