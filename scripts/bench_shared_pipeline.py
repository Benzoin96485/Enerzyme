"""Compare collation, range selection and result collection on identical model weights."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import sys
from unittest.mock import patch

import numpy as np
import torch

from enerzyme.models.ff import build_model, get_ff_core
from enerzyme.models.layers.geometry import DistanceLayer, RangeSeparationLayer
from enerzyme.tasks import batch


def reference_module(root, name):
    package, _, leaf = name.rpartition(".")
    path = root.joinpath(*name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(f"{package}._reference_{leaf}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_model(name, dtype, device):
    _, settings, layers = get_ff_core(name)
    settings = dict(settings, dim_embedding=32, num_rbf=16)
    layers = deepcopy(layers)
    for layer in layers:
        if layer["name"] == "Core":
            params = layer["params"]
            params["num_interactions"] = 2
            if name == "SchNet":
                params["hidden_channels"] = 32
    if name == "SchNet":
        # Short-range E/F model with the same Gaussian RBF and atom-energy head.
        layers = [x for x in layers if x["name"] in (
            "RangeSeparation", "GaussianRBF", "RandomAtomEmbedding", "Core",
            "AtomicAffine", "EnergyReduce", "Force",
        )]
    torch.manual_seed(129)
    return build_model(name, layers, settings, verbose=0).to(device=device, dtype=dtype)


def samples():
    rng = np.random.default_rng(72)
    batch_samples = []
    for i in range(16):
        n = 16 + i
        features = dict(N=n, Ra=rng.uniform(0, 7, (n, 3)), Za=rng.choice([1, 6, 8], n))
        batch_samples.append((features, dict(E=0., Fa=np.zeros((n, 3))), str(i)))
    return batch_samples


def measure(args):
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(1)
    old_batch = reference_module(args.reference_tree, "enerzyme.tasks.batch")
    old_batch.full_neighbor_list = reference_module(args.reference_tree, "enerzyme.data.neighbor_list").full_neighbor_list
    old_geometry = reference_module(args.reference_tree, "enerzyme.models.layers.geometry")
    records = {}
    data = samples()
    for name in ("SchNet", "PaiNN"):
        model = make_model(name, dtype, device)
        for mode in ("predict", "force_backward"):
            train = mode == "force_backward"
            model.train(train)

            def step(module, capture=False):
                inputs, targets = module._to_device(module._decorate_batch_input(data, dtype, device), device)
                outputs = model(inputs)
                if train:
                    model.zero_grad(set_to_none=True)
                    (outputs["E"].square().mean() + outputs["Fa"].square().mean()).backward()
                else:
                    module._decorate_batch_output(outputs, inputs, targets)
                if capture:
                    values = {key: outputs[key].detach().cpu() for key in ("E", "Fa")}
                    if train:
                        values.update({key: p.grad.detach().cpu().clone() for key, p in model.named_parameters()
                                       if p.grad is not None})
                    return values

            rounds = {"baseline": [], "candidate": []}
            peaks = {"baseline": [], "candidate": []}
            for label in ("baseline", "candidate", "candidate", "baseline"):
                old = label == "baseline"
                module = old_batch if old else batch
                with ExitStack() as stack:
                    if old:
                        for layer in (DistanceLayer, RangeSeparationLayer):
                            old_layer = getattr(old_geometry, layer.__name__)
                            stack.enter_context(patch.object(layer, "get_output", old_layer.get_output))
                    actual = step(module, capture=True)
                    if old:
                        expected = actual
                    else:
                        assert expected.keys() == actual.keys()
                        for key, value in actual.items():
                            atol, rtol = (1e-10, 1e-9) if dtype == torch.float64 else (2e-5, 1e-4)
                            torch.testing.assert_close(value, expected[key], atol=atol, rtol=rtol,
                                                       msg=lambda msg: f"{name} {mode} {key}: {msg}")
                    for _ in range(3):
                        step(module)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.reset_peak_memory_stats(device)
                    times = []
                    for _ in range(args.reps):
                        start = perf_counter()
                        step(module)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        times.append((perf_counter() - start) * 1000)
                    rounds[label].append(median(times))
                    if device.type == "cuda":
                        peaks[label].append(torch.cuda.max_memory_allocated(device))
            records[f"{name}/{mode}"] = {
                "round_medians_ms": rounds,
                "peak_bytes": peaks,
                "speedup": median(rounds["baseline"]) / median(rounds["candidate"]),
                "compared_tensors": len(expected),
            }
    report = {
        "device": "cpu",
        "torch": str(torch.__version__),
        "dtype": args.dtype,
        "iterations_per_round": args.reps,
        "results": records,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(device=props.name, device_uuid=str(props.uuid), cuda=torch.version.cuda)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-tree", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    report = json.dumps(measure(args), indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")
