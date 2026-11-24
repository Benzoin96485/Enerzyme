"""Coordinate, feature and force contracts for native registry presets.

AllScAIP uses its equivariant encoder, EquFlashV2 its copy-major gate, and
MACE-Polar its analytic multipole field for the symmetry checks. Compiler
capture always exercises the default public configuration.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from e3nn import o3

from enerzyme.models.presets import available_architectures, default_params, get_architecture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core_harness import build, forward, make_batch, random_molecules


NATIVE_PRESETS = [name for name in available_architectures() if get_architecture(name).family != "wrapper"]


def _run(model, sample, positions, order, *, capture_features=True):
    mol = dict(sample, Ra=positions[order].numpy(), Za=sample["Za"][order.numpy()])
    torch.manual_seed(171)
    result = {}

    def latent(_module, _args, output):
        for key in ("atom_feature", "atom_vector_feature"):
            if key in output:
                result[key] = output[key].detach()

    handle = model.head.register_forward_hook(latent) if capture_features else None
    try:
        out = forward(model, make_batch([mol]), torch.float64)
        result.update((key, out[key].detach()) for key in ("E", "Fa"))
    finally:
        if handle is not None:
            handle.remove()
    return result


@pytest.fixture(scope="module", params=NATIVE_PRESETS)
def case(request):
    name = request.param
    torch.manual_seed(83)
    params, layers = default_params(name)
    core = next(layer for layer in layers if layer["name"] == "Core")
    if name == "allscaip":
        core.setdefault("params", {})["equivariant_mode"] = True
    elif name == "equflashv2":
        options = core.setdefault("params", {})
        for key in ("nonlinear_kwargs", "ff_nonlinear_kwargs"):
            options.setdefault(key, {})["copy_major"] = True
    elif name == "mace_polar":
        field = next(layer for layer in layers if layer["name"] == "PolarField")
        field.setdefault("params", {})["electrostatics_backend"] = "analytic"
    model = build(name, layers, params).double().eval()
    sample, = random_molecules(42, [5], density=0.14)
    xyz = torch.from_numpy(sample["Ra"])
    return name, model, sample, xyz, _run(model, sample, xyz, torch.arange(len(xyz)))


def _check_transform(case, transform, translation, indices):
    name, model, sample, xyz, ref = case
    out = _run(model, sample, xyz @ transform.T + translation, indices)
    atol = 1e-5 if name in ("e2former_lsr", "e2former_v2", "e2former") else 3e-6
    torch.testing.assert_close(out["E"], ref["E"], rtol=1e-5, atol=atol)
    expected = torch.einsum("ij,nj...->ni...", transform, ref["Fa"][indices])
    torch.testing.assert_close(out["Fa"], expected, rtol=2e-5, atol=atol)
    spec = model.feature_spec
    if spec is not None:
        expected = ref["atom_feature"][indices]
        if spec.irreps and any(ir.l or ir.p < 0 for _, ir in o3.Irreps(spec.irreps)):
            D = o3.Irreps(spec.irreps).D_from_matrix(transform).to(expected)
            expected = torch.einsum("ij,nj...->ni...", D, expected)
        torch.testing.assert_close(out["atom_feature"], expected, rtol=2e-5, atol=atol)
    if "atom_vector_feature" in ref and ref["atom_vector_feature"].shape[1] == 3:
        expected = torch.einsum("ij,nj...->ni...", transform, ref["atom_vector_feature"][indices])
        torch.testing.assert_close(out["atom_vector_feature"], expected, rtol=2e-5, atol=atol)


def test_translation_and_relabeling(case):
    xyz = case[3]
    _check_transform(case, torch.eye(3, dtype=xyz.dtype), xyz.new_tensor([1.1, -0.2, 0.63]),
                     torch.tensor([2, 0, 4, 1, 3]))


def test_rotation(case):
    xyz = case[3]
    axis = xyz.new_tensor([0.37, -0.18, 0.86])
    skew = torch.linalg.cross(axis.expand(3, 3), torch.eye(3, dtype=xyz.dtype), dim=-1)
    _check_transform(case, torch.linalg.matrix_exp(skew), 0, torch.arange(len(xyz)))


def test_reflection_when_supported(case):
    name, model, _, xyz, _ = case
    parity = getattr(model, "is_parity", getattr(model, "parity", None))
    if not (parity or name in ("schnet", "physnet", "spookynet", "painn", "mace", "mace_polar",
                               "tensornet", "tensornet2", "camp", "cace", "dpa4c", "spinconv")):
        pytest.skip("preset does not declare inversion symmetry")
    _check_transform(case, -torch.eye(3, dtype=xyz.dtype), 0, torch.arange(len(xyz)))


def test_force_is_energy_derivative(case):
    name, model, sample, xyz, ref = case
    order = torch.arange(len(xyz))
    direction = torch.from_numpy(np.random.default_rng(18).normal(size=xyz.shape))
    direction /= direction.norm()
    # DPA4's normalized spherical grid computes intermediate statistics in fp32.
    step = 1e-3 if name == "dpa4" else 2e-5
    plus = _run(model, sample, xyz + step * direction, order)
    minus = _run(model, sample, xyz - step * direction, order)
    slope = (plus["E"] - minus["E"]).sum(dim=0) / (2 * step)
    force_slope = -torch.einsum("ni,ni...->...", direction, ref["Fa"])
    torch.testing.assert_close(slope, force_slope, rtol=2e-4, atol=5e-5 if name == "dpa4" else 3e-6)


def test_dynamo_capture_of_energy_and_force(case):
    name, model, sample, xyz, ref = case
    if name in ("allscaip", "equflashv2", "mace_polar"):
        torch.manual_seed(83)
        model = build(name).double().eval()
        ref = _run(model, sample, xyz, torch.arange(len(xyz)), capture_features=False)
    graphs = []

    def capture(graph, _examples):
        graphs.append(graph)
        return graph.forward

    torch.compiler.reset()
    try:
        compiled = torch.compile(model, backend=capture)
        out = _run(compiled, sample, xyz, torch.arange(len(xyz)), capture_features=False)
        operators = {
            getattr(node.target, "__name__", str(node.target))
            for graph in graphs for node in graph.graph.nodes
            if node.op in ("call_function", "call_method")
        }
        message_ops = {"linear", "addmm", "mm", "matmul", "einsum", "bmm",
                       "index_add", "index_add_", "scatter_add", "scatter_add_", "embedding"}
        assert operators.intersection(message_ops), sorted(operators)
        for key in ("E", "Fa"):
            torch.testing.assert_close(out[key], ref[key], rtol=1e-8, atol=1e-8)
    finally:
        torch.compiler.reset()
