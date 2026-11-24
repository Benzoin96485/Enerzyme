"""Trace e3nn's generated kernels through the ordinary force-training entry point."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import e3nn
import pytest
import torch

from enerzyme.models import pipeline
from enerzyme.models.ff import get_ff_core
from scripts.core_harness import build, forward, make_batch, random_molecules


@pytest.mark.parametrize("architecture", ["nequip", "allegro", "botnet"])
def test_fx_codegen_preserves_checkpoint_derivatives_and_captures_contractions(architecture):
    torch.compiler.reset()
    try:
        torch.manual_seed(142)
        _, build_params, _ = get_ff_core(architecture)
        reference = build(architecture).double()
        model = build(architecture, build_params={**build_params, "e3nn_codegen": "fx"}).double()
        assert reference._scripted_force_kernels and not model._scripted_force_kernels
        model.load_state_dict(reference.state_dict(), strict=True)

        calls = set()

        def record(graph, _):
            calls.update(getattr(node.target, "__name__", "") for node in graph.graph.nodes)
            return graph.forward

        batch = make_batch(random_molecules(107, [5, 4]), torch.float64)
        compiled = torch.compile(model, backend=record)
        expected = forward(reference, batch, torch.float64, train=True)
        actual = forward(compiled, batch, torch.float64, train=True)
        for outputs in (expected, actual):
            (outputs["E"].square().sum() + outputs["Fa"].square().sum()).backward()
        for key in ("E", "Fa"):
            torch.testing.assert_close(actual[key], expected[key])
        assert calls & {"tensordot", "einsum"}
        assert dict(model.named_parameters()).keys() == dict(reference.named_parameters()).keys()
        for (name, weight), ref_weight in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(weight.grad, ref_weight.grad, msg=name)
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("architecture", ["equflash_v2", "nequix"])
def test_fx_cpu_force_training_uses_native_derivatives(architecture):
    torch.compiler.reset()
    try:
        torch.manual_seed(142)
        _, params, _ = get_ff_core(architecture)
        reference = build(architecture).double()
        model = build(architecture, build_params={**params, "e3nn_codegen": "fx"}).double()
        model.load_state_dict(reference.state_dict(), strict=True)
        batch = make_batch(random_molecules(107, [5, 4]), torch.float64)
        expected = forward(reference, batch, torch.float64, train=True)
        actual = forward(torch.compile(model), batch, torch.float64, train=True)
        for output in (expected, actual):
            (output["E"].square().sum() + output["Fa"].square().sum()).backward()
        for key in ("E", "Fa"):
            torch.testing.assert_close(actual[key], expected[key])
            assert type(actual[key].grad_fn).__name__ != "CompiledFunctionBackward"
        for (name, weight), ref_weight in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(weight.grad, ref_weight.grad, msg=name)
    finally:
        torch.compiler.reset()


def test_concurrent_default_build_keeps_the_callers_e3nn_configuration(monkeypatch):
    entered, release, default_started, default_read = Event(), Event(), Event(), Event()
    original = e3nn.get_optimization_defaults()["jit_script_fx"]

    def read_codegen(_core, _layers, params, _verbose):
        if params.get("e3nn_codegen") == "fx":
            entered.set()
            assert release.wait(timeout=10)
        else:
            default_read.set()
        return e3nn.get_optimization_defaults()["jit_script_fx"]

    def default_build():
        default_started.set()
        return pipeline.build_model("nequip", [], {})

    monkeypatch.setattr(pipeline, "_build_layers", read_codegen)
    with ThreadPoolExecutor(2) as pool:
        fx = pool.submit(pipeline.build_model, "nequip", [], {"e3nn_codegen": "fx"})
        try:
            assert entered.wait(timeout=10)
            regular = pool.submit(default_build)
            assert default_started.wait(timeout=10)
            assert not default_read.wait(timeout=0.1)
        finally:
            release.set()
        assert fx.result() is False
        assert regular.result() is original
    assert e3nn.get_optimization_defaults()["jit_script_fx"] is original
