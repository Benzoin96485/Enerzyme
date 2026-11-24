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
        assert reference._scripted_force_kernels == (architecture == "botnet")
        assert not model._scripted_force_kernels
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


@pytest.mark.parametrize("architecture,dtype", [
    ("equflash_v2", torch.float32),
    ("equflash_v2", torch.float64),
    ("equiformer", torch.float32),
    ("equiformer", torch.float64),
    ("nequix", torch.float32),
    ("nequix", torch.float64),
])
def test_fx_cpu_force_training_matches_scripted_derivatives(architecture, dtype):
    torch.compiler.reset()
    try:
        torch.manual_seed(142)
        _, params, _ = get_ff_core(architecture)
        reference = build(architecture).to(dtype)
        model = build(architecture, build_params={**params, "e3nn_codegen": "fx"}).to(dtype)
        model.load_state_dict(reference.state_dict(), strict=True)
        batch = make_batch(random_molecules(107, [5, 4]), dtype)
        expected = forward(reference, batch, dtype, train=True)
        actual = forward(torch.compile(model), batch, dtype, train=True)
        for output in (expected, actual):
            (output["E"].square().sum() + output["Fa"].square().sum()).backward()
        from torch._functorch import config as autograd_config
        from torch._inductor import config

        compiled_expected = (architecture in ("nequix", "equflash_v2") and hasattr(config.cpp, "simdlen")
                             and hasattr(autograd_config, "force_non_lazy_backward_lowering"))
        for key in ("E", "Fa"):
            torch.testing.assert_close(actual[key], expected[key])
            compiled = type(actual[key].grad_fn).__name__ == "CompiledFunctionBackward"
            assert compiled == compiled_expected
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
        return pipeline.build_model("botnet", [], {})

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


@pytest.mark.parametrize("architecture", ["nequip", "allegro"])
def test_default_compiled_kernels_match_the_scripted_checkpoint(architecture):
    import copy
    import io

    torch.compiler.reset()
    try:
        torch.manual_seed(21)
        core, params, layers = get_ff_core(architecture)
        reference = pipeline._build_layers(core, layers, params, verbose=0).double()
        model = build(architecture).double()
        assert reference._scripted_force_kernels
        assert not model._scripted_force_kernels
        assert model.state_dict().keys() == reference.state_dict().keys()
        model.load_state_dict(reference.state_dict(), strict=True, assign=True)
        model = copy.deepcopy(model)
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        model = torch.load(stream, map_location="cpu", weights_only=False)
        assert model.state_dict().keys() == reference.state_dict().keys()
        batch = make_batch(random_molecules(107, [5, 4]), torch.float64)
        calls = set()

        def record(graph, _):
            calls.update(getattr(n.target, "__name__", "") for n in graph.graph.nodes)
            return graph.forward

        def result(module, runner):
            module.zero_grad(set_to_none=True)
            output = forward(runner, batch, torch.float64, train=True)
            (output["E"].square().sum() + output["Fa"].square().sum()).backward()
            return {key: output[key] for key in ("E", "Fa")}, {
                name: p.grad for name, p in module.named_parameters()
            }

        gold, expected_gradients = result(reference, reference)
        for runner in (model, torch.compile(model, backend=record)):
            actual, gradients = result(model, runner)
            for key in gold:
                torch.testing.assert_close(actual[key], gold[key])
            assert gradients.keys() == expected_gradients.keys()
            for key in gradients:
                torch.testing.assert_close(gradients[key], expected_gradients[key], msg=key)
        assert calls & {"tensordot", "einsum"}
    finally:
        torch.compiler.reset()


def test_generated_tensor_product_keeps_one_set_of_movable_constants():
    import copy
    import io
    from e3nn import o3
    from e3nn.util.jit import script
    from enerzyme.models._e3nn_codegen import traceable_e3nn_kernels
    from enerzyme.models.mp.equivariant.e3nn.dispatch import install_generated_kernels

    def tensor_product():
        return o3.FullyConnectedTensorProduct("3x1o+3x2e", "2x2e", "3x1o+2x2e")

    stock = tensor_product()
    with traceable_e3nn_kernels():
        product = install_generated_kernels(tensor_product())
    product.load_state_dict(stock.state_dict(), strict=True)
    assert product.state_dict().keys() == stock.state_dict().keys()
    kernel = product._compiled_main_left_right
    kernel.double()
    for name, buffer in kernel.named_buffers():
        assert "_w3j" in name and buffer.dtype == torch.float64
        assert getattr(kernel._script, name).data_ptr() == getattr(kernel._fx, name).data_ptr() == buffer.data_ptr()
    stream = io.BytesIO()
    torch.save(product, stream)
    assert kernel._fx.graph.owning_module is kernel._fx
    stream.seek(0)
    meta = torch.load(stream, map_location="meta", weights_only=False)
    assert all(buffer.device.type == "meta" for buffer in meta.buffers())
    meta.to_empty(device="cpu")
    stock.double()
    weights = {key: val.clone() for key, val in stock.state_dict().items()}
    for key in weights:
        if "_w3j" in key:
            weights[key].mul_(1.125)
    stock.load_state_dict(weights)
    meta.double().load_state_dict(copy.deepcopy(weights), assign=True)
    args = (torch.randn(8, stock.irreps_in1.dim, dtype=torch.float64),
            torch.randn(8, stock.irreps_in2.dim, dtype=torch.float64))
    torch.testing.assert_close(meta(*args), stock(*args))
    torch.testing.assert_close(script(meta)(*args), stock(*args))
