"""Compiled pipelines must not recompile when called again on the same inputs."""

import pytest
import torch
import torch._dynamo


@pytest.mark.parametrize("fullgraph", [False, True])
@pytest.mark.parametrize("use_pyg", [False, True])
def test_compiled_energy_reduction_freezes_each_layers_first_fields(fullgraph, use_pyg):
    from enerzyme.models.layers.reduce import EnergyReduceLayer
    from torch_geometric.data import Data

    @torch.compile(backend="eager", fullgraph=fullgraph)
    def energy(layer, fields):
        return layer(fields)["E"]

    def fields(correction=None):
        values = {"Ea": torch.tensor([1., 2.]), "Za": torch.ones(2, dtype=torch.long),
                  "batch" if use_pyg else "batch_seg": torch.zeros(2, dtype=torch.long), "N": torch.tensor([2])}
        if correction is not None:
            values[correction] = torch.tensor([10., 20.])
        return Data(**values) if use_pyg else values

    torch._dynamo.reset()
    try:
        for layer in (EnergyReduceLayer(), EnergyReduceLayer()):
            torch.testing.assert_close(energy(layer, fields()), torch.tensor([3.]))
            with torch._dynamo.config.patch(error_on_recompile=True):
                torch.testing.assert_close(energy(layer, fields()), torch.tensor([3.]))
            torch.testing.assert_close(energy(layer, fields("Eextraa")), torch.tensor([3.]))
            torch.testing.assert_close(layer(fields("Eextraa"))["E"], torch.tensor([3.]))

        layer = EnergyReduceLayer()
        torch.testing.assert_close(energy(layer, fields("Eextraa")), torch.tensor([33.]))
        layer.reset_field_name(Eextraa="correction")
        torch.testing.assert_close(energy(layer, fields("correction")), layer(fields("correction"))["E"])
    finally:
        torch._dynamo.reset()


def test_compiled_pipeline_second_call_reuses_its_graphs():
    # Layers adapt their input fields lazily on the first call. Doing that by
    # mutating the layer while it is traced made the second call fail the
    # guard on the flag and recompile every graph that inlines the layer.
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(0)
    model = build("schnet").double()
    batch = make_batch(random_molecules(3, [4, 5], density=0.12), torch.float64)
    expected = forward(model, batch, torch.float64)
    expected = {k: expected[k].detach() for k in ("E", "Fa")}
    fresh = build("schnet").double()
    fresh.load_state_dict(model.state_dict())
    torch._dynamo.reset()
    compiled = torch.compile(fresh, backend="eager")
    forward(compiled, batch, torch.float64)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(2):
            out = forward(compiled, batch, torch.float64)
            for key, value in expected.items():
                torch.testing.assert_close(out[key].detach(), value, rtol=1e-12, atol=1e-12)
    torch._dynamo.reset()


def test_float64_cuda_forces_compile_without_mix_order_reductions():
    # PyTorch 2.13 emits float32 accumulators for float64 mix-order reductions,
    # which Triton rejects. Only float64 CUDA coordinates change the setting.
    from types import SimpleNamespace
    from torch._functorch import config as autograd_config
    from torch._inductor import config
    from enerzyme.models.layers.compiled_grad import float64_cuda_reduction_scope

    if not hasattr(config.triton, "mix_order_reduction"):
        return
    default = config.triton.mix_order_reduction
    lazy = getattr(autograd_config, "force_non_lazy_backward_lowering", None)
    cuda64 = SimpleNamespace(is_cuda=True, dtype=torch.float64)
    for training in (False, True):
        core = SimpleNamespace(training=training)
        with float64_cuda_reduction_scope(core, {"Ra": cuda64}):
            assert config.triton.mix_order_reduction is False
            if lazy is not None:
                assert autograd_config.force_non_lazy_backward_lowering is (training or lazy)
        assert config.triton.mix_order_reduction == default
    with float64_cuda_reduction_scope(SimpleNamespace(training=False, _cuda_mix_order_reduction=False),
                                      {"Ra": SimpleNamespace(is_cuda=True, dtype=torch.float32)}):
        assert config.triton.mix_order_reduction is False
    for coordinates in (torch.zeros(2, 3, dtype=torch.float64), SimpleNamespace(is_cuda=True, dtype=torch.float32)):
        with float64_cuda_reduction_scope(SimpleNamespace(training=True), {"Ra": coordinates}):
            assert config.triton.mix_order_reduction == default


def test_float64_cuda_scope_is_not_traced_into_recompiles():
    # The force entry points only skip their own frame. Traced, the scope would
    # guard on the Inductor flag it patches and recompile on every later call.
    from types import SimpleNamespace
    from enerzyme.models.layers.compiled_grad import float64_cuda_reduction_scope

    core = SimpleNamespace(training=False)
    inputs = {"Ra": SimpleNamespace(is_cuda=True, dtype=torch.float64)}

    def step(x):
        with float64_cuda_reduction_scope(core, inputs):
            y = x * 2
        return y + 1

    torch._dynamo.reset()
    compiled = torch.compile(step, backend="eager")
    x = torch.ones(3)
    compiled(x)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for _ in range(3):
            torch.testing.assert_close(compiled(x), x * 2 + 1)
    torch._dynamo.reset()
