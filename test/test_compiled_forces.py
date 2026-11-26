"""Conservative force training through torch.compile's joint forward/backward."""

import copy
from contextlib import nullcontext

import pytest
import torch

from enerzyme.models.ff import build_model


@pytest.fixture(autouse=True)
def isolate_compiler_cache():
    # These tests compile the same forward method with different model classes
    # and backends. Each case needs the compiler's normal specialization budget.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def test_compiled_force_layer_keeps_hooks_added_after_first_call():
    from torch._dynamo.exc import TorchRuntimeError
    from enerzyme.models.layers.gradient import ForceLayer
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(93)
    model = build("schnet").double()
    batch = make_batch(random_molecules(107, [5, 4]), torch.float64)
    layer, = (stage for stage in model.post_sequence if type(stage) is ForceLayer)

    def values(runner):
        out = forward(runner, batch, torch.float64, train=True)
        grads = torch.autograd.grad(out["Fa"].square().sum(), tuple(model.parameters()), allow_unused=True)
        return out["E"], out["Fa"], grads

    runner = torch.compile(model, backend="eager")
    original = values(runner)

    def scale_energy(module, args):
        return ({**args[0], "E": 2 * args[0]["E"]},)

    def scale_forces(module, args, result):
        return {**result, "Fa": 3 * result["Fa"]}

    with layer.register_forward_pre_hook(scale_energy), layer.register_forward_hook(scale_forces):
        expected = values(model)
        torch.testing.assert_close(expected[1], 6 * original[1])
        torch.testing.assert_close(values(runner), expected)
        torch.compiler.reset()
        with pytest.raises((AssertionError, TorchRuntimeError), match="ForceLayer hooks require ordinary"):
            values(torch.compile(model, backend="eager", fullgraph=True))

    torch.testing.assert_close(values(runner), original)



def test_compiled_energy_reduction_uses_current_forward_hooks():
    from torch._dynamo import config
    from enerzyme.models.layers.reduce import EnergyReduceLayer
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    core = build("schnet").double()
    runner = torch.compile(core, backend="aot_eager")
    reduce, = (layer for layer in core.post_sequence if type(layer) is EnergyReduceLayer)
    batch = make_batch(random_molecules(107, [5, 4]), torch.float64)

    def values(target):
        out = forward(target, batch, torch.float64, train=True)
        loss = out["E"].square().sum() + out["Fa"].square().sum()
        grads = torch.autograd.grad(loss, tuple(core.parameters()), allow_unused=True)
        return out["E"], out["Fa"], grads

    def scale_atoms(module, args):
        fields, = args
        return ({**fields, "Ea": 1.5 * fields["Ea"]},)

    def scale_total(module, args, fields):
        return {**fields, "E": 2 * fields["E"]}

    def check():
        torch.testing.assert_close(values(runner), values(core))

    initial_setting = getattr(config, "skip_nnmodule_hook_guards", None)
    check()
    pre = reduce.register_forward_pre_hook(scale_atoms)
    try:
        check()
        with reduce.register_forward_hook(scale_total):
            check()
            pre.remove()
            check()
    finally:
        pre.remove()
    check()
    assert getattr(config, "skip_nnmodule_hook_guards", None) == initial_setting

@pytest.mark.parametrize("architecture", ["cace", "grace", "equiformer_v3"])
def test_conflicting_cpu_kernel_option_uses_native_gradients(monkeypatch, architecture):
    from torch._dynamo.exc import TorchRuntimeError

    # Fail before compiler code generation if the unsafe coordinate VJP is traced.
    def unexpected_lowering(*args, **kwargs):
        raise AssertionError(f"CPU Inductor received a {architecture} force-training graph")

    monkeypatch.setattr(torch._inductor, "compile", unexpected_lowering)
    torch.manual_seed(31)
    model = build_model(architecture, verbose=0).train()
    receivers, senders = torch.where(~torch.eye(4, dtype=torch.bool))
    batch = dict(Ra=torch.randn(4, 3, requires_grad=True), Za=torch.tensor([1, 6, 7, 8]),
                 batch_seg=torch.zeros(4, dtype=torch.long), idx_i=receivers, idx_j=senders)

    def step(runner):
        torch.manual_seed(52)
        model.zero_grad(set_to_none=True)
        out = runner(dict(batch))
        (out["E"].square().sum() + out["Fa"].square().sum()).backward()
        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        return out["E"].detach(), out["Fa"].detach(), gradients

    expected_e, expected_f, expected_g = step(model)
    options = {"cpp.simdlen": 256}

    def delegate(graph, inputs, **kwargs):
        return torch._inductor.compile(graph, inputs, **kwargs)

    backend = delegate if architecture == "grace" else "inductor"
    actual_e, actual_f, actual_g = step(torch.compile(model, backend=backend, options=options))
    torch.testing.assert_close(actual_e, expected_e, atol=0, rtol=0)
    torch.testing.assert_close(actual_f, expected_f, atol=0, rtol=0)
    assert actual_g.keys() == expected_g.keys() and len(actual_g) > 1
    for name in expected_g:
        torch.testing.assert_close(actual_g[name], expected_g[name], atol=0, rtol=0)
    # PyTorch's code cache can reuse a previous non-strict trace of the same
    # method when only fullgraph changes. Exercise an independent strict trace.
    torch.compiler.reset()
    with pytest.raises((AssertionError, TorchRuntimeError), match=r"Fullgraph CPU forces.*cpp\.simdlen"):
        torch.compile(model, backend=backend, fullgraph=True, options=options)(dict(batch))


def test_batched_charge_dipole_and_energy_keep_empty_structures_in_fullgraph():
    from enerzyme.models.layers.electrostatics import AtomicCharge2DipoleLayer, ChargeConservationLayer
    from enerzyme.models.layers.reduce import EnergyReduceLayer

    layers = torch.nn.Sequential(ChargeConservationLayer(), AtomicCharge2DipoleLayer(), EnergyReduceLayer())
    data = dict(N=[2, 0], Za=torch.tensor([1, 1]), batch_seg=torch.tensor([0, 0]),
                Ra=torch.tensor([[1., 0., 0.], [0., 1., 0.]]), Q=torch.tensor([0., 0.]),
                Qa=torch.tensor([0.3, -0.1]), Ea=torch.tensor([2., 3.]))
    result = torch.compile(layers, fullgraph=True, backend="eager")(data)
    torch.testing.assert_close(result["E"], torch.tensor([5., 0.]))
    torch.testing.assert_close(result["Qa"], torch.tensor([0.2, -0.2]))
    torch.testing.assert_close(result["M2"], torch.tensor([[0.2, -0.2, 0.], [0., 0., 0.]]))


def test_inductor_keeps_d4_coordination_and_charge_derivatives():
    from enerzyme.models.layers.dispersion.grimme_d4 import GrimmeD4EnergyLayer
    from scripts.core_harness import make_batch, random_molecules

    batch = make_batch(random_molecules(107, [5, 4], density=0.12), torch.float32)
    za, xyz, i, j = (batch[key] for key in ("Za", "Ra", "idx_i", "idx_j"))
    distance = (xyz[i] - xyz[j]).norm(dim=-1).requires_grad_()
    charge = torch.zeros_like(za, dtype=xyz.dtype, requires_grad=True)
    d4 = GrimmeD4EnergyLayer(max_Za=86).float()

    def derivatives(dist, q):
        def energy(r, charges):
            per_atom = d4.get_E_disp_a(za, charges, r, i, j)
            return per_atom.sum(), per_atom

        total, pullback, per_atom = torch.func.vjp(energy, dist, q, has_aux=True)
        dr, dq = pullback(torch.ones_like(total))
        return per_atom, dr, dq

    expected = derivatives(distance, charge)
    actual = torch.compile(derivatives, fullgraph=True)(distance, charge)
    assert expected[2].abs().max() > 1e-3
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, atol=2e-7, rtol=2e-5)
    weights = (d4._s8, d4._a1, d4._a2, d4._scaleq)
    for output, reference in zip(
        torch.autograd.grad(actual[1].square().sum(), weights),
        torch.autograd.grad(expected[1].square().sum(), weights),
    ):
        torch.testing.assert_close(output, reference, atol=2e-7, rtol=2e-5)


def _schnet(ensemble):
    return build_model(
        "schnet",
        build_params={"dim_embedding": 8, "num_rbf": 6, "cutoff_sr": 2.0, "max_Za": 8},
        layer_params=[
            {"name": "RangeSeparation"},
            {"name": "GaussianRBF", "params": {"flavor": "SchNet", "apply_cutoff_fn": False}},
            {"name": "RandomAtomEmbedding"},
            {"name": "Core", "params": {"hidden_channels": 8, "num_interactions": 1,
                                        "shallow_ensemble_size": ensemble}},
            {"name": "EnergyReduce"},
            {"name": "Force"},
        ],
        verbose=0,
    ).double()


def test_default_compiled_training_captures_the_learned_interactions():
    from torch._dynamo import config

    graphs = []

    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    torch.compiler.reset()
    try:
        torch.manual_seed(705)
        reference = _schnet(1).train()
        candidate = copy.deepcopy(reference)
        with config.patch(capture_dynamic_output_shape_ops=False, capture_scalar_outputs=False):
            compiled = torch.compile(candidate, backend=capture)
            for separation in (1.1, 2.5):
                actual, expected = compiled(_batch(separation)), reference(_batch(separation))
                for key in ("E", "Fa"):
                    torch.testing.assert_close(actual[key], expected[key])
                a = torch.autograd.grad(actual["E"].sum() + actual["Fa"].square().sum(),
                                        candidate.parameters(), allow_unused=True)
                b = torch.autograd.grad(expected["E"].sum() + expected["Fa"].square().sum(),
                                        reference.parameters(), allow_unused=True)
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y)
            assert not config.capture_dynamic_output_shape_ops and not config.capture_scalar_outputs
        operators = {getattr(node.target, "__name__", "") for graph in graphs for node in graph.graph.nodes}
        assert "linear" in operators
    finally:
        torch.compiler.reset()


def test_compiled_mace_preserves_scripted_parameter_gradients():
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    model = build("mace").double()
    batch = make_batch(random_molecules(107, [5, 4]), torch.float64)
    weights = tuple(model.parameters())

    def derivatives(runner):
        out = forward(runner, batch, torch.float64, train=True)
        return [torch.autograd.grad(out[field].square().sum(), weights,
                                    retain_graph=True, allow_unused=True) for field in ("E", "Fa")]

    reference = derivatives(model)
    compiled = derivatives(torch.compile(model, backend="eager"))
    for loss_gradients, expected in zip(compiled, reference):
        for gradient, target in zip(loss_gradients, expected):
            torch.testing.assert_close(gradient, target)


def test_nuclear_embedding_keeps_vjp_tensors_out_of_its_eval_cache():
    from enerzyme.models.layers.atom_embedding import NuclearEmbedding

    model = NuclearEmbedding(max_Za=86, dim_embedding=3, zero_init=False).double()
    species = torch.tensor([1, 6, 8])
    coords = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

    def energy(pos):
        return (model.get_embedding(species) * pos.square()).sum()

    def force_loss(pos):
        potential, pullback = torch.func.vjp(energy, pos)
        forces, = pullback(torch.ones_like(potential))
        return potential + forces.square().sum()

    reference = energy(coords)
    forces, = torch.autograd.grad(reference, coords, create_graph=True)
    weights = tuple(model.parameters())
    expected = torch.autograd.grad(reference + forces.square().sum(), weights)
    table = model.embedding
    loss = torch.compile(force_loss, fullgraph=True, backend="aot_eager")(coords)
    assert model.embedding is table
    for grad, target in zip(torch.autograd.grad(loss, weights), expected):
        torch.testing.assert_close(grad, target)


@pytest.mark.parametrize("architecture", ["dimenet", "m3gnet", "gemnet-oc", "dpa3", "tensornet2"])
def test_compiled_topology_preserves_force_and_parameter_gradients(architecture):
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    model = build(architecture).double()
    weights = tuple(model.parameters())
    operators = set()

    def capture(graph, _inputs):
        operators.update(getattr(node.target, "__name__", "") for node in graph.graph.nodes)
        return graph.forward

    compiled = torch.compile(model, backend=capture)
    for sizes in ([5, 4], [2]):
        batch = make_batch(random_molecules(107, sizes), torch.float64)
        reference = forward(model, batch, torch.float64, train=True)
        result = forward(compiled, batch, torch.float64, train=True)
        for key in ("E", "Fa"):
            torch.testing.assert_close(result[key], reference[key])
        def gradients(out):
            return torch.autograd.grad(out["E"].square().sum() + out["Fa"].square().sum(),
                                       weights, allow_unused=True)
        for gradient, target in zip(gradients(result), gradients(reference)):
            torch.testing.assert_close(gradient, target)
    assert "linear" in operators


@pytest.mark.parametrize("architecture", ["so3krates", "mattersim", "dpa3", "facto", "cace", "grace", "equiformer_v3", "camp"])
def test_cpu_compiler_keeps_force_gradients(architecture):
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    model = build(architecture).double()
    batch = make_batch(random_molecules(107, [5, 4], density=0.12), torch.float64)
    reference = forward(model, batch, torch.float64, train=True)
    from torch._functorch import config as autograd_config
    from torch._inductor import config

    fusion = getattr(config.cpp, "max_horizontal_fusion_size", None)
    simdlen = getattr(config.cpp, "simdlen", None)
    eager_backward = getattr(autograd_config, "force_non_lazy_backward_lowering", None)
    actual = forward(torch.compile(model), batch, torch.float64, train=True)
    assert getattr(config.cpp, "max_horizontal_fusion_size", None) == fusion
    assert getattr(config.cpp, "simdlen", None) == simdlen
    assert getattr(autograd_config, "force_non_lazy_backward_lowering", None) == eager_backward
    for field in ("E", "Fa"):
        torch.testing.assert_close(actual[field], reference[field], rtol=1e-9, atol=1e-12)
    weights = tuple(model.parameters())

    def gradients(out):
        return torch.autograd.grad(out["E"].square().sum() + out["Fa"].square().sum(), weights, allow_unused=True)

    for value, target in zip(gradients(actual), gradients(reference)):
        torch.testing.assert_close(value, target, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dpa3_cpu_inference_keeps_reusable_coordinate_gradients(dtype):
    from torch._functorch import config as autograd_config
    from torch._inductor import config
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    model = build("dpa3").to(dtype)
    batch = make_batch(random_molecules(107, [5, 4], density=0.12), dtype)

    def differentiate(runner):
        out = forward(runner, batch, dtype, train=False)
        first = torch.autograd.grad(out["E"].sum(), out["Ra"], retain_graph=True)[0]
        second = torch.autograd.grad(out["E"].sum(), out["Ra"], retain_graph=True)[0]
        weights = torch.autograd.grad(out["E"].sum(), tuple(model.parameters()), allow_unused=True)
        return out["E"], out["Fa"], first, second, *weights

    expected = differentiate(model)
    scalar_widths = []

    # A custom compiler can delegate to Inductor under any Python name.
    def eager(graph, inputs):
        if any(getattr(node.target, "__name__", "") == "linear" for node in graph.graph.nodes):
            scalar_widths.append(config.cpp.simdlen)
        return torch._inductor.compile(graph, inputs)

    backend = eager if dtype == torch.float64 else "inductor"
    actual = differentiate(torch.compile(model, backend=backend))
    if (backend is eager and hasattr(config.cpp, "simdlen")
            and hasattr(autograd_config, "force_non_lazy_backward_lowering")):
        assert set(scalar_widths) == {1}
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("architecture", ["et", "cace", "grace"])
def test_cpu_inference_matches_force_and_repeated_energy_derivatives(architecture, dtype):
    from torch._functorch import config
    from scripts.core_harness import build, forward, make_batch, random_molecules

    torch.manual_seed(142)
    model = build(architecture).float().to(dtype)
    batch = make_batch(random_molecules(107, [5, 4], density=0.12), torch.float32)
    expected = forward(model, batch, dtype)
    result = forward(torch.compile(model), batch, dtype)
    torch.testing.assert_close(result["E"], expected["E"])
    torch.testing.assert_close(result["Fa"], expected["Fa"])
    assert not result["Fa"].requires_grad
    scope = config.patch(donated_buffer=False) if hasattr(config, "donated_buffer") else nullcontext()
    with scope:
        for retain in (True, False):
            grad, = torch.autograd.grad(result["E"].sum(), result["Ra"], retain_graph=retain)
            torch.testing.assert_close(grad, -expected["Fa"])


def _batch(distance):
    return {
        "Ra": torch.tensor([[0., 0., 0.], [distance, .1, -.2], [2.7, -.4, .3]],
                           dtype=torch.float64, requires_grad=True),
        "Za": torch.tensor([1, 7, 8]), "batch_seg": torch.zeros(3, dtype=torch.long), "N": [3],
        "idx_i": torch.tensor([0, 0, 1, 1, 2, 2]), "idx_j": torch.tensor([1, 2, 0, 2, 0, 1]),
    }


def test_compiled_training_differentiates_replaced_coordinates():
    from torch._dynamo.exc import TorchRuntimeError
    from enerzyme.models.layers import BaseFFLayer

    class ScaleCoordinates(BaseFFLayer):
        def __init__(self):
            super().__init__(input_fields={"Ra"}, output_fields={"Ra"})

        def get_Ra(self, Ra):
            return 0.8 * Ra

    reference = _schnet(1)
    reference.pre_sequence.insert(0, ScaleCoordinates())
    target = copy.deepcopy(reference)

    def values(run, model):
        fields = _batch(0.9)
        out = run(fields)
        loss = out["E"].square().sum() + out["Fa"].square().sum()
        gradients = torch.autograd.grad(loss, [fields["Ra"], *model.parameters()], allow_unused=True)
        return out["E"], out["Fa"], gradients

    expected = values(reference, reference)
    assert expected[1].abs().max() > 0
    torch.testing.assert_close(values(torch.compile(target, backend="aot_eager"), target), expected)
    with pytest.raises((AssertionError, TorchRuntimeError), match="Replaced force coordinates"):
        torch.compile(target, backend="eager", fullgraph=True)(_batch(0.9))


def test_cuda_reduction_scope_uses_the_force_layers_coordinate_names():
    from torch._inductor import config
    from torch._subclasses.fake_tensor import FakeTensorMode
    from enerzyme.models.layers.compiled_grad import float64_cuda_reduction_scope
    from enerzyme.models.layers.gradient import ForceLayer

    if not hasattr(config.triton, "mix_order_reduction"):
        pytest.skip("Inductor has no mix-order reduction option")
    core = _schnet(1)
    force = next(layer for layer in core.post_sequence if type(layer) is ForceLayer)
    force.reset_field_name(Ra="pos")
    with FakeTensorMode(), config.patch({"triton.mix_order_reduction": True}):
        cuda = torch.empty(3, 3, device="cuda", dtype=torch.float64)
        cpu = torch.empty(3, 3, dtype=torch.float64)
        with float64_cuda_reduction_scope(core, {"pos": cuda}):
            assert not config.triton.mix_order_reduction
        with float64_cuda_reduction_scope(core, {"pos": cpu, "Ra": cuda}):
            assert config.triton.mix_order_reduction
        assert config.triton.mix_order_reduction


@pytest.mark.parametrize("ensemble, backend", [(1, "inductor"), (2, "aot_eager")])
def test_compiled_energy_force_and_parameter_gradients(ensemble, backend):
    torch.manual_seed(37)
    eager = _schnet(ensemble)
    target = copy.deepcopy(eager)
    compiled = torch.compile(target, fullgraph=True, backend=backend)
    # The second geometry brings an additional neighbor pair inside the cutoff.
    for separation in (0.4, 1.8):
        eager.zero_grad(set_to_none=True)
        target.zero_grad(set_to_none=True)
        expected = eager(_batch(separation))
        result = compiled(_batch(separation))
        assert set(result) == set(expected)
        assert result["N"] == [3]
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(result[key], value)
        # Each ensemble member has a distinct coordinate gradient.
        for out in (expected, result):
            (out["Fa"].square().sum() + 0.3 * out["E"].square().sum()).backward()
        for (name, p), (_, q) in zip(eager.named_parameters(), target.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(q.grad, p.grad, msg=name)

    eager.eval()
    compiled.eval()
    result = compiled(_batch(1.6))
    torch.testing.assert_close(result["Fa"], eager(_batch(1.6))["Fa"])
    assert not result["Fa"].requires_grad


def test_compiled_force_graph_batch():
    from torch_geometric.data import Data

    def graph():
        fields = _batch(1.1)
        fields["batch"] = fields.pop("batch_seg")
        return Data(**fields)

    eager = _schnet(1)
    compiled = torch.compile(copy.deepcopy(eager), backend="aot_eager", fullgraph=True)
    expected, actual = eager(graph()), compiled(graph())
    assert set(actual.keys()) == set(expected.keys())
    torch.testing.assert_close(actual.Fa, expected.Fa)


def test_inference_partitions_preserve_energy_gradients():
    from torch._functorch import config

    torch.manual_seed(37)
    eager = _schnet(2).eval()
    target = copy.deepcopy(eager)
    compiled = torch.compile(target, backend="inductor", dynamic=True)
    previous = getattr(config, "donated_buffer", None)
    compiler_config = getattr(torch.compiler, "config", None)
    tag = getattr(compiler_config, "cache_key_tag", None)
    expected, result = eager(_batch(1.1)), compiled(_batch(1.1))
    assert getattr(config, "donated_buffer", None) == previous
    assert getattr(compiler_config, "cache_key_tag", None) == tag
    torch.testing.assert_close(result["E"], expected["E"])
    torch.testing.assert_close(result["Fa"], expected["Fa"])
    assert not result["Fa"].requires_grad

    # After one VJP per ensemble head, the energy still supports an ordinary
    # parameter backward. Callers asking for another retained VJP can select
    # reusable buffers for that external backward as well.
    for out in (expected, result):
        reuse = nullcontext() if previous is None else config.patch(donated_buffer=False)
        with reuse:
            coordinate_grad, = torch.autograd.grad(out["E"].sum(), out["Ra"], retain_graph=True)
        torch.testing.assert_close(coordinate_grad, -out["Fa"].sum(dim=-1))
        out["E"].sum().backward()
    for (name, p), (_, q) in zip(eager.named_parameters(), target.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, msg=name)


def test_reusable_inference_uses_a_distinct_compiler_cache_tag():
    from torch._functorch import config

    compiler_config = getattr(torch.compiler, "config", None)
    if not (hasattr(config, "donated_buffer") and hasattr(compiler_config, "cache_key_tag")):
        pytest.skip("This PyTorch version does not expose reusable-buffer cache configuration")

    tags = set()

    def capture(graph, _inputs):
        if any(getattr(node.target, "__name__", "") == "linear" for node in graph.graph.nodes):
            tags.add(compiler_config.cache_key_tag)
        return graph.forward

    torch.compiler.reset()
    try:
        with compiler_config.patch(cache_key_tag="project-A"):
            torch.compile(_schnet(1).eval(), backend=capture)(_batch(1.1))
            assert compiler_config.cache_key_tag == "project-A"
        assert tags == {"project-A:enerzyme-retained-backward"}
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("architecture", ["visnet", "quinnet", "freecg", "visnet_lsrm"])
def test_compiled_force_and_vector_supervision_preserve_terminal_gradients(architecture):
    from torch._dynamo.backends.registry import lookup_backend
    from enerzyme.models.ff import get_ff_core
    from scripts.core_harness import make_batch, random_molecules

    torch.manual_seed(53)
    _, defaults, layers = get_ff_core(architecture)
    layers = copy.deepcopy(layers)
    for spec in layers:
        if spec["name"] == "Core":
            spec["params"].update(num_layers=2, num_heads=2)
            if architecture == "freecg":
                spec["params"]["num_groups"] = 2
    reduce_at = next(i for i, layer in enumerate(layers) if layer["name"] == "EnergyReduce")
    layers.insert(reduce_at, {"name": "EquivariantDipole", "params": {"n_layers": 1}})
    model = build_model(
        architecture, layers, {**defaults, "dim_embedding": 8, "num_rbf": 6, "max_Za": 8}, verbose=0
    ).double().train()
    batch = make_batch(random_molecules(107, [3, 2], density=0.12), torch.float64)
    batch["Ra"].requires_grad_()
    params = dict(model.named_parameters())
    dense_layers_per_graph = []

    def capture(graph, example):
        dense_layers_per_graph.append(sum(node.target is torch.nn.functional.linear for node in graph.graph.nodes))
        return lookup_backend("aot_eager")(graph, example)

    compiled = torch.compile(model, backend=capture)
    scalar_only = ("head.out_norm.", "post_sequence.0.head.")
    if architecture == "quinnet":
        # The final block's higher-body terms contribute only to the scalar update.
        scalar_only += tuple("blocks.1.module." + field for field in (
            "ratio", "f_proj", "vec_proj_improper", "dst_proj", "src_proj", "trg_proj",
        ))
    elif architecture == "visnet_lsrm":
        scalar_only += ("head.short.out_norm.", "head.out_norm_long.",
                        *(f"blocks.4.module.mix.{i}." for i in (2, 3, 4, 5)))

    def step(runner, target):
        result = runner(batch)
        loss = result["E"].square().sum() + result["Fa"].square().sum()
        vector_loss = result["atom_vector_feature"].square().sum()
        if "atom_l2_feature" in result:
            vector_loss = vector_loss + result["atom_l2_feature"].square().sum()
        if target == "vector":
            loss = vector_loss
        elif target == "forces_and_vector":
            loss = loss + vector_loss
        elif target == "forces_and_dipole":
            loss = loss + result["M2"].square().sum()
        gradients = torch.autograd.grad(loss, [batch["Ra"], *params.values()], allow_unused=True)
        return result, dict(zip(["Ra", *params], gradients))

    for target in ("forces", "forces_and_vector", "forces_and_dipole", "vector"):
        expected, reference = step(model, target)
        actual, gradients = step(compiled, target)
        assert expected.keys() == actual.keys()
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, atol=1e-10, rtol=1e-8)
        for name, value in reference.items():
            observed = gradients[name]
            if target == "vector" and value is None and observed is not None:
                # The shared prefix still returns energy and scalar fields when a
                # caller uses only raw vectors. AOT supplies zeros to these scalar heads.
                assert name.startswith(scalar_only)
                assert torch.count_nonzero(observed) == 0
            elif value is None:
                assert observed is None, name
            else:
                torch.testing.assert_close(observed, value, atol=1e-10, rtol=1e-8)
        if architecture == "visnet_lsrm":
            terminal = [grad for name, grad in gradients.items()
                        if name.startswith(("blocks.4.module.mlp_", "blocks.4.module.mix.0.", "blocks.4.module.mix.1."))]
            assert len(terminal) == 10
        else:
            terminal = [gradients["blocks.1.module.s_proj." + p] for p in ("weight", "bias")]
            if architecture == "freecg":
                terminal.append(gradients["blocks.1.module.group_cg.tp_weight"])
        if target == "forces":
            assert all(grad is None for grad in terminal)
        else:
            assert all(grad is not None and torch.count_nonzero(grad) for grad in terminal)
    vector_projections = 6 if architecture == "visnet_lsrm" else 1
    assert vector_projections in dense_layers_per_graph and max(dense_layers_per_graph) > vector_projections
