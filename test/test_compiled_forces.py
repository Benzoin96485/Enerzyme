"""Conservative force training through torch.compile's joint forward/backward."""

import copy
from contextlib import nullcontext

import pytest
import torch

from enerzyme.models.ff import build_model


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


def _batch(distance):
    return {
        "Ra": torch.tensor([[0., 0., 0.], [distance, .1, -.2], [2.7, -.4, .3]],
                           dtype=torch.float64, requires_grad=True),
        "Za": torch.tensor([1, 7, 8]), "batch_seg": torch.zeros(3, dtype=torch.long), "N": [3],
        "idx_i": torch.tensor([0, 0, 1, 1, 2, 2]), "idx_j": torch.tensor([1, 2, 0, 2, 0, 1]),
    }


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
