"""The optional xPaiNN preset uses the same learned operators as XequiNet."""
import pytest
import torch
from torch_geometric.data import Data

from enerzyme.models.ff import build_model
from enerzyme.models.mp.core import ComposableCore
from enerzyme.models.presets import default_params


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
@pytest.mark.parametrize("periodic", [False, True], ids=["molecule", "periodic"])
def test_xpainn_upstream_energy_forces_and_force_loss(dtype, compiled, periodic):
    upstream = pytest.importorskip("xequinet.nn.model")
    from xequinet.utils.config import NetConfig

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        params, layers = default_params("xpainn")
        options = next(layer["params"] for layer in layers if layer["name"] == "Core")
        torch.manual_seed(71)
        expected = upstream.XPaiNN(NetConfig(cutoff=params["cutoff_sr"], **options)).train()
        torch.manual_seed(71)
        actual = build_model("xpainn", layers, params, verbose=0).train()
        assert isinstance(actual, ComposableCore) and len(actual.blocks) == options["action_blocks"]
        z = torch.tensor([1, 6, 8, 7, 1])
        batch = torch.tensor([0, 0, 0, 1, 1])
        i, j = torch.where((batch[:, None] == batch) & ~torch.eye(5, dtype=torch.bool))
        pos = torch.randn(5, 3, requires_grad=True)
        data = Data(at_no=z, pos=pos, batch=batch, edge_index=torch.stack((i, j)))
        inputs = dict(Za=z, Ra=pos, batch_seg=batch, idx_i=i, idx_j=j, N=[3, 2])
        if periodic:
            offsets = pos.new_zeros((i.numel(), 3))
            offsets[:, 0] = (i - j).sign() * 2.3
            data.shifts = inputs["offsets"] = offsets
        e, f = expected(data)
        runner = torch.compile(actual, backend="eager", fullgraph=True) if compiled else actual
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            out = runner(inputs)
        torch.testing.assert_close(out["E"], e.squeeze(-1))
        torch.testing.assert_close(out["Fa"], f)
        def loss(en, fo):
            return en.square().sum() + fo.square().sum()

        pairs = [(expected.embed, actual.embed.embedding), (expected.out, actual.head.output)]
        for message, update, block in zip(expected.message, expected.update, actual.blocks):
            pairs.extend([(message, block.message), (update, block.update)])
        reference_inputs = [pos]
        actual_inputs = [pos]
        for source, target in pairs:
            reference = dict(source.named_parameters())
            weights = dict(target.named_parameters())
            assert weights.keys() == reference.keys()
            reference_inputs.extend(reference.values())
            actual_inputs.extend(weights[name] for name in reference)
        assert len(actual_inputs) == 1 + len(list(actual.parameters()))
        expected_grads = torch.autograd.grad(loss(e, f), reference_inputs)
        actual_grads = torch.autograd.grad(loss(out["E"], out["Fa"]), actual_inputs)
        for gradient, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(gradient, reference)
    finally:
        torch.set_default_dtype(previous)


def test_xpainn_generated_contractions_feed_the_output():
    pytest.importorskip("xequinet.nn.model")
    params, layers = default_params("xpainn")
    core = build_model("xpainn", layers, params, verbose=0)
    contractions = [
        node for kernel in core.modules() if isinstance(kernel, torch.fx.GraphModule)
        for node in kernel.graph.nodes if node.op == "call_function" and node.target is torch.einsum
    ]
    assert all(node.users for node in contractions)
