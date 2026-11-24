"""The optional xPaiNN preset uses the same learned operators as XequiNet."""
import pytest
import torch
from torch_geometric.data import Data

from enerzyme.models.ff import build_model
from enerzyme.models.mp.core import ComposableCore
from enerzyme.models.presets import default_params


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph"])
def test_xpainn_upstream_energy_forces_and_force_loss(dtype, compiled):
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
        e, f = expected(Data(at_no=z, pos=pos, batch=batch, edge_index=torch.stack((i, j))))
        runner = torch.compile(actual, backend="eager", fullgraph=True) if compiled else actual
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            out = runner(dict(Za=z, Ra=pos, batch_seg=batch, idx_i=i, idx_j=j, N=[3, 2]))
        torch.testing.assert_close(out["E"], e.squeeze(-1))
        torch.testing.assert_close(out["Fa"], f)
        def loss(en, fo):
            return en.square().sum() + fo.square().sum()

        ref_grad, = torch.autograd.grad(loss(e, f), expected.embed.node_lin.weight)
        grad, = torch.autograd.grad(loss(out["E"], out["Fa"]), actual.embed.embedding.node_lin.weight)
        torch.testing.assert_close(grad, ref_grad)
    finally:
        torch.set_default_dtype(previous)
