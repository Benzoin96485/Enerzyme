"""MACE's optional FX kernels preserve the eager checkpoint and its derivatives."""
import copy

import pytest
import torch

from enerzyme.models.ff import build_model, get_ff_core


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mace_fx_checkpoint_sparse_edges_and_derivatives(dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        _, defaults, layer_params = get_ff_core("mace")
        params = {**defaults, "dim_embedding": 4, "num_rbf": 3, "max_Za": 8, "cutoff_sr": 2.0}
        layers = copy.deepcopy(layer_params)
        core = next(layer["params"] for layer in layers if layer["name"] == "Core")
        core.update(additional_hidden_irreps="4x1o", radial_MLP=[8], MLP_irreps="4x0e")
        torch.manual_seed(61)
        reference = build_model("mace", layers, params, verbose=0)
        model = build_model("mace", layers, {**params, "e3nn_codegen": "fx"}, verbose=0)
        assert set(model.state_dict()) == set(reference.state_dict())
        model.load_state_dict(reference.state_dict(), strict=True)
        model = copy.deepcopy(model)
        compiled = torch.compile(model, fullgraph=True, backend="eager")
        features = dict(
            Za=torch.tensor([1, 8]), batch_seg=torch.zeros(2, dtype=torch.long),
            N=[2],
        )
        for pair_count, distance in ((1, 1.1), (1, 3.0), (0, 1.1)):
            # Exercise a directed pair, a filtered pair, and an empty input list.
            features["idx_i"] = torch.tensor([0], dtype=torch.long)[:pair_count]
            features["idx_j"] = torch.tensor([1], dtype=torch.long)[:pair_count]
            positions = torch.tensor([[0.1, -0.2, 0.2], [distance, 0.1, -0.3]], requires_grad=True)
            expected = reference({**features, "Ra": positions})
            actual = compiled({**features, "Ra": positions})
            for field in ("E", "Fa"):
                torch.testing.assert_close(actual[field], expected[field])
            for net, outputs in ((reference, expected), (model, actual)):
                net.zero_grad(set_to_none=True)
                (outputs["E"].square().sum() + outputs["Fa"].square().sum()).backward()
            for (name, p), (reference_name, q) in zip(model.named_parameters(), reference.named_parameters()):
                assert name == reference_name
                if q.grad is None:
                    if p.grad is not None:
                        # A traced cutoff can empty a previously nonempty edge batch.
                        assert pair_count == 1 and distance > params["cutoff_sr"]
                        torch.testing.assert_close(p.grad, torch.zeros_like(p.grad), atol=0, rtol=0)
                    continue
                assert p.grad is not None
                torch.testing.assert_close(p.grad, q.grad, msg=name)

            def force_jacobian(net):
                return torch.autograd.functional.jacobian(
                    lambda xyz: net({**features, "Ra": xyz})["Fa"], positions,
                )

            torch.testing.assert_close(force_jacobian(compiled), force_jacobian(reference))
    finally:
        torch.set_default_dtype(old_dtype)
