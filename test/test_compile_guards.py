"""Compiled pipelines must not recompile when called again on the same inputs."""

import torch
import torch._dynamo


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
