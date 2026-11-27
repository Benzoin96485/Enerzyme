"""Periodic pair offsets preserve a complete model's energy, force and mixed gradients."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from core_harness import build, forward


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required",
))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_schnet_force_training_across_cell_boundary(device, dtype):
    torch.manual_seed(7)
    model = build("schnet").to(device=device, dtype=dtype)
    positions = torch.tensor([[0.25, 0.5, 0], [7.5, 0.5, 0.25], [0.5, 1., 0.5]], device=device, dtype=dtype)
    shift = torch.zeros_like(positions)
    shift[1, 0] = -8
    i = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    j = torch.tensor([1, 2, 0, 2, 0, 1], device=device)
    batch = {"Za": torch.tensor([1, 8, 6], device=device), "batch_seg": torch.zeros_like(i[:3]),
             "cell": 8 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0), "idx_i": i, "idx_j": j}

    def evaluate(coords, offsets):
        model.zero_grad(set_to_none=True)
        out = forward(model, dict(batch, Ra=coords, offsets=offsets), dtype, train=True)
        (out["E"].square().sum() + out["Fa"].square().sum()).backward()
        grads = {k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}
        return out["E"].detach(), out["Fa"].detach(), grads

    wrapped = evaluate(positions, shift[j] - shift[i])
    unwrapped = evaluate(positions + shift, torch.zeros_like(shift[i]))
    assert unwrapped[1].abs().max() > 0
    assert wrapped[2].keys() == unwrapped[2].keys() and wrapped[2]
    tol = 3e-5 if dtype == torch.float32 else 1e-10
    for a, b in [(wrapped[0], unwrapped[0]), (wrapped[1], unwrapped[1]), *zip(wrapped[2].values(), unwrapped[2].values())]:
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
