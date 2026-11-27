from copy import deepcopy

import pytest
import torch

muon = pytest.importorskip("muon")

from enerzyme.tasks.muon_batched import BatchedSingleDeviceMuonWithAuxAdam


def _optimizer(cls, device, dtype):
    shapes = [(12, 8), (12, 8), (8, 12), (2, 4, 3), (2, 4, 3), (8,), (), (8,)]
    params = [torch.nn.Parameter(torch.zeros(s, device=device, dtype=dtype)) for s in shapes]
    # Exercise mixed precision within the auxiliary parameter group.
    params[-1] = torch.nn.Parameter(params[-1].double())
    return params, cls([
        {"params": params[:5], "use_muon": True, "lr": 0.003, "weight_decay": 0.01},
        {"params": params[5:], "use_muon": False, "lr": 0.001},
    ])


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_muon_resume_and_optional_gradients(device, dtype):
    ps, ref = _optimizer(muon.SingleDeviceMuonWithAuxAdam, device, dtype)
    qs, batched = _optimizer(BatchedSingleDeviceMuonWithAuxAdam, device, dtype)
    gen = torch.Generator().manual_seed(221)
    for step in range(4):
        for i, (p, q) in enumerate(zip(ps, qs)):
            if (i, step) in {(0, 0), (1, 2), (5, 1), (6, 0), (7, 0)}:
                p.grad = q.grad = None
            else:
                grad = torch.randn(p.shape, dtype=p.dtype, generator=gen).to(device)
                p.grad, q.grad = grad, grad.clone()
        # The serial optimizer assumes every group entry has a gradient.
        groups = [g["params"] for g in ref.param_groups]
        for group in ref.param_groups:
            group["params"] = [p for p in group["params"] if p.grad is not None]
        ref.step()
        for group, params in zip(ref.param_groups, groups):
            group["params"] = params
        batched.step()
        for i, (p, q) in enumerate(zip(ps, qs)):
            # Batched and single bfloat16 matrix products can round differently.
            tol = 0.0002 if i < 5 else 1e-7
            torch.testing.assert_close(q, p, atol=tol, rtol=0)
            assert (p in ref.state) == (q in batched.state)
            if p in ref.state:
                torch.testing.assert_close(batched.state[q], ref.state[p])
        assert ref.state_dict()["param_groups"] == batched.state_dict()["param_groups"]
        if step == 1:
            # Switch implementations in both directions and compare the next steps.
            ref_state, batched_state = deepcopy(ref.state_dict()), deepcopy(batched.state_dict())
            batched.load_state_dict(ref_state)
            ref.load_state_dict(batched_state)


def test_batched_convolution_uses_original_gradient_aspect_ratio():
    shape = (2, 3, 8, 2)
    params = [torch.nn.Parameter(torch.zeros(shape)) for _ in range(2)]
    opt = BatchedSingleDeviceMuonWithAuxAdam([dict(params=params, use_muon=True, lr=0.01)])
    gen = torch.Generator().manual_seed(143)
    expected = []
    for p in params:
        p.grad = torch.randn(shape, generator=gen)
        update = muon.muon_update(p.grad.clone(), torch.zeros_like(p))
        expected.append(-0.01 * update.reshape_as(p).float())
    opt.step()
    for actual, target in zip(params, expected):
        torch.testing.assert_close(actual, target, rtol=0, atol=0.0002)


def test_batched_muon_selection_and_parameter_groups(monkeypatch):
    from enerzyme.tasks import optimizer

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 1)

    model = Core()
    monkeypatch.setitem(optimizer.MODEL_HEAD_NAMES, "Core", {"head"})
    name, config = optimizer.get_optimizer_config(Optimizer={"name": "Muon", "batched": True})
    assert type(optimizer.get_optimizer(name, model, {})) is muon.SingleDeviceMuonWithAuxAdam
    opt = optimizer.get_optimizer(name, model, config)
    assert type(opt) is BatchedSingleDeviceMuonWithAuxAdam
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(model.linear.weight)]
    assert set(map(id, opt.param_groups[1]["params"])) == set(map(id, [
        model.linear.bias, model.head.weight, model.head.bias,
    ]))
    monkeypatch.setattr(optimizer.dist, "is_initialized", lambda: True)
    assert type(optimizer.get_optimizer(name, model, config)) is muon.MuonWithAuxAdam


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_shape_on_distinct_devices():
    params = [torch.nn.Parameter(torch.zeros(8, 8, device=device)) for device in ("cpu", "cuda")]
    opt = BatchedSingleDeviceMuonWithAuxAdam([dict(params=params, use_muon=True, lr=0.01)])
    expected = []
    for p in params:
        p.grad = torch.eye(8, device=p.device)
        expected.append(muon.muon_update(p.grad.clone(), torch.zeros_like(p)).float().mul_(-0.01))
    opt.step()
    for p, target in zip(params, expected):
        torch.testing.assert_close(p, target)
