"""Minimal flow-matching integration checks: velocity conservation, one train step, ODE predict."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module

from enerzyme.models.layers.electrostatics import VelocityConservationLayer
from enerzyme.models.loss import CFMLoss
from enerzyme.tasks.batch import _decorate_batch_input
from enerzyme.tasks.generator_ode import generator_predict_forward


def test_velocity_conservation_layer_zero_sum_per_graph():
    """After VelocityConservation, per-graph sums of Q_vel_a and S_vel_a are zero."""
    layer = VelocityConservationLayer()
    # graph 0: 3 atoms; graph 1: 2 atoms
    batch_seg = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    qv = torch.tensor([1.0, -2.0, 5.0, 0.5, -0.25], dtype=torch.float64)
    sv = torch.tensor([-1.0, 0.0, 3.0, 2.0, 4.0], dtype=torch.float64)
    out = layer(
        {"batch_seg": batch_seg, "Q_vel_a": qv.clone(), "S_vel_a": sv.clone()}
    )
    qo, so = out["Q_vel_a"], out["S_vel_a"]
    for g in (0, 1):
        mask = batch_seg == g
        assert qo[mask].sum().abs() < 1e-12
        assert so[mask].sum().abs() < 1e-12


def _make_single_graph_batch(n: int):
    return [
        (
            {
                "N": n,
                "Za": list(range(6, 6 + n)),
                "Ra": [[0.0, 0.0, float(i)] for i in range(n)],
                "Q_init_a": [0.2 * (i + 1) for i in range(n)],
                "S_init_a": [0.1 * i for i in range(n)],
            },
            {
                "Qa": [0.5 + 0.1 * i for i in range(n)],
                "Sa": [-0.25 + 0.05 * i for i in range(n)],
            },
            "k0",
        )
    ]


class _LearnableVelocityModel(Module):
    """Scalar velocities per atom; one SGD step should reduce CFM loss."""

    def __init__(self, n: int, dtype=torch.float64):
        super().__init__()
        self.pq = torch.nn.Parameter(torch.zeros(n, dtype=dtype))
        self.ps = torch.nn.Parameter(torch.zeros(n, dtype=dtype))

    def forward(self, inp):
        _ = inp  # structure unused
        return {"Q_vel_a": self.pq, "S_vel_a": self.ps}


def test_one_generator_training_step_cfm():
    """One forward + CFMLoss + backward + optimizer step runs without NaN and updates parameters."""
    torch.manual_seed(0)
    n = 6
    gen = {"enabled": True}
    batch = _make_single_graph_batch(n)
    bf, bt = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, gen, generator_training=True
    )
    model = _LearnableVelocityModel(n, dtype=torch.float64)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    opt.zero_grad(set_to_none=True)
    out = model(bf)
    loss0 = CFMLoss(weight_q=1.0, weight_s=1.0)(out, bt)
    assert torch.isfinite(loss0).all()
    loss0.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    out1 = model(bf)
    loss1 = CFMLoss(weight_q=1.0, weight_s=1.0)(out1, bt)
    assert torch.isfinite(loss1).all()
    assert loss1.item() < loss0.item()


def _spatial_vel_model(vq: float, vs: float, dtype, device) -> Module:
    class M(Module):
        def forward(self, inp: dict) -> dict[str, Tensor]:
            m = inp["Za"].shape[0]
            return {
                "Q_vel_a": torch.full((m,), vq, dtype=dtype, device=device),
                "S_vel_a": torch.full((m,), vs, dtype=dtype, device=device),
            }

    return M()


def test_generator_ode_predict_two_graphs_no_nan():
    """Integrate ODE for a batched dict input (two molecules); outputs are finite and match batch_seg."""
    gen = {
        "enabled": True,
        "ode_method": "euler",
        "ode_options": {"step_size": 0.25},
    }
    batch = [
        (
            {
                "N": 3,
                "Za": [6, 7, 8],
                "Ra": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "Q_init_a": [0.0, 0.0, 0.0],
                "S_init_a": [1.0, 1.0, 1.0],
            },
            {"Qa": [0.3, 0.3, 0.3], "Sa": [0.0, 0.0, 0.0]},
            "a",
        ),
        (
            {
                "N": 2,
                "Za": [6, 6],
                "Ra": [[2.0, 0.0, 0.0], [2.5, 1.0, 0.0]],
                "Q_init_a": [0.5, 0.5],
                "S_init_a": [0.25, 0.25],
            },
            {"Qa": [0.1, -0.1], "Sa": [0.5, 0.5]},
            "b",
        ),
    ]
    bf, _bt = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, gen, generator_training=False
    )
    m = _spatial_vel_model(0.08, -0.04, torch.float64, torch.device("cpu"))
    out = generator_predict_forward(m, bf, gen)
    q, s = out["Qa"], out["Sa"]
    assert q.shape == (5,) and s.shape == (5,)
    assert torch.isfinite(q).all() and torch.isfinite(s).all()
    iq = bf["Q_init_a"]
    is_ = bf["S_init_a"]
    torch.testing.assert_close(q, iq + 0.08)
    torch.testing.assert_close(s, is_ - 0.04)
