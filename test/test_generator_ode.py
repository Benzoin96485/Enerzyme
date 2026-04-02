"""Tests for Generator ODE prediction (torchdiffeq)."""

import torch
from torch.nn import Module

from enerzyme.tasks.batch import _decorate_batch_input
from enerzyme.tasks.generator_ode import (
    generator_ode_predict_enabled,
    generator_predict_forward,
)
from enerzyme.models.loss import CFMLoss


class _ConstantVelModel(Module):
    """Returns fixed per-atom velocities (batch ignored)."""

    def __init__(self, vq: float, vs: float, n: int):
        super().__init__()
        self.register_buffer("_vq", torch.full((n,), vq))
        self.register_buffer("_vs", torch.full((n,), vs))

    def forward(self, inp):
        return {"Q_vel_a": self._vq, "S_vel_a": self._vs}


def test_generator_predict_forward_endpoint():
    n = 5
    m = _ConstantVelModel(0.1, -0.3, n)
    gen = {
        "enabled": True,
        "ode_method": "euler",
        "ode_options": {"step_size": 0.2},
    }
    batch = [
        (
            {
                "N": n,
                "Za": list(range(6, 6 + n)),
                "Ra": [[0.0, 0.0, float(i)] for i in range(n)],
                "Q_init_a": [0.0] * n,
                "S_init_a": [1.0] * n,
            },
            {"Qa": [0.5] * n, "Sa": [0.0] * n},
            "k0",
        )
    ]
    bf, bt = _decorate_batch_input(
        batch, torch.float64, torch.device("cpu"), True, gen, generator_training=False
    )
    out = generator_predict_forward(m, bf, gen)
    torch.testing.assert_close(
        out["Qa"], torch.full((n,), 0.1, dtype=torch.float64), rtol=1e-7, atol=1e-7
    )
    torch.testing.assert_close(
        out["Sa"], torch.full((n,), 0.7, dtype=torch.float64), rtol=1e-7, atol=1e-7
    )


def test_cfm_loss_endpoint_branch():
    loss_fn = CFMLoss(weight_q=1.0, weight_s=1.0)
    out = {
        "Qa": torch.tensor([0.0, 2.0], dtype=torch.float64),
        "Sa": torch.tensor([1.0, 1.0], dtype=torch.float64),
    }
    tgt = {
        "Qa": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "Sa": torch.tensor([1.0, 3.0], dtype=torch.float64),
        "Q_init_a": torch.zeros(2, dtype=torch.float64),
        "S_init_a": torch.zeros(2, dtype=torch.float64),
    }
    l = loss_fn(out, tgt)
    # mean Q err^2 = 0.5, mean S err^2 = 2.0
    torch.testing.assert_close(l, torch.tensor(2.5, dtype=torch.float64))


def test_generator_ode_predict_enabled():
    assert generator_ode_predict_enabled(None) is False
    assert generator_ode_predict_enabled({"enabled": False}) is False
    assert generator_ode_predict_enabled({"enabled": True}) is True
    assert generator_ode_predict_enabled({"ode_predict": False}) is False
