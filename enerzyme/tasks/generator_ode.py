"""Generator (flow) inference: integrate dq/dt, ds/dt = v_theta(q, s, t) with torchdiffeq."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module


def generator_ode_predict_enabled(generator_config: Dict[str, Any] | None) -> bool:
    if generator_config is None:
        return False
    if isinstance(generator_config, dict) and generator_config.get("enabled") is False:
        return False
    return bool(generator_config.get("ode_predict", True))


def generator_predict_forward(
    model: Module,
    net_input: Dict[str, Tensor],
    generator_config: Dict[str, Any],
) -> Dict[str, Tensor]:
    """Integrate flow from t=0 (init per-atom q,s) to t=1; return dict with target field keys.

    Expects ``net_input`` to contain init and static fields (structure, total Q/S, batch, etc.).
    Eval collate should set ``Q_flow_a``/``S_flow_a`` to init and ``flow_t`` to 0; this routine
    overwrites them at each ODE evaluation.

    Hyperparameters (optional) on ``generator_config``:
        ``ode_method`` (default ``dopri5``), ``ode_rtol``, ``ode_atol``, ``ode_options``,
        ``ode_adjoint`` (default False; uses ``odeint_adjoint`` when True — for future train use).
    """
    from torchdiffeq import odeint_adjoint, odeint

    cfg = generator_config
    init_q_key = cfg.get("init_q_key", "Q_init_a")
    init_s_key = cfg.get("init_s_key", "S_init_a")
    out_q_key = cfg.get("out_q_key", "Q_flow_a")
    out_s_key = cfg.get("out_s_key", "S_flow_a")
    t_key = cfg.get("t_key", "flow_t")
    pred_q_key = cfg.get("target_q_key", "Qa")
    pred_s_key = cfg.get("target_s_key", "Sa")
    vel_q_key = cfg.get("vel_q_key", "Q_vel_a")
    vel_s_key = cfg.get("vel_s_key", "S_vel_a")

    batch_seg = net_input["batch_seg"]
    num_graphs = int(batch_seg.max().item()) + 1
    init_q = net_input[init_q_key]
    init_s = net_input[init_s_key]
    n = init_q.shape[0]
    device = init_q.device
    dtype = init_q.dtype

    y0 = torch.cat([init_q.reshape(-1), init_s.reshape(-1)], dim=0)

    method = cfg.get("ode_method", "dopri5")
    rtol = float(cfg.get("ode_rtol", 1e-5))
    atol = float(cfg.get("ode_atol", 1e-5))
    ode_options = cfg.get("ode_options")
    use_adjoint = bool(cfg.get("ode_adjoint", False))
    integrator = odeint_adjoint if use_adjoint else odeint

    def odefunc(t: Tensor, y: Tensor) -> Tensor:
        q = y[:n]
        s = y[n:]
        inp = dict(net_input)
        inp[out_q_key] = q
        inp[out_s_key] = s
        t_val = t.reshape(()).to(dtype=dtype)
        inp[t_key] = torch.full((num_graphs,), t_val, device=device, dtype=dtype)
        out = model(inp)
        vq = out[vel_q_key].reshape(-1)
        vs = out[vel_s_key].reshape(-1)
        return torch.cat([vq, vs], dim=0)

    t_span = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
    kwargs = {"rtol": rtol, "atol": atol, "method": method}
    if ode_options is not None:
        kwargs["options"] = ode_options
    traj = integrator(odefunc, y0, t_span, **kwargs)
    y1 = traj[-1]
    q1 = y1[:n]
    s1 = y1[n:]

    return {pred_q_key: q1, pred_s_key: s1}
