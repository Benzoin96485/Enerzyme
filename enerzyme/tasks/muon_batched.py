'''
Single-device Muon with auxiliary AdamW (muon-optimizer's SingleDeviceMuonWithAuxAdam)
with the same update rule and optimizer state, but batched over parameters: Newton-Schulz
orthogonalization runs once per group of equally shaped matrices and the elementwise
updates use foreach kernels, which removes most of the per-parameter kernel launches.
'''
from collections import defaultdict
from typing import Dict, List
import torch
from torch import Tensor
from muon import SingleDeviceMuonWithAuxAdam, zeropower_via_newtonschulz5


class BatchedSingleDeviceMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    ns_steps = 5

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            if group["use_muon"]:
                self._muon_step(group, params)
            else:
                self._adam_step(group, params)

    def _muon_step(self, group: Dict, params: List[Tensor]) -> None:
        grads = [p.grad for p in params]
        bufs = []
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p)
            bufs.append(state["momentum_buffer"])
        beta = group["momentum"]
        # momentum.lerp_(grad, 1 - beta); nesterov update = grad.lerp_(momentum, beta)
        torch._foreach_lerp_(bufs, grads, 1 - beta)
        torch._foreach_lerp_(grads, bufs, beta)
        by_shape = defaultdict(list)
        for i, g in enumerate(grads):
            by_shape[(tuple(g.shape), g.dtype)].append(i)
        updates: List[Tensor] = [None] * len(params)
        for idx in by_shape.values():
            stacked = torch.stack([grads[i].view(len(grads[i]), -1) if grads[i].ndim == 4 else grads[i] for i in idx])
            ortho = zeropower_via_newtonschulz5(stacked, steps=self.ns_steps)
            ortho *= max(1, stacked.size(-2) / stacked.size(-1)) ** 0.5
            ortho = ortho.to(params[idx[0]].dtype)
            for k, i in enumerate(idx):
                updates[i] = ortho[k].reshape(params[i].shape)
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, updates, alpha=-group["lr"])

    def _adam_step(self, group: Dict, params: List[Tensor]) -> None:
        beta1, beta2 = group["betas"]
        by_step = defaultdict(list)
        for p in params:
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = 0
            state["step"] += 1
            by_step[state["step"]].append(p)
        for step, ps in by_step.items():
            grads = [p.grad for p in ps]
            exp_avgs = [self.state[p]["exp_avg"] for p in ps]
            exp_avg_sqs = [self.state[p]["exp_avg_sq"] for p in ps]
            torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
            torch._foreach_lerp_(exp_avg_sqs, torch._foreach_mul(grads, grads), 1 - beta2)
            m = torch._foreach_div(exp_avgs, 1 - beta1 ** step)
            v = torch._foreach_div(exp_avg_sqs, 1 - beta2 ** step)
            torch._foreach_sqrt_(v)
            torch._foreach_add_(v, group["eps"])
            torch._foreach_div_(m, v)
            torch._foreach_mul_(ps, 1 - group["lr"] * group["weight_decay"])
            torch._foreach_add_(ps, m, alpha=-group["lr"])
