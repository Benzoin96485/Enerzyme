"""Single-device Muon with shape-batched Newton–Schulz and foreach Adam updates."""
from collections import defaultdict

import torch
from muon import SingleDeviceMuonWithAuxAdam, muon_update, zeropower_via_newtonschulz5


def _chunks(params):
    # Bound temporary stacks and Adam intermediates independently of model size.
    chunk, elements = [], 0
    for p in params:
        if chunk and elements + p.numel() > 1024 ** 2:
            yield chunk
            chunk, elements = [], 0
        chunk.append(p)
        elements += p.numel()
    if chunk:
        yield chunk


class BatchedSingleDeviceMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    """Retain muon-optimizer's parameter groups and per-parameter checkpoint state."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    key = (p.device, p.dtype, p.shape)
                else:
                    if not state:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    key = (p.device, p.dtype, state["step"])
                buckets[key].append(p)
            for params in buckets.values():
                for chunk in _chunks(params):
                    if group["use_muon"]:
                        self._step_muon(chunk, group)
                    else:
                        self._step_adam(chunk, group)
        return loss

    def _step_muon(self, params, group):
        grads = [p.grad for p in params]
        buffers = [self.state[p]["momentum_buffer"] for p in params]
        beta = group["momentum"]
        if len(params) == 1:
            update = muon_update(grads[0], buffers[0], beta=beta)
            updates = [update.reshape_as(params[0])]
        else:
            torch._foreach_lerp_(buffers, grads, 1 - beta)
            torch._foreach_lerp_(grads, buffers, beta)
            stacked = torch.stack(grads)
            # muon-optimizer flattens each convolution filter before Newton–Schulz
            # but scales the result by the original gradient's last two dimensions.
            if params[0].ndim == 4:
                stacked = stacked.flatten(2)
            update = zeropower_via_newtonschulz5(stacked, steps=5)
            shape = params[0].shape
            update.mul_(max(1, shape[-2] / shape[-1]) ** 0.5)
            updates = update.to(params[0].dtype).reshape(len(params), *shape).unbind()
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, updates, alpha=-group["lr"])

    def _step_adam(self, params, group):
        grads = [p.grad for p in params]
        avgs = [self.state[p]["exp_avg"] for p in params]
        squares = [self.state[p]["exp_avg_sq"] for p in params]
        beta1, beta2 = group["betas"]
        step = self.state[params[0]]["step"]
        torch._foreach_lerp_(avgs, grads, 1 - beta1)
        torch._foreach_lerp_(squares, torch._foreach_mul(grads, grads), 1 - beta2)
        numerator = torch._foreach_div(avgs, 1 - beta1 ** step)
        denom = torch._foreach_div(squares, 1 - beta2 ** step)
        torch._foreach_sqrt_(denom)
        torch._foreach_add_(denom, group["eps"])
        torch._foreach_div_(numerator, denom)
        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, numerator, alpha=-group["lr"])
