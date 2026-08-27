"""Diagnostic residual training: zero / x-only / Stage A, three seeds, three slices."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from ..data.dc_uma_cpcm.constants import (
    DEFAULT_EXPERIMENTS,
    DIAGNOSTIC_BASELINES,
    DIAGNOSTIC_HUBER_DELTA,
    DIAGNOSTIC_LR,
    DIAGNOSTIC_MAX_EPOCHS,
    DIAGNOSTIC_MICROBATCH,
    DIAGNOSTIC_PATIENCE,
    DIAGNOSTIC_SEEDS,
    DIAGNOSTIC_WEIGHT_DECAY,
)
from ..data.dc_uma_cpcm.diagnostic import (
    DEFAULT_DIAGNOSTIC_RUNS,
    decide_go_nogo,
    metrics_from_predictions,
    split_records,
    subgroup_tables,
)
from ..data.datatype import validate_dielectric_tensor
from ..models.esen.cpcm import ADAPTER_MODES, STAGE_B_LOG_KEYS, energy_gradient_forces, load_stage_a_adapter_weights
from ..models.loss import ResidualHuberLoss

N_ATOMS_MICROBENCH = (1, 6, 15, 30, 45, 60)


class ResidualSampleDataset(Dataset):
    """In-memory residual C-PCM samples for the diagnostic trainer."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(rec) for rec in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def collate_residual_batch(samples: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    ras = []
    zas = []
    segs = []
    charges = []
    spins = []
    diel = []
    dE = []
    dFa = []
    e_cpcm = []
    f_cpcm = []
    qa_list = []
    qa_ok = True
    for i, rec in enumerate(samples):
        pos = np.asarray(rec["positions"], dtype=np.float32)
        num = np.asarray(rec["numbers"], dtype=np.int64)
        ras.append(pos)
        zas.append(num)
        segs.append(np.full(len(num), i, dtype=np.int64))
        charges.append(int(rec["charge"]))
        spins.append(int(rec["spin"]) - 1)
        diel.append(float(rec["dielectric"]))
        dE.append(float(rec["dE"]))
        dFa.append(np.asarray(rec["dFa"], dtype=np.float32))
        e_cpcm.append(float(rec["E_cpcm"]))
        f_cpcm.append(np.asarray(rec["F_cpcm"], dtype=np.float32))
        lowdin = rec.get("lowdin_charges")
        if qa_ok and lowdin is not None:
            arr = np.asarray(lowdin, dtype=np.float32).reshape(-1)
            if arr.shape[0] == len(num) and np.isfinite(arr).all():
                qa_list.append(arr)
            else:
                qa_ok = False
        else:
            qa_ok = False
    ra = torch.from_numpy(np.concatenate(ras, axis=0)).to(device=device)
    ra.requires_grad_(True)
    batch = {
        "Ra": ra,
        "Za": torch.from_numpy(np.concatenate(zas, axis=0)).to(device=device),
        "batch_seg": torch.from_numpy(np.concatenate(segs, axis=0)).to(device=device),
        "Q": torch.tensor(charges, dtype=torch.long, device=device),
        "S": torch.tensor(spins, dtype=torch.long, device=device),
        "dielectric": validate_dielectric_tensor(torch.tensor(diel, dtype=torch.float32, device=device)),
        "dE": torch.tensor(dE, dtype=torch.float32, device=device),
        "dFa": torch.from_numpy(np.concatenate(dFa, axis=0)).to(device=device),
        "E_cpcm": torch.tensor(e_cpcm, dtype=torch.float32, device=device),
        "F_cpcm": torch.from_numpy(np.concatenate(f_cpcm, axis=0)).to(device=device),
        "records": list(samples),
        "N": torch.tensor([len(np.asarray(s["numbers"])) for s in samples], dtype=torch.long, device=device),
    }
    if qa_ok and qa_list:
        batch["Qa"] = torch.from_numpy(np.concatenate(qa_list, axis=0)).to(device=device)
    return batch


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    if hasattr(model, "adapter_parameters"):
        return list(model.adapter_parameters())
    adapter = getattr(model, "adapter", None)
    if adapter is None:
        return [param for param in model.parameters() if param.requires_grad]
    return [param for param in adapter.parameters() if param.requires_grad]


def _forward(model: nn.Module, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    out = model.get_output(
        Ra=batch["Ra"],
        Za=batch["Za"],
        batch_seg=batch["batch_seg"],
        Q=batch["Q"],
        S=batch["S"],
        dielectric=batch["dielectric"],
    )
    if "batch_seg" not in out:
        out = dict(out)
        out["batch_seg"] = batch["batch_seg"]
        out["N"] = batch["N"]
    return out


def recommend_accumulate_grad_batches(
    peak_mem_gb: float | None,
    *,
    device_mem_gb: float | None = None,
    microbatch: int = DIAGNOSTIC_MICROBATCH,
    headroom: float = 0.45,
) -> int:
    """Pick gradient accumulation from a measured second-order step; never assume 32."""
    if peak_mem_gb is None or peak_mem_gb <= 0:
        return 1
    if device_mem_gb is None or device_mem_gb <= 0:
        return 1
    usable = float(device_mem_gb) * float(headroom)
    max_micro = max(1, int(usable // float(peak_mem_gb)))
    # Effective batch = microbatch * accum. Keep it small on the 128-label set.
    target_eff = min(8, max_micro)
    return max(1, target_eff // max(int(microbatch), 1))


def microbenchmark_second_order(
    model: nn.Module,
    *,
    n_atoms_list: Sequence[int] = N_ATOMS_MICROBENCH,
    microbatch: int = DIAGNOSTIC_MICROBATCH,
    device: torch.device | None = None,
    repeats: int = 2,
) -> dict[str, Any]:
    """Peak memory and seconds/step for adapter force training (create_graph=True)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    rows = []
    loss_fn = ResidualHuberLoss(delta=DIAGNOSTIC_HUBER_DELTA, dE=1.0, dFa=1.0)
    for n_atoms in n_atoms_list:
        numbers = [6] * max(int(n_atoms) - 1, 0) + ([1] if int(n_atoms) > 0 else [])
        if not numbers:
            numbers = [1]
        n = len(numbers)
        pos = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            pos[i, 0] = 1.1 * i
        rec = {
            "numbers": numbers,
            "positions": pos.tolist(),
            "charge": 0,
            "spin": 1,
            "dielectric": 80.0,
            "dE": 0.1,
            "dFa": np.zeros((n, 3), dtype=np.float32).tolist(),
            "E_cpcm": 0.1,
            "F_cpcm": np.zeros((n, 3), dtype=np.float32).tolist(),
        }
        samples = [rec for _ in range(max(int(microbatch), 1))]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        times = []
        for _ in range(max(int(repeats), 1)):
            batch = collate_residual_batch(samples, device)
            params = adapter_parameters(model)
            if params:
                for param in params:
                    if param.grad is not None:
                        param.grad = None
            t0 = time.perf_counter()
            out = _forward(model, batch)
            loss = loss_fn(out, {"dE": batch["dE"], "dFa": batch["dFa"], "N": batch["N"]})
            if params:
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak = None
        if device.type == "cuda":
            peak = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        rows.append(
            {
                "n_atoms": n,
                "microbatch": int(microbatch),
                "seconds_per_step": float(np.median(times)),
                "peak_mem_gb": peak,
            }
        )
    peaks = [row["peak_mem_gb"] for row in rows if row["peak_mem_gb"] is not None]
    device_mem = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_mem = float(props.total_memory / (1024**3))
    recommend = recommend_accumulate_grad_batches(
        max(peaks) if peaks else None,
        device_mem_gb=device_mem,
        microbatch=microbatch,
    )
    return {
        "schema": "dc_uma_cpcm_microbench_v1",
        "device": str(device),
        "device_mem_gb": device_mem,
        "microbatch": int(microbatch),
        "rows": rows,
        "recommend_accumulate_grad_batches": recommend,
        "note": "Do not assume effective batch 32; accumulation follows measured peak memory.",
    }


def _mean_log_components(out: Mapping[str, Tensor]) -> dict[str, float]:
    logged = {}
    for key in STAGE_B_LOG_KEYS:
        if key in out and torch.is_tensor(out[key]):
            logged[key] = float(out[key].detach().mean().cpu())
    return logged


def _lowdin_aux_loss(out: Mapping[str, Tensor], batch: Mapping[str, Tensor], weight: float) -> Tensor | None:
    if weight == 0 or "Qa" not in out or batch.get("Qa") is None:
        return None
    pred = out["Qa"]
    tgt = batch["Qa"].to(device=pred.device, dtype=pred.dtype)
    if pred.shape[0] != tgt.shape[0]:
        return None
    return float(weight) * torch.nn.functional.mse_loss(pred, tgt)


def evaluate_records(model: nn.Module, records: Sequence[Mapping[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    pred_dE = []
    pred_dFa = []
    pred_E = []
    pred_Fa = []
    comp_sums = {k: 0.0 for k in STAGE_B_LOG_KEYS}
    comp_n = 0
    was_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        for rec in records:
            batch = collate_residual_batch([rec], device)
            out = _forward(model, batch)
            pred_dE.append(float(out["dE"].detach().cpu().numpy().reshape(-1)[0]))
            pred_dFa.append(out["dFa"].detach().cpu().numpy())
            pred_E.append(float(out["E"].detach().cpu().numpy().reshape(-1)[0]))
            pred_Fa.append(out["Fa"].detach().cpu().numpy())
            logged = _mean_log_components(out)
            if logged:
                for key, value in logged.items():
                    comp_sums[key] += value
                comp_n += 1
    finally:
        torch.set_grad_enabled(was_grad)
    packed = metrics_from_predictions(records, pred_dE, pred_dFa, pred_E=pred_E, pred_Fa=pred_Fa)
    packed["slices"] = subgroup_tables(packed["rows"])
    if comp_n:
        packed["energy_components"] = {k: v / comp_n for k, v in comp_sums.items()}
        packed["summary"]["energy_components"] = packed["energy_components"]
    return packed


def train_baseline(
    model: nn.Module,
    train_records: Sequence[Mapping[str, Any]],
    valid_records: Sequence[Mapping[str, Any]],
    *,
    baseline: str,
    seed: int,
    device: torch.device | None = None,
    out_dir: Path | str | None = None,
    lr: float = DIAGNOSTIC_LR,
    max_epochs: int = DIAGNOSTIC_MAX_EPOCHS,
    patience: int = DIAGNOSTIC_PATIENCE,
    batch_size: int = DIAGNOSTIC_MICROBATCH,
    accumulate_grad_batches: int = 1,
    huber_delta: float = DIAGNOSTIC_HUBER_DELTA,
    weight_decay: float = DIAGNOSTIC_WEIGHT_DECAY,
    max_steps: int | None = None,
    eval_splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    warm_start: Path | str | None = None,
    lowdin_aux_weight: float = 0.0,
) -> dict[str, Any]:
    baseline = str(baseline).strip().lower().replace("-", "_")
    if baseline not in DIAGNOSTIC_BASELINES and baseline not in ADAPTER_MODES:
        raise ValueError(f"unknown baseline {baseline!r}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(int(seed))
    model = model.to(device)
    warm_info = None
    if warm_start is not None and hasattr(model, "adapter"):
        warm_info = load_stage_a_adapter_weights(model.adapter, warm_start)
    aux_weight = float(lowdin_aux_weight)
    if baseline != "stage_b":
        aux_weight = 0.0
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_DIAGNOSTIC_RUNS / f"{baseline}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_fn = ResidualHuberLoss(delta=huber_delta, dE=1.0, dFa=1.0)
    params = adapter_parameters(model)
    n_train = len(train_records)
    micro = max(int(batch_size), 1)
    accum = max(int(accumulate_grad_batches), 1)
    steps_per_epoch = max(1, math.ceil(n_train / micro))
    opt_steps_per_epoch = max(1, math.ceil(steps_per_epoch / accum))
    if max_steps is None:
        max_steps = int(max_epochs) * opt_steps_per_epoch
    history = []
    best_score = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    n_steps = 0
    trained = False
    if params and baseline != "zero" and n_train:
        trained = True
        optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
        loader = DataLoader(
            ResidualSampleDataset(train_records),
            batch_size=micro,
            shuffle=True,
            collate_fn=lambda batch: collate_residual_batch(batch, device),
            drop_last=False,
        )
        for epoch in range(int(max_epochs)):
            model.train()
            epoch_loss = []
            epoch_comp = {k: [] for k in STAGE_B_LOG_KEYS}
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                out = _forward(model, batch)
                loss = loss_fn(out, {"dE": batch["dE"], "dFa": batch["dFa"], "N": batch["N"]})
                aux = _lowdin_aux_loss(out, batch, aux_weight)
                if aux is not None:
                    loss = loss + aux
                (loss / accum).backward()
                epoch_loss.append(float(loss.detach().cpu()))
                for key, value in _mean_log_components(out).items():
                    epoch_comp[key].append(value)
                boundary = ((i + 1) % accum == 0) or ((i + 1) == len(loader))
                if boundary:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    n_steps += 1
                    if n_steps >= int(max_steps):
                        break
            val = evaluate_records(model, valid_records, device) if valid_records else {"slices": {}}
            score = (val.get("slices") or {}).get("held_out_parent", {}).get("joint")
            if score is None:
                score = (val.get("summary") or {}).get("joint")
            score = float("inf") if score is None else float(score)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(epoch_loss)) if epoch_loss else None,
                    "val_joint": score,
                    "n_steps": n_steps,
                    **{
                        k: float(np.mean(v))
                        for k, v in epoch_comp.items()
                        if v
                    },
                }
            )
            if score + 1.0e-12 < best_score:
                best_score = score
                best_epoch = epoch
                wait = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.adapter.state_dict().items()
                }
            else:
                wait += 1
            if wait >= int(patience) or n_steps >= int(max_steps):
                break
        if best_state is not None:
            model.adapter.load_state_dict(best_state)
    val_metrics = evaluate_records(model, valid_records, device) if valid_records else {"rows": [], "summary": {}, "slices": {}}
    train_metrics = evaluate_records(model, train_records, device) if train_records else {"rows": [], "summary": {}, "slices": {}}
    extra_metrics = {}
    for name, rows in (eval_splits or {}).items():
        if rows:
            packed = evaluate_records(model, rows, device)
            extra_metrics[name] = packed["summary"]
            extra_metrics[f"{name}_slices"] = packed.get("slices") or {}
            if packed.get("energy_components"):
                extra_metrics[f"{name}_energy_components"] = packed["energy_components"]
    ckpt_path = out_dir / "adapter.pt"
    torch.save(
        {
            "baseline": baseline,
            "seed": int(seed),
            "adapter_state_dict": model.adapter.state_dict() if hasattr(model, "adapter") else {},
            "best_epoch": best_epoch,
            "best_score": None if not math.isfinite(best_score) else best_score,
            "n_steps": n_steps,
        },
        ckpt_path,
    )
    report = {
        "schema": "dc_uma_cpcm_diagnostic_run_v1",
        "baseline": baseline,
        "seed": int(seed),
        "trained": trained,
        "n_train": n_train,
        "n_valid": len(valid_records),
        "microbatch": micro,
        "accumulate_grad_batches": accum,
        "effective_batch": micro * accum,
        "steps_per_epoch": opt_steps_per_epoch,
        "max_steps": int(max_steps),
        "n_steps": n_steps,
        "best_epoch": best_epoch,
        "best_score": None if not math.isfinite(best_score) else best_score,
        "lr": float(lr),
        "history": history,
        "train": train_metrics["summary"],
        "validation": val_metrics["summary"],
        "slices": val_metrics.get("slices") or {},
        "checkpoint": str(ckpt_path),
        "unseen_claim": "adapter-unseen parent",
        "warm_start": warm_info,
        "lowdin_aux_weight": aux_weight,
    }
    report.update(extra_metrics)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def run_seed_grid(
    build_model,
    records: Sequence[Mapping[str, Any]],
    *,
    baselines: Sequence[str] = DIAGNOSTIC_BASELINES,
    seeds: Sequence[int] = DIAGNOSTIC_SEEDS,
    device: torch.device | None = None,
    out_root: Path | str | None = None,
    accumulate_grad_batches: int = 1,
    max_epochs: int = DIAGNOSTIC_MAX_EPOCHS,
    patience: int = DIAGNOSTIC_PATIENCE,
    batch_size: int = DIAGNOSTIC_MICROBATCH,
    max_steps: int | None = None,
    lr: float = DIAGNOSTIC_LR,
) -> dict[str, Any]:
    split = split_records(records)
    out_root = Path(out_root) if out_root is not None else Path(DEFAULT_EXPERIMENTS) / "runs" / "diagnostic"
    runs = []
    for baseline in baselines:
        for seed in seeds:
            model = build_model(baseline, seed)
            run_dir = out_root / f"{baseline}_seed{seed}"
            report = train_baseline(
                model,
                split["train"],
                split["validation"],
                baseline=baseline,
                seed=int(seed),
                device=device,
                out_dir=run_dir,
                accumulate_grad_batches=accumulate_grad_batches,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                max_steps=max_steps,
                lr=lr,
            )
            runs.append(report)
    gate = decide_go_nogo(runs)
    payload = {
        "schema": "dc_uma_cpcm_diagnostic_report_v1",
        "n_records": len(records),
        "n_train": len(split["train"]),
        "n_valid": len(split["validation"]),
        "runs": [
            {
                "baseline": run["baseline"],
                "seed": run["seed"],
                "checkpoint": run["checkpoint"],
                "n_steps": run["n_steps"],
                "best_score": run["best_score"],
                "slices": run["slices"],
                "validation": run["validation"],
            }
            for run in runs
        ],
        "gate": gate,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / "diagnostic_train.json"
    dest.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    payload["report_path"] = str(dest)
    return payload


class ToyCpcmModel(nn.Module):
    """Tiny residual adapter used by tests (no UMA checkpoint)."""

    def __init__(self, adapter_mode: str = "stage_a", dim: int = 8, hidden: Sequence[int] = (16, 8)) -> None:
        super().__init__()
        from ..models.esen.cpcm import StageASolvationAdapter, StageBSolvationAdapter

        mode = str(adapter_mode).strip().lower().replace("-", "_")
        self.adapter_mode = mode
        self.lift = nn.Linear(3, dim)
        if mode == "zero":
            self.adapter = nn.Module()
        elif mode == "stage_b":
            self.adapter = StageBSolvationAdapter(
                sphere_channels=dim,
                hidden=tuple(hidden),
            )
        else:
            self.adapter = StageASolvationAdapter(
                sphere_channels=dim,
                hidden=tuple(hidden),
                x_only=mode == "x_only",
            )

    def adapter_parameters(self):
        if self.adapter_mode == "zero":
            return []
        params = list(self.lift.parameters()) + list(self.adapter.parameters())
        return [param for param in params if param.requires_grad]

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        batch_seg: Tensor,
        dielectric: Tensor,
        Q: Tensor | None = None,
        S: Tensor | None = None,
    ) -> dict[str, Tensor]:
        n_graphs = int(batch_seg.max().item()) + 1 if batch_seg.numel() else 0
        extras: dict[str, Tensor] = {}
        if self.adapter_mode == "zero":
            dE = Ra.new_zeros(n_graphs)
            dFa = torch.zeros_like(Ra)
        elif self.adapter_mode == "stage_b":
            node = self.lift(Ra)
            if Q is None:
                q_tot = Ra.new_zeros(n_graphs)
            else:
                q_tot = Q
            adapted = self.adapter(node, batch_seg, dielectric, Ra, Za, q_tot)
            dE = adapted["dE"]
            extras = {k: adapted[k] for k in ("E_local", "E_self", "E_pair", "Qa") if k in adapted}
            dFa = energy_gradient_forces(
                dE,
                Ra,
                create_graph=self.training,
                retain_graph=True,
            )
        else:
            node = self.lift(Ra)
            dE = self.adapter(node, batch_seg, dielectric)
            dFa = energy_gradient_forces(
                dE,
                Ra,
                create_graph=self.training,
                retain_graph=True,
            )
        return {"E": dE, "Fa": dFa, "dE": dE, "dFa": dFa, "batch_seg": batch_seg, **extras}
