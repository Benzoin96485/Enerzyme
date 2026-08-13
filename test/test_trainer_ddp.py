"""Unit tests for native DDP trainer helpers (no Lightning import)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from enerzyme.tasks.metrics import Metrics
from enerzyme.tasks.trainer import (
    _convert_lightning_state_dict,
    _load_state_dict,
    _unwrap_ddp,
)


def test_unwrap_ddp_identity_for_plain_module():
    model = nn.Linear(2, 1)
    assert _unwrap_ddp(model) is model


def test_convert_lightning_state_dict_without_lightning():
    lightning_ckpt = {
        "pytorch-lightning_version": "2.5.0",
        "state_dict": {
            "model.layer.weight": torch.tensor([1.0]),
            "other": torch.tensor([2.0]),
        },
        "epoch": 5,
        "lr_schedulers": [{"last_epoch": 4}],
        "optimizer_states": [{"step": 10}],
        "callbacks": {
            "EMACallback": {"decay": 0.999},
            "EarlyStopping{'monitor': '_judge_score'}": {
                "best_score": 0.1,
                "wait_count": 2,
            },
        },
    }
    converted = _convert_lightning_state_dict(lightning_ckpt)
    assert torch.equal(converted["model_state_dict"]["layer.weight"], torch.tensor([1.0]))
    assert torch.equal(converted["model_state_dict"]["other"], torch.tensor([2.0]))
    assert converted["epoch"] == 5
    assert converted["scheduler_state_dict"]["last_epoch"] == 4
    assert converted["optimizer_state_dict"]["step"] == 10
    assert converted["ema_state_dict"]["decay"] == 0.999
    assert converted["best_score"] == 0.1
    assert converted["best_epoch"] == 3


def test_load_state_dict_converts_lightning_checkpoint(tmp_path):
    model = nn.Linear(2, 1, bias=False)
    ckpt = {
        "pytorch-lightning_version": "2.5.0",
        "state_dict": {"model.weight": torch.ones_like(model.weight)},
        "epoch": 2,
    }
    path = tmp_path / "lightning.pth"
    torch.save(ckpt, path)

    target = nn.Linear(2, 1, bias=False)
    info = _load_state_dict(target, device=torch.device("cpu"), pretrain_path=str(path))
    assert torch.allclose(target.weight, torch.ones_like(target.weight))
    assert info["epoch"] == 2


def test_cal_metric_from_partials_matches_cal_metric():
    metrics = Metrics({"Ea": {"rmse": 1.0, "mae": 0.5}})
    y_truth = {"Ea": [1.0, 2.0, 3.0, 4.0]}
    y_pred = {"Ea": [1.5, 2.5, 2.5, 3.5]}
    direct = metrics.cal_metric(y_truth, y_pred)

    shard0 = metrics.accumulate_partials(
        {"Ea": y_truth["Ea"][:2]}, {"Ea": y_pred["Ea"][:2]}
    )
    shard1 = metrics.accumulate_partials(
        {"Ea": y_truth["Ea"][2:]}, {"Ea": y_pred["Ea"][2:]}
    )
    merged = Metrics.merge_partials([shard0, shard1])
    from_partials = metrics.cal_metric_from_partials(merged)

    assert from_partials.keys() == direct.keys()
    for key in direct:
        assert from_partials[key] == pytest.approx(direct[key], rel=1e-6)
