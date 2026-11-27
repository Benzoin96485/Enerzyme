"""Unit tests for native DDP trainer helpers (no Lightning import)."""
from __future__ import annotations

import os
import pytest
import torch
from torch import nn

from enerzyme.tasks.trainer import (
    _convert_lightning_state_dict,
    _load_state_dict,
    _merge_sharded_prediction_dicts,
    _require_nonempty_train_loader,
    _strided_shard_indices,
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
        "global_step": 11,
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
    assert converted["global_step"] == 11
    assert converted["wait"] == 2
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


def test_merge_sharded_prediction_dicts_restores_dataset_order():
    n_samples = 10
    world_size = 4
    gathered_preds = []
    gathered_truths = []
    gathered_indices = []
    for rank in range(world_size):
        indices = _strided_shard_indices(rank, world_size, n_samples)
        gathered_indices.append(indices)
        gathered_preds.append(
            {
                "Ea": [float(i) for i in indices],
                "data_key": [f"s{i}" for i in indices],
            }
        )
        gathered_truths.append({"Ea": [10.0 + i for i in indices]})

    preds, truths = _merge_sharded_prediction_dicts(
        gathered_preds, gathered_truths, gathered_indices
    )
    assert preds["Ea"] == [float(i) for i in range(n_samples)]
    assert preds["data_key"] == [f"s{i}" for i in range(n_samples)]
    assert truths["Ea"] == [10.0 + i for i in range(n_samples)]
    rank_concat = [v for shard in gathered_preds for v in shard["Ea"]]
    assert preds["Ea"] != rank_concat


def test_merge_sharded_prediction_dicts_skips_empty_ranks():
    n_samples = 2
    world_size = 4
    gathered_preds = []
    gathered_truths = []
    gathered_indices = []
    for rank in range(world_size):
        indices = _strided_shard_indices(rank, world_size, n_samples)
        gathered_indices.append(indices)
        if indices:
            gathered_preds.append({"Ea": [float(i) for i in indices]})
            gathered_truths.append({"Ea": [100.0 + i for i in indices]})
        else:
            gathered_preds.append({})
            gathered_truths.append({})

    preds, truths = _merge_sharded_prediction_dicts(
        gathered_preds, gathered_truths, gathered_indices
    )
    assert preds["Ea"] == [0.0, 1.0]
    assert truths["Ea"] == [100.0, 101.0]


def test_require_nonempty_train_loader_passes_when_batches_exist():
    _require_nonempty_train_loader(1, n_samples=128, batch_size=32, world_size=4)


def test_require_nonempty_train_loader_fails_fast_for_ddp():
    with pytest.raises(RuntimeError, match="zero optimizer steps"):
        _require_nonempty_train_loader(0, n_samples=100, batch_size=32, world_size=4)


def test_require_nonempty_train_loader_fails_fast_single_process():
    with pytest.raises(RuntimeError, match="zero optimizer steps"):
        _require_nonempty_train_loader(0, n_samples=8, batch_size=32, world_size=1)


def test_legacy_native_best_recovers_raw_weights_and_preserves_ema(tmp_path):
    from torch_ema import ExponentialMovingAverage

    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters())
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    ema.update()
    raw = model.weight.detach().clone()
    path = tmp_path / "legacy_best.pth"
    with ema.average_parameters():
        assert not torch.equal(model.weight, raw)
        torch.save({"model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, path)

    target = nn.Linear(2, 1, bias=False)
    opt = torch.optim.Adam(target.parameters())
    average = ExponentialMovingAverage(target.parameters(), decay=0.1)
    _load_state_dict(target, pretrain_path=path, optimizer=opt, ema=average)
    torch.testing.assert_close(target.weight, raw, rtol=0, atol=0)
    torch.testing.assert_close(average.shadow_params, ema.shadow_params, rtol=0, atol=0)


@pytest.mark.parametrize("resume, expected_weight, expected_shadow", [(0, 2.0, None), (1, 5.0, 2.0)])
def test_lightning_warm_start_and_explicit_cpu(tmp_path, monkeypatch, resume, expected_weight, expected_shadow):
    pytest.importorskip("lightning")
    from torch_ema import ExponentialMovingAverage
    from enerzyme.tasks.trainer import Trainer

    for key in list(os.environ):
        if key.startswith(("SLURM_", "TORCHELASTIC_")) or key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            monkeypatch.delenv(key)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)
    with torch.no_grad():
        model.weight.fill_(5)
    checkpoint = tmp_path / "pretrained.pth"
    torch.save({"model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict()}, checkpoint)
    with torch.no_grad():
        model.weight.fill_(100)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = Trainer(str(tmp_path), {}, lightning=True, cuda=False, resume=resume,
                      Splitter={"parts": ["training"], "ratios": [1]}, batch_size=1, num_workers=0)
    assert trainer.lightning_trainer.strategy.root_device.type == "cpu"

    def inspect_fit(module, *loaders, ckpt_path):
        assert ckpt_path is None
        assert module.model.weight.item() == expected_weight
        assert module.optimizer.state == {}
        initial = getattr(module, "initial_ema_state", None)
        if expected_shadow is None:
            assert initial is None
        else:
            assert initial["shadow_params"][0].item() == expected_shadow

    monkeypatch.setattr(trainer.lightning_trainer, "fit", inspect_fit)
    trainer.fit_predict(model, checkpoint, [None], None, {}, str(tmp_path), None)


@pytest.mark.parametrize("warmup, expected", [(2, [0, .5, 1, .5, 0, 0]), (0, [1, .75, .5, .25, 0, 0])])
def test_default_scheduler_matches_linear_warmup_and_decay(warmup, expected):
    from enerzyme.tasks.trainer import _training_scheduler

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = _training_scheduler("linear", optimizer, warmup, 4)
    actual = []
    for _ in expected:
        actual.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert actual == expected
    # The previous Transformers default stored a functools.partial in LambdaLR as well.
    assert scheduler.state_dict()["lr_lambdas"] == [{}]
