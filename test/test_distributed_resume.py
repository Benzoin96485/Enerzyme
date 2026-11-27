"""Verify epoch-boundary continuation through Trainer's public fit and prediction APIs."""
from datetime import timedelta
import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from enerzyme.models.loss import MSELoss
from enerzyme.tasks.trainer import Trainer


class _Potential(nn.Module):
    def __init__(self):
        super().__init__()
        self.stiffness = nn.Parameter(torch.randn(()))

    def forward(self, features):
        coordinates = features["Ra"]
        scale = self.stiffness
        if self.training:
            scale = scale * (random.random() + np.random.random() + torch.rand(()))
        atomic_energy = scale * coordinates.square().sum(dim=1)
        energy = atomic_energy.new_zeros(len(features["N"])).index_add(
            0, features["batch_seg"], atomic_energy
        )
        forces, = torch.autograd.grad(energy.sum(), coordinates, create_graph=self.training)
        return {"E": energy, "Fa": -forces}


def _dataset():
    samples = []
    for index in range(8):
        atoms = 1 + index % 3
        coordinates = 0.1 * (np.arange(3 * atoms).reshape(atoms, 3) + index + 1)
        features = {"N": atoms, "Za": np.ones(atoms), "Ra": coordinates}
        targets = {"E": np.square(coordinates).sum(), "Fa": -2 * coordinates}
        samples.append((features, targets, str(index)))
    return samples


def _train(directory, checkpoint=None, limit=-1, rank=0):
    torch.manual_seed(44 + rank)
    model = _Potential()
    trainer = Trainer(
        str(directory), {"E": {"mae": 1}, "Fa": {"rmse": 1}},
        Splitter={"parts": ["train"], "ratios": [1]}, batch_size=2, max_epochs=3,
        num_workers=0, seed=123, tensorboard=False, cuda=False, learning_rate=0.01,
        resume=2, warmup_ratio=0.1, dump_interval=1,
    )
    dataset = _dataset()
    losses = {"mse": MSELoss(E=1, Fa=1)}
    # Only rank zero has a validation sample. Test has uneven, nonempty shards.
    result = trainer.fit_predict(
        model, checkpoint, dataset, dataset[:1], losses, str(directory), None,
        test_dataset=dataset[:3], max_epoch_per_iter=limit,
    )
    if rank == 0:
        assert result["y_pred"]["data_key"] == ["0", "1", "2"]
    else:
        assert result["y_pred"] == {}
    if dist.is_initialized():
        scores = [None] * dist.get_world_size()
        dist.all_gather_object(scores, result["metric_score"])
        assert scores[0] == scores[1]
    standalone = trainer.predict(model, dataset[:1], losses, str(directory), None, load_model=True)
    if rank == 0:
        assert standalone["y_truth"]["data_key"] == ["0"]
    else:
        assert standalone["y_truth"] == {}


def _assert_checkpoint_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_checkpoint_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_checkpoint_equal(first, second)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        assert left == right


def _run_resume(rank, world, workdir):
    if world > 1:
        os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                          LOCAL_WORLD_SIZE=str(world), GROUP_RANK="0")
        dist.init_process_group("gloo", init_method=(workdir / "rendezvous").as_uri(),
                                rank=rank, world_size=world, timeout=timedelta(seconds=60))
    try:
        full = workdir / "uninterrupted"
        resumed = workdir / "resumed"
        _train(full, rank=rank)
        _train(resumed, limit=1, rank=rank)
        _train(resumed, resumed / "model_last.pth", rank=rank)
        if rank != 0:
            return
        for suffix in ("epoch=1", "best", "last"):
            left = torch.load(full / f"model_{suffix}.pth", map_location="cpu", weights_only=True)
            right = torch.load(resumed / f"model_{suffix}.pth", map_location="cpu", weights_only=True)
            assert len(left["rank_states"]) == world
            assert left["global_step"] == (left["epoch"] + 1) * 4 // world
            _assert_checkpoint_equal(left, right)
        if world > 1:
            assert not torch.equal(left["rank_states"][0]["torch"], left["rank_states"][1]["torch"])
    finally:
        if world > 1:
            dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _isolated_launcher(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("SLURM_", "TORCHELASTIC_")) or key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            monkeypatch.delenv(key)


def test_single_process_epoch_resume(tmp_path):
    _run_resume(0, 1, tmp_path)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires CPU Gloo")
def test_two_process_epoch_resume(tmp_path):
    mp.start_processes(_run_resume, args=(2, tmp_path), nprocs=2, start_method="fork")


def test_lightning_checkpoint_native_next_step(tmp_path):
    """Compare a live Lightning trajectory with a native load on a frozen E/F batch."""
    lightning = pytest.importorskip("lightning.pytorch")
    from torch.utils.data import DataLoader
    from torch_ema import ExponentialMovingAverage
    from enerzyme.tasks.batch import CollateBatch
    from enerzyme.tasks.lightning_utils import EMACallback, LightningModel, evaluation_loader
    from enerzyme.tasks.metrics import Metrics
    from enerzyme.tasks.trainer import _load_state_dict

    torch.manual_seed(42)
    losses = {"mse": MSELoss(E=1, Fa=1)}
    dataset = _dataset()
    collate = CollateBatch(pyg=False, dtype=torch.float32, device=None, otf_graph=True,
                           generator_config=None, generator_training=False)
    train = DataLoader(dataset[:4], batch_size=2, collate_fn=collate)
    valid = evaluation_loader(dataset[:1], collate, batch_size=2)

    def components():
        model = _Potential()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return model, optimizer, scheduler

    original, optimizer, scheduler = components()
    module = LightningModel(original, losses, str(tmp_path), optimizer, scheduler, None, None,
                            Metrics({"E": {"mae": 1}}), True, 0.9, True, -1)
    early = lightning.callbacks.EarlyStopping(monitor="_judge_score", mode="min")
    trainer = lightning.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, callbacks=[early],
        inference_mode=False, logger=False, enable_model_summary=False,
        enable_progress_bar=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, train, valid)
    original_ema = next(cb.ema for cb in trainer.callbacks if isinstance(cb, EMACallback))
    restored, optimizer2, scheduler2 = components()
    restored_ema = ExponentialMovingAverage(restored.parameters(), decay=0.9)
    info = _load_state_dict(restored, torch.device("cpu"), tmp_path / "model_last.pth",
                            restored_ema, optimizer=optimizer2, scheduler=scheduler2)
    assert (info["global_step"], info["epoch"], info["wait"]) == (len(train), 0, 0)
    assert info["best_score"] == pytest.approx(early.best_score.item())

    def next_step(model, opt, sch, ema):
        random.seed(15)
        np.random.seed(22)
        torch.manual_seed(36)
        inputs, targets = collate(dataset[4:6])
        predictions = model.train()(inputs)
        loss = losses["mse"](predictions, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()
        sch.step()
        return (predictions, loss.detach(), model.state_dict(), opt.state_dict(),
                sch.state_dict(), ema.shadow_params)

    expected = next_step(original, optimizer, scheduler, original_ema)
    actual = next_step(restored, optimizer2, scheduler2, restored_ema)
    _assert_checkpoint_equal(expected, actual)
