"""Use unequal numbers of atoms per rank to compare against one full-batch step."""

import os
import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from enerzyme.models.loss import (
    CFMLoss, CRPSLoss, EvaluationLoss, L2Penalty, MAELoss, MSELoss, NLLLoss,
    NLLLossVarOnly, RMSELoss, training_loss,
)
from enerzyme.models.newtonnet.loss import LatentForceCosineLoss
from enerzyme.models.physnet.loss import NHLoss
from enerzyme.tasks.metrics import Metrics


class _Potential(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def forward(self, coords, sizes):
        coords = coords.detach().requires_grad_()
        energy = torch.stack([v.sum() for v in (self.scale * coords.square().sum(-1)).split(sizes)])
        forces, = torch.autograd.grad(-energy.sum(), coords, create_graph=True)
        q = forces[:, 0]
        return {"E": energy, "Fa": forces, "E_var": 0.5 + energy * 0.02,
                "Fa_std": 1 + forces.square() * 0.01, "E_members": torch.stack((energy, energy * 1.5), -1),
                "Qa": q, "Sa": 2*q, "Q_vel_a": q, "S_vel_a": q + 0.3,
                "l2_penalty": self.scale.square(), "nh_loss": q.square().mean(),
                "latent_force": forces + coords.roll(1, dims=-1)}


def _data(sizes, start):
    coords = torch.arange(start, start + 3 * sum(sizes), dtype=torch.float64).view(-1, 3) / 20
    q = coords[:, 0]
    energy = torch.stack([v.sum() for v in coords[:, 1].split(sizes)])
    return coords, {"E": energy, "Fa": coords, "E_members": energy, "Qa": q, "Sa": -q,
                    "Q_init_a": q * 0.7, "S_init_a": 0.2 - q}


class _LightningPotential(_Potential):
    def forward(self, features):
        out = super().forward(features["Ra"], features["N"])
        return {"E": out["E"], "Fa": out["Fa"]}


def _collate_molecules(rows):
    sizes = [i + 1 for i in rows]
    coords = torch.cat([_data((i + 1,), 3 * i)[0] for i in rows])
    features = {"N": sizes, "Ra": coords, "Za": torch.ones(len(coords), dtype=torch.long)}
    labels = {"E": torch.zeros(len(sizes), dtype=coords.dtype), "Fa": torch.zeros_like(coords),
              "data_key": [str(i) for i in rows]}
    return features, labels


def _check_lightning(rank, directory):
    from lightning import Trainer
    from lightning.fabric.plugins.environments import LightningEnvironment
    from enerzyme.tasks.lightning_utils import LightningModel, evaluation_loader

    os.environ["LOCAL_RANK"] = str(rank)
    model = _LightningPotential()
    # The model and its EMA must use rank zero's initialization.
    with torch.no_grad():
        model.scale.add_(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    metrics = Metrics({"Fa": {"rmse": 1}, "E": {"mae": 1}})
    module = LightningModel(model, {"mse": MSELoss(E=1, Fa=1)}, str(directory),
                            optimizer, torch.optim.lr_scheduler.StepLR(optimizer, 1),
                            None, None, metrics, True, 0.8, True, 1)
    trainer = Trainer(accelerator="cpu", devices=2, strategy="ddp", max_epochs=1,
                      plugins=[LightningEnvironment()], logger=False, inference_mode=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    train = DataLoader(range(4), batch_size=1, collate_fn=_collate_molecules)
    valid = evaluation_loader(range(1), _collate_molecules, batch_size=1)
    trainer.fit(module, train, valid)
    ckpt = torch.load(directory / "model_best.pth", map_location="cpu", weights_only=False)
    assert len(ckpt["enerzyme_rank_rng"]) == 2
    shadow, = ckpt["callbacks"]["EMACallback"]["shadow_params"]
    checkpoint = trainer.checkpoint_callback
    sample, _ = _collate_molecules([0])
    valid_force = (-2 * shadow * sample["Ra"]).abs().mean()
    valid_energy = (shadow * sample["Ra"].square().sum()).abs()
    assert checkpoint.best_model_score.item() == pytest.approx((valid_energy + valid_force).item())

    test = evaluation_loader(range(3), _collate_molecules, batch_size=1)
    trainer.test(module, test, verbose=False)
    result = module.test_result
    if rank == 0:
        assert result["y_pred"]["data_key"] == ["0", "1", "2"]
        for idx, actual in enumerate(result["y_pred"]["Fa"]):
            sample, _ = _collate_molecules([idx])
            np.testing.assert_allclose(actual, (-2 * shadow * sample["Ra"]).numpy())
        expected = metrics.cal_metric(result["y_truth"], result["y_pred"])
        assert result["metric_score"] == pytest.approx(expected)
    else:
        assert result["y_pred"] is None


def _worker(rank, directory):
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(directory / "rendezvous").resolve().as_uri())
    try:
        sizes = (1, 2, 4, 7)
        local_sizes = sizes[:2] if rank == 0 else sizes[2:]
        coords, labels = _data(local_sizes, 0 if rank == 0 else 9)
        full_coords, full_labels = _data(sizes, 0)
        model = DistributedDataParallel(_Potential())
        reference = _Potential()
        losses = [
            MSELoss(E=0.2, Fa=2.1, E_members=0.4), RMSELoss(E=0.1, Fa=4), MAELoss(Fa=1),
            NLLLoss(E=1), NLLLossVarOnly(E=1), CRPSLoss(Fa=1), L2Penalty(weight=0.4),
            CFMLoss(), CFMLoss(vel_q_key="absent", weight_s=0.4),
            NHLoss(weight=0.7), LatentForceCosineLoss(weight=0.3),
        ]
        update = torch.optim.SGD(model.parameters(), lr=1e-3)
        ref_update = torch.optim.SGD(reference.parameters(), lr=1e-3)
        for criterion in losses:
            update.zero_grad()
            ref_update.zero_grad()
            out = model(coords, local_sizes)
            full_out = reference(full_coords, sizes)
            actual = training_loss({"term": criterion}, out, labels)
            expected = criterion(full_out, full_labels)
            torch.testing.assert_close(actual, expected)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(model.module.scale.grad, reference.scale.grad)
            update.step()
            ref_update.step()
            torch.testing.assert_close(model.module.scale, reference.scale)

        # Only one process sees a physical structure on this evaluation.
        terms = {"energy_force": RMSELoss(E=0.1, Fa=0.9), "reg": L2Penalty(0.4)}
        evaluation = EvaluationLoss(terms)
        full_output = reference(full_coords, sizes)
        if rank == 0:
            evaluation.update(full_output, full_labels)
        expected_loss = sum(t(full_output, full_labels) for t in terms.values())
        assert evaluation.compute("cpu") == pytest.approx(expected_loss.item())

        evaluation = EvaluationLoss(terms)
        start = 0 if rank == 0 else 9
        for size in local_sizes:
            subset, ref = _data((size,), start)
            evaluation.update(reference(subset, (size,)), ref)
            start += 3 * size
        assert evaluation.compute() == pytest.approx(expected_loss.item())

        from enerzyme.tasks.lightning_utils import _collect_predictions, evaluation_loader
        for n in (1, 5):
            loader = evaluation_loader(list(range(n)), collate_fn=list, batch_size=2)
            sampler = DistributedSampler(loader.dataset, shuffle=False)
            items = [loader.collate_fn([loader.dataset[i]]) for i in sampler]
            ids = [i for inds, rows in items for i in inds]
            assert ids == list(range(rank, n, 2))
            merged, _ = _collect_predictions(ids, {"E": ids}, {"E": ids})
            if rank == 0:
                assert merged["E"] == list(range(n))
            else:
                assert merged is None
            metric = Metrics({"Fa": {"rmse": 0.5, "mae": 1}, "E": {"rmse": 1}})
            prediction = {"Fa": [np.array([[i + 1.0, 0, 0]]) for i in ids], "E": ids}
            truth = {"Fa": [np.zeros((1, 3)) for _ in ids], "E": [0] * len(ids)}
            scores = metric.distributed_metric(truth, prediction)
            f_rmse = np.sqrt(np.mean(np.arange(1, n + 1, dtype=float) ** 2)) / 3
            assert scores["Fa_rmse"] == pytest.approx(f_rmse)
            assert scores["_judge_score"] == pytest.approx(0.5 * f_rmse + np.mean(np.arange(1, n+1))/3
                                                            + np.sqrt(np.mean(np.arange(n)**2)))
        _check_lightning(rank, directory)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_two_ranks_match_full_batch(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(tmp_path,), nprocs=2)


def test_componentwise_tensor_metrics_and_ensemble_reduction():
    metric = Metrics({"polarizability": {"rmse": 1, "mae": 1}})
    values = np.zeros((5, 3, 3))
    values[:, 0, 0] = [1, 2, 7, 4, 2]
    labels = {"polarizability": [np.zeros((1, 3, 3)) for _ in values]}
    members = np.stack((values - 0.25, values + 0.25), -1)
    predictions = {"polarizability": [v[None] for v in members]}
    expected = metric.cal_metric(labels, predictions)
    actual = metric.cal_metric_from_partials(metric.merge_partials([
        metric.accumulate_partials({k: v[sl] for k, v in labels.items()},
                                   {k: v[sl] for k, v in predictions.items()})
        for sl in (slice(0, 1), slice(1, None), slice(0, 0))
    ]))
    assert actual == pytest.approx(expected)


def test_scalar_metric_with_singleton_label_column():
    metric = Metrics({"E": {"rmse": 1}})
    truth = {"E": [np.array([1]), np.array([4])]}
    predicted = {"E": [0, 2]}
    score = metric.cal_metric_from_partials(metric.accumulate_partials(truth, predicted))
    assert score["E_rmse"] == pytest.approx(np.sqrt(2.5))
