from collections import defaultdict
from functools import partial
from typing import Any, Optional, Iterable, Callable, Dict
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_ema import ExponentialMovingAverage
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from .monitor import Monitor
from ..data.transform import Transform
from ..models.loss import EvaluationLoss, training_loss
from .metrics import Metrics
from .batch import _decorate_batch_output
from .generator_ode import generator_ode_predict_enabled, generator_predict_forward


class _EvaluationDataset(Dataset):
    """Equal-length DDP steps with empty slots instead of replicated molecules."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        n = len(self.dataset)
        replicas = dist.get_world_size() if dist.is_initialized() else 1
        return ((n + replicas - 1) // replicas) * replicas

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            return None
        return idx, self.dataset[idx]


def _collate_evaluation(batch, *, collate_fn):
    samples = [row for row in batch if row is not None]
    if not samples:
        return [], None
    indices, rows = zip(*samples)
    return indices, collate_fn(rows)


def evaluation_loader(dataset, collate_fn, **kwargs):
    return DataLoader(_EvaluationDataset(dataset),
                      collate_fn=partial(_collate_evaluation, collate_fn=collate_fn), **kwargs)


def _evaluation_batch(batch):
    # Direct LightningModel callers can also pass the ordinary training-style batch.
    return batch if isinstance(batch[0], (list, tuple)) else (None, batch)


def _collect_predictions(indices, predicted, truth):
    """Reassemble full test results by dataset position on the writing process."""
    from .trainer import _merge_sharded_prediction_dicts
    packet = (dict(predicted), dict(truth), indices)
    packets = [packet]
    if dist.is_initialized() and dist.get_world_size() > 1:
        packets = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        dist.gather_object(packet, packets, dst=0)
        if packets is None:
            return None, None
    return _merge_sharded_prediction_dicts(*zip(*packets))


class CollectOutputCallback(L.Callback):
    def _collect_output(self, outputs, batch, storage):
        if outputs is None:
            return
        indices, (net_input, net_target) = _evaluation_batch(batch)
        y_pred, y_truth = _decorate_batch_output(outputs["raw_output"], net_input, net_target)
        storage.append((indices, y_pred, y_truth))

    def _finish(self, trainer, pl_module, stage, storage):
        indices, preds, truths = [], defaultdict(list), defaultdict(list)
        for idx, pred, truth in storage:
            if idx is None:
                idx = range(len(indices), len(indices) + len(pred["Za"]))
            indices.extend(idx)
            for key, values in pred.items():
                preds[key].extend(values)
            for key, values in truth.items():
                truths[key].extend(values)
        storage.clear()
        if indices and pl_module.transform is not None:
            pl_module.transform.inverse_transform(preds)
            pl_module.transform.inverse_transform(truths)
        metric_score = pl_module.metrics.distributed_metric(truths, preds)
        loss = pl_module.eval_loss.compute(pl_module.device)
        pl_module.eval_loss = EvaluationLoss(pl_module.loss_terms)
        # Every rank already has the pooled result, including an empty shard.
        pl_module.log(stage + "_loss", loss, batch_size=1)
        pl_module.log_dict(metric_score, batch_size=1)
        if stage == "test":
            preds, truths = _collect_predictions(indices, preds, truths)
            pl_module.test_result = {"y_pred": preds, "y_truth": truths, "metric_score": metric_score}

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect_output(outputs, batch, pl_module.validation_step_outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finish(trainer, pl_module, "val", pl_module.validation_step_outputs)
        pl_module.log("lr", pl_module.optimizer.param_groups[0]["lr"], batch_size=1)
        stop = trainer.early_stopping_callback
        if stop is not None:
            pl_module.log("last_wait_count", stop.wait_count, batch_size=1)
            pl_module.log("last_best_score", stop.best_score, batch_size=1)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._collect_output(outputs, batch, pl_module.test_step_outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        self._finish(trainer, pl_module, "test", pl_module.test_step_outputs)


class MonitorCallback(L.Callback):
    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is not None:
            self.monitor.collect(outputs["raw_output"])

    on_test_batch_end = on_validation_batch_end

    def _finish(self, pl_module, stage):
        stats = {}
        for term, collected in self.monitor.collection.items():
            values = np.asarray(collected, dtype=np.float64)
            if values.size:
                stats[term] = (values.size, values.mean(), values.var(), values.min(), values.max())
        shards = [stats]
        if dist.is_initialized() and dist.get_world_size() > 1:
            shards = [None] * dist.get_world_size()
            dist.all_gather_object(shards, stats)
        for term, names in self.monitor.terms.items():
            parts = [shard[term] for shard in shards if term in shard]
            if not parts:
                continue
            count = sum(n for n, *_ in parts)
            mean = sum(n * m for n, m, *_ in parts) / count
            variance = sum(n * (var + (m - mean) ** 2) for n, m, var, *_ in parts) / count
            summary = {"mean": mean, "std": np.sqrt(variance),
                       "min": min(s[3] for s in parts), "max": max(s[4] for s in parts)}
            for name in names:
                pl_module.log(f"{stage}_{term}_{name}", summary[name], batch_size=1)
        self.monitor._reset()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._finish(pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._finish(pl_module, "test")


class EMACallback(L.Callback):
    def __init__(self, use_ema: bool, ema_decay: Optional[float]=None, ema_use_num_updates: Optional[bool]=None):
        super().__init__()
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_use_num_updates = ema_use_num_updates
        self.ema = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        if not self.use_ema:
            return
        if self.ema is None:
            self.ema = ExponentialMovingAverage(
                pl_module.model.to(pl_module.device).parameters(), self.ema_decay, self.ema_use_num_updates
            )
            initial_state = getattr(pl_module, "initial_ema_state", None)
            if initial_state is not None:
                self.ema.load_state_dict(initial_state)
                pl_module.initial_ema_state = None
        else:
            self.ema.to(pl_module.device)

    def on_fit_start(self, trainer, pl_module):
        # DDP broadcasts model parameters after setup has initialized the shadows.
        if self.use_ema and dist.is_initialized() and dist.get_world_size() > 1:
            for shadow in self.ema.shadow_params:
                dist.broadcast(shadow, src=0)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self.use_ema:
            self.ema.update()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.use_ema:
            self.ema.restore()

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.use_ema:
            self.ema.store()
            self.ema.copy_to()

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self.use_ema:
            self.ema.restore()

    def state_dict(self):
        if self.use_ema:
            return {**self.ema.state_dict(), "collected_params": None}
        else:
            return dict()

    def load_state_dict(self, state_dict: dict):
        if self.use_ema and state_dict:
            self.ema.load_state_dict(state_dict)


class LightningModel(L.LightningModule):
    def __init__(self, 
        model: Module, loss_terms: Iterable[Callable], 
        dump_dir: str,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        monitor: Monitor,
        transform: Transform,
        metrics: Metrics,
        use_ema: bool,
        ema_decay: float,
        ema_use_num_updates: int,
        dump_interval: int,
        generator_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_terms = loss_terms
        self.eval_loss = EvaluationLoss(loss_terms)
        self.dump_dir = dump_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.transform = transform
        self.metrics = metrics
        self.ema_decay = ema_decay
        self.ema_use_num_updates = ema_use_num_updates
        self.use_ema = use_ema
        self.initial_ema_state = None
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_result = None
        self.dump_interval = dump_interval
        self.generator_config = generator_config
        self._fit_callbacks = None

    def on_save_checkpoint(self, checkpoint):
        from .trainer import _capture_rng
        state = _capture_rng(self.device)
        states = [state]
        if dist.is_initialized() and dist.get_world_size() > 1:
            states = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
            dist.gather_object(state, states, dst=0)
        if states is not None:
            checkpoint["enerzyme_rank_rng"] = states

    def on_load_checkpoint(self, checkpoint):
        from .trainer import _restore_rng
        states = checkpoint.get("enerzyme_rank_rng")
        if self.trainer.training and states and len(states) == self.trainer.world_size:
            _restore_rng(states[self.global_rank], self.trainer.strategy.root_device)

    def training_step(self, batch, batch_idx):
        """Return the sum of the configured training loss terms under ``loss``."""
        net_input, net_target = batch
        with torch.set_grad_enabled(True):
            output = self.model(net_input)
            loss = training_loss(self.loss_terms, output, net_target)
        result = {"loss": loss}
        return result
    
    def _prediction_step(self, batch):
        _, data = _evaluation_batch(batch)
        if data is None:
            return None
        net_input, net_target = data
        if generator_ode_predict_enabled(self.generator_config):
            with torch.no_grad():
                output = generator_predict_forward(
                    self.model, net_input, self.generator_config
                )
        else:
            with torch.enable_grad():
                net_input["Ra"].requires_grad_(True)
                output = self.model(net_input)
        loss = self.eval_loss.update(output, net_target)
        result = {"loss": loss, "raw_output": output}
        return result

    def validation_step(self, batch, batch_idx):
        """Return the validation loss and raw model outputs for this batch."""
        return self._prediction_step(batch)

    def test_step(self, batch, batch_idx):
        """Return the test loss and raw model outputs for this batch."""
        return self._prediction_step(batch)

    def configure_optimizers(self):
        """Return the configured optimizer and its learning-rate scheduler, stepped after each optimizer step."""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def configure_callbacks(self):
        # Lightning asks again before trainer.test(). Keep the fitted EMA.
        if self._fit_callbacks is not None:
            return self._fit_callbacks
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.dump_dir,
            monitor="_judge_score",
            mode="min",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
            filename="model_best"
        )
        best_checkpoint_callback.FILE_EXTENSION = ".pth"
        best_checkpoint_callback.CHECKPOINT_NAME_LAST = "model_last"
        collect_output_callback = CollectOutputCallback()
        callbacks = [best_checkpoint_callback, collect_output_callback]
        if self.dump_interval > 0:
            epoch_checkpoint_callback = ModelCheckpoint(
                dirpath=self.dump_dir,
                filename="model_{epoch}",
                every_n_epochs=self.dump_interval,
                save_top_k=-1
            )
            epoch_checkpoint_callback.FILE_EXTENSION = ".pth"
            callbacks.append(epoch_checkpoint_callback)
        if self.monitor is not None:
            monitor_callback = MonitorCallback(self.monitor)
            callbacks.append(monitor_callback)
        if self.use_ema:
            callbacks.append(EMACallback(
                self.use_ema,
                self.ema_decay,
                self.ema_use_num_updates
            ))
        self._fit_callbacks = callbacks
        return callbacks
