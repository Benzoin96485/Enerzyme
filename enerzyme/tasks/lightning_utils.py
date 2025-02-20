from collections import defaultdict
from typing import Any, Optional, Iterable, Callable, Dict, Literal, Union, Tuple, List
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_ema import ExponentialMovingAverage
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from .monitor import Monitor
from ..data import Transform
from .metrics import Metrics
from .batch import _decorate_batch_output


class CollectOutputCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def _collect_output(self, 
        outputs: Dict[Literal["raw_output", "loss"], Union[Dict[str, Tensor], float]], 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]]
    ):
        net_input, net_target = batch
        y_pred, y_truth = _decorate_batch_output(outputs["raw_output"], net_input, net_target)
        result = {"y_pred": y_pred, "y_truth": y_truth, "loss": outputs["loss"]}
        return result

    def _reduce_outputs(self, 
        step_outputs: List[Dict[str, Any]], 
        transform: Transform, 
        metrics: Metrics,
    ):
        y_preds = defaultdict(list)
        y_truths = defaultdict(list)
        total_loss = np.mean([output["loss"].item() for output in step_outputs])
        for output in step_outputs:
            for k, v in output["y_pred"].items():
                y_preds[k].extend(v)
            for k, v in output["y_truth"].items():
                y_truths[k].extend(v)
        if transform is not None:
            transform.inverse_transform(y_preds)
            transform.inverse_transform(y_truths)
        metric_score = metrics.cal_metric(y_truths, y_preds)
        step_outputs.clear()
        return y_preds, y_truths, total_loss, metric_score
    
    def on_validation_batch_end(self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Dict[Literal["raw_output", "loss"], Union[Dict[str, Tensor], float]], 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
        batch_idx: int, 
        dataloader_idx: int=0
    ) -> None:
        pl_module.validation_step_outputs.append(self._collect_output(outputs, batch))

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _, _, total_loss, metric_score = self._reduce_outputs(
            pl_module.validation_step_outputs,
            pl_module.transform,
            pl_module.metrics
        )
        pl_module.log("val_loss", total_loss, sync_dist=True)
        pl_module.log_dict(metric_score, sync_dist=True)

    def on_test_batch_end(self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Dict[Literal["raw_output", "loss"], Union[Dict[str, Tensor], float]], 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
        batch_idx: int, dataloader_idx: int=0
    ) -> None:
        pl_module.test_step_outputs.append(self._collect_output(outputs, batch))

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        y_preds, y_truths, total_loss, metric_score = self._reduce_outputs(
            pl_module.test_step_outputs,
            pl_module.transform,
            pl_module.metrics,
        )
        pl_module.log("test_loss", total_loss, sync_dist=True)
        pl_module.log_dict(metric_score, sync_dist=True)
        pl_module.test_result = {"y_pred": y_preds, "y_truth": y_truths, "metric_score": metric_score}


class MonitorCallback(L.Callback):
    def __init__(self, monitor: Monitor):
        super().__init__()
        self.monitor = monitor

    def on_validation_batch_end(self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        outputs: Dict[Literal["raw_output", "loss"], Union[Dict[str, Tensor], float]], 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
        batch_idx: int, 
        dataloader_idx: int=0
    ) -> None:
        self.monitor.collect(outputs["raw_output"])

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.monitor.summary()

    def on_test_batch_end(self, 
        pl_module: L.LightningModule, 
        outputs: Dict[Literal["raw_output", "loss"], Union[Dict[str, Tensor], float]], 
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], 
        batch_idx: int, 
        dataloader_idx: int=0
    ) -> None:
        self.monitor.collect(outputs["raw_output"])

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.monitor.summary()


class EMACallback(L.Callback):
    def __init__(self, use_ema: bool, ema_decay: Optional[float]=None, ema_use_num_updates: Optional[bool]=None):
        super().__init__()
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_use_num_updates = ema_use_num_updates

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        if self.use_ema:
            self.ema = ExponentialMovingAverage(pl_module.model.to(pl_module.device).parameters(), self.ema_decay, self.ema_use_num_updates)
        else:
            self.ema = ExponentialMovingAverage(pl_module.model.to(pl_module.device).parameters(), 1, True)

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if self.use_ema:
            self.ema.update()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        #self.ema.average_parameters()
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
            return self.ema.state_dict()
        else:
            return dict()

    def load_state_dict(self, state_dict: dict):
        if self.use_ema and state_dict:
            self.ema.load_state_dict(state_dict)
        if not self.use_ema:
            self.ema.copy_to(self.model.parameters())


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
        ema_use_num_updates: int
    ):
        super().__init__()
        self.model = model
        self.loss_terms = loss_terms
        self.dump_dir = dump_dir
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.monitor = monitor
        self.transform = transform
        self.metrics = metrics
        self.ema_decay = ema_decay
        self.ema_use_num_updates = ema_use_num_updates
        self.use_ema = use_ema
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_result = None

    def training_step(self, batch, batch_idx):
        net_input, net_target = batch      
        loss = 0
        with torch.set_grad_enabled(True):
            output = self.model(net_input)
            for loss_term in self.loss_terms.values():
                loss += loss_term(output, net_target)
        result = {"loss": loss}
        return result
    
    def _prediction_step(self, batch):
        net_input, net_target = batch
        with torch.enable_grad():
            net_input["Ra"].requires_grad_(True)
            output = self.model(net_input)
        loss = 0
        with torch.no_grad():
            for loss_term in self.loss_terms.values():
                loss += loss_term(output, net_target)
        result = {"loss": loss, "raw_output": output}
        return result

    def validation_step(self, batch, batch_idx):
        return self._prediction_step(batch)

    def test_step(self, batch, batch_idx):
        return self._prediction_step(batch)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler
        }

    def configure_callbacks(self):
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.dump_dir,
            monitor="_judge_score",
            mode="min",
            save_top_k=1,
            filename="model_best"
        )
        best_checkpoint_callback.FILE_EXTENSION = ".pt"
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=self.dump_dir,
            filename="model_last"
        )
        last_checkpoint_callback.FILE_EXTENSION = ".pt"
        collect_output_callback = CollectOutputCallback()
        monitor_callback = MonitorCallback(self.monitor)
        callbacks = [best_checkpoint_callback, last_checkpoint_callback, collect_output_callback, monitor_callback]
        if self.use_ema:
            callbacks.append(EMACallback(
                self.use_ema,
                self.ema_decay,
                self.ema_use_num_updates
            ))
        return callbacks