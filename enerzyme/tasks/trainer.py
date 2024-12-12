from functools import partial
from typing import Iterable, Optional, Callable, Tuple, Dict, Any, Literal, List, Union
from collections import defaultdict
import time, os, logging, contextlib
from sympy import sqf
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch_ema import ExponentialMovingAverage
try:
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
except:
    pass
from transformers.optimization import get_scheduler
import numpy as np
from .splitter import Splitter
from .monitor import Monitor
from ..data import is_atomic, is_int, is_idx, requires_grad, is_target, Transform, full_neighbor_list
from ..utils import logger
from .metrics import Metrics


DTYPE_MAPPING = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "double": torch.float64,
    "single": torch.float32
}


def _decorate_batch_input(batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]], dtype: torch.dtype, device: torch.device) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    features, targets = zip(*batch)
    batch_features = dict()
    batch_targets = dict()
    
    for k in features[0]:
        if is_atomic(k):
            batch_features[k] = torch.tensor(
                np.concatenate([feature[k][:feature["N"]] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
                requires_grad=requires_grad(k)
            ).to(device)
        elif not is_idx(k):
            batch_features[k] = torch.tensor(
                np.array([feature[k] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
            ).to(device)

    batch_idx_i = []
    batch_idx_j = []
    batch_seg = []
    count = 0

    for i, feature in enumerate(features):
        if "idx_i" in feature:
            batch_idx_i.append(feature["idx_i"][:feature["N_pair"]] + count)
            batch_idx_j.append(feature["idx_j"][:feature["N_pair"]] + count)
        else:
            idx_i, idx_j = full_neighbor_list(feature["N"])
            batch_idx_i.append(idx_i + count)
            batch_idx_j.append(idx_j + count)
        batch_seg.append(np.full(feature["N"], i, dtype=int))
        count += feature["N"]
    batch_features["N"] = [feature["N"] for feature in features]
    batch_features["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long).to(device)
    batch_features["idx_i"] = torch.tensor(np.concatenate(batch_idx_i), dtype=torch.long).to(device)
    batch_features["idx_j"] = torch.tensor(np.concatenate(batch_idx_j), dtype=torch.long).to(device)

    if targets[0] is not None:
        for k in targets[0]:
            if is_atomic(k): 
                batch_targets[k] = torch.tensor(
                    np.concatenate([target[k][:features[i]["N"]] for i, target in enumerate(targets)]), 
                    dtype=torch.long if is_int(k) else dtype
                ).to(device)
            else:
                batch_targets[k] = torch.tensor(
                    np.array([target[k] for target in targets]), 
                    dtype=torch.long if is_int(k) else dtype,
                ).to(device)
    
    return batch_features, batch_targets


def _decorate_batch_output(output: Dict[str, Any], features: Dict[str, Any], targets: Optional[Dict[str, Any]], non_target_features: List[str]=[]) -> Tuple[Dict[str, Union[np.ndarray, List]], Optional[Dict[str, Union[np.ndarray, List]]]]:
    y_pred = dict()
    y_truth = dict()
    for k, v in output.items():
        if is_target(k):
            if is_atomic(k):
                y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
            else:
                y_pred[k] = v.detach().cpu().numpy()
    for k in non_target_features:
        if len(output[k]) == len(features["Za"]):
            y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(output[k], features["N"])))
        elif len(output[k]) == len(features["N"]):
            y_pred[k] = output[k].detach().cpu().numpy()
        else:
            raise ValueError(f"non-target feature {k} has invalid length {len(output[k])}")
    y_pred["Za"] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(features["Za"], features["N"])))
    
    if targets is not None:
        for k, v in targets.items():
            if is_target(k):
                if is_atomic(k):
                    y_truth[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
                else:
                    y_truth[k] = v.detach().cpu().numpy()
    y_truth["Za"] = y_pred["Za"]

    return y_pred, (y_truth if y_truth else None)


def _load_state_dict(model: Module, device: torch.device, pretrain_path: Optional[str], ema: Optional[ExponentialMovingAverage]=None, inference: bool=False, optimizer: Optional[Optimizer]=None, scheduler: Optional[LRScheduler]=None) -> Dict:
    other_info = dict()
    if pretrain_path is None:
        return other_info
    loaded_info = torch.load(pretrain_path, map_location=device)
    if ema is not None and "ema_state_dict" in loaded_info:
        model.load_state_dict(loaded_info["model_state_dict"])
        ema.load_state_dict(loaded_info["ema_state_dict"])
        logger.info(f"loading ema state dict from {pretrain_path}...")
    else:
        if inference and "ema_state_dict" in loaded_info:
            tmp_ema = ExponentialMovingAverage(model.parameters(), decay=1, use_num_updates=True)
            tmp_ema.load_state_dict(loaded_info["ema_state_dict"])
            tmp_ema.copy_to(model.parameters())
            logger.info(f"loading averaged model state dict from {pretrain_path}...")
        else:
            model.load_state_dict(loaded_info["model_state_dict"])
            logger.info(f"loading model state dict from {pretrain_path}...")
    if not inference:
        if optimizer is not None and "optimizer_state_dict" in loaded_info:
            optimizer.load_state_dict(loaded_info["optimizer_state_dict"])
            logger.info(f"loading optimizer state dict from {pretrain_path}...")
        if scheduler is not None and "scheduler_state_dict" in loaded_info:
            scheduler.load_state_dict(loaded_info["scheduler_state_dict"])
            logger.info(f"loading scheduler state dict from {pretrain_path}...")
        if "epoch" in loaded_info:
            other_info["epoch"] = loaded_info["epoch"]
        if "best_epoch" in loaded_info:
            other_info["best_epoch"] = loaded_info["best_epoch"]
        if "best_score" in loaded_info:
            other_info["best_score"] = loaded_info["best_score"]
    # if "torch_rng_state" in loaded_info:
    #     torch.random.set_rng_state(loaded_info["torch_rng_state"])
    #     logger.info(f"loading torch random generator state from {pretrain_path}...")
    # if "torch_cuda_rng_state_all" in loaded_info:
    #     torch.cuda.random.set_rng_state_all(loaded_info["torch_cuda_rng_state_all"])
    #     logger.info(f"loading torch cuda random generator state from {pretrain_path}...")
    # if "np_rng_state" in loaded_info:
    #     np.random.set_state(loaded_info["np_rng_state"])
    #     logger.info(f"loading numpy random generator state from {pretrain_path}...")
    return other_info


class Trainer:
    def __init__(self, out_dir: str=None, metric_config: Metrics=dict(), **params) -> None:
        self.config = params
        self.out_dir = out_dir
        self.metric_config = metric_config
        self.metrics = Metrics(metric_config)
        self.splitter = Splitter(**params["Splitter"])
        if "Monitor" in params:
            self.monitor = Monitor(**params["Monitor"])
        else:
            self.monitor = None
        self.seed = params.get('seed', 114514)
        self.learning_rate = float(params.get('learning_rate', 1e-3))
        self.batch_size = params.get('batch_size', 8)
        self.inference_batch_size = params.get('inference_batch_size', self.batch_size)
        self.max_epochs = params.get('max_epochs', 1000)
        self.warmup_ratio = params.get('warmup_ratio', 0.01)
        self.patience = params.get('patience', 50)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.schedule = params.get('schedule', "linear")
        self.weight_decay = float(params.get('weight_decay', 0))
        self.amsgrad = params.get('amsgrad', True)
        if torch.cuda.is_available():
            logger.info("GPU found!")
            self.device = torch.device("cuda:0" if self.cuda else "cpu")
        else:
            logger.info("GPU not found, turn to CPU!")
            self.device = torch.device("cpu")
        self.data_in_memory = params.get("data_in_memory", True)
        self.use_ema = params.get("use_ema", True)
        self.ema_decay = params.get("ema_decay", 0.999)
        self.ema_use_num_updates = params.get("ema_use_num_updates", True)
        self.dtype = DTYPE_MAPPING[params.get('dtype', "float32")]
        self.committee_size = params.get("committee_size", 1)
        self.active_learning_params = params.get("active_learning_params", None)
        if self.active_learning_params is not None and self.active_learning_params.get("active", False):
            self.active_learning = True
        else:
            self.active_learning = False
        self.resume = params.get("resume", 1)
        self.non_target_features = params.get("non_target_features", [])

    def decorate_batch_input(self, batch):
        return _decorate_batch_input(batch, self.dtype, self.device)
    
    def decorate_batch_output(self, output, features, targets):
        return _decorate_batch_output(output, features, targets, self.non_target_features)
    
    def _set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def load_state_dict(
        self, 
        model: Module, 
        optimizer: Optional[Optimizer]=None,
        scheduler: Optional[LRScheduler]=None,
        pretrain_path: Optional[str]=None, 
        ema: Optional[ExponentialMovingAverage]=None, 
        inference: bool=False
    ) -> None:
        return _load_state_dict(model, self.device, pretrain_path, ema, inference, optimizer, scheduler)

    def save_state_dict(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, dump_dir: str, ema: Optional[ExponentialMovingAverage]=None, suffix="last", model_rank=None, epoch: Optional[int]=None, best_score: Optional[float]=None, best_epoch: Optional[int]=None):
        if model_rank is None:
            model_rank = ''
        os.makedirs(dump_dir, exist_ok=True)
        info = {"model_state_dict": model.state_dict()}
        if ema is not None:
            info["ema_state_dict"] = ema.state_dict()
        info["optimizer_state_dict"] = optimizer.state_dict()
        info["scheduler_state_dict"] = scheduler.state_dict()
        if epoch is not None:
            info["epoch"] = epoch
        if best_score is not None:
            info["best_score"] = best_score
        if best_epoch is not None:
            info["best_epoch"] = best_epoch
        # info["torch_rng_state"] = torch.random.get_rng_state()
        # if self.cuda:
        #     info["torch_cuda_rng_state_all"] = torch.cuda.random.get_rng_state_all()
        # info["np_rng_state"] = np.random.get_state()
        torch.save(info, os.path.join(dump_dir, f'model{model_rank}_{suffix}.pth'))

    def fit_predict(self, 
        model: Module, pretrain_path: Optional[str],
        train_dataset: Dataset, valid_dataset: Optional[Dataset], 
        loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, 
        test_dataset: Optional[Dataset]=None, model_rank: Optional[int]=None, max_epoch_per_iter: int=-1,
        meta_state_dict: Dict=dict(), refresh_patience: bool=False
    ) -> Dict[Literal["y_pred", "y_truth", "metric_score"], Any]:
        self._set_seed(self.seed + (model_rank if model_rank is not None else 0))
        model = model.to(self.device).type(self.dtype)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.decorate_batch_input,
            drop_last=True 
        )
        
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        if self.use_ema:
            ema = ExponentialMovingAverage(model.parameters(), self.ema_decay, self.ema_use_num_updates)
        else:
            ema = None
        scheduler = get_scheduler(self.schedule, optimizer, num_warmup_steps, num_training_steps)

        if self.resume > 1:
            other_info = self.load_state_dict(model, optimizer, scheduler, pretrain_path, ema)
        elif self.resume == 1:
            other_info = self.load_state_dict(model, pretrain_path=pretrain_path, ema=ema)
        else:
            other_info = self.load_state_dict(model, pretrain_path=pretrain_path, inference=True)
        
        if self.resume > 1 and "best_epoch" in other_info and "epoch" in other_info:
            wait = other_info["epoch"] - other_info["best_epoch"]
            best_score = other_info.get("best_score", float("inf"))
            start_epoch = other_info["epoch"] + 1
            if wait >= self.patience and refresh_patience:
                wait = 0
        else:
            wait = 0
            if self.resume > 1:
                start_epoch = other_info.get("epoch", -1) + 1
            else:
                start_epoch = 0
            if valid_dataset is not None:
                if self.resume > 1:
                    best_score = other_info.get("best_score", float("inf"))
                else:
                    best_score = float("inf")
            else:
                best_score = None

        if self.resume > 1:
            max_epochs = self.max_epochs
        else:
            max_epochs = start_epoch + self.max_epochs
        
        if valid_dataset is not None:
            if self.resume > 1:
                best_epoch = other_info.get("best_epoch", start_epoch)
            else:
                best_epoch = None
        else:
            best_epoch = None

        epoch = start_epoch
        epoch_in_iter = meta_state_dict.get("epoch_in_iter", 0)
        if start_epoch > 0:
            if epoch_in_iter > 0:
                logger.info(f"Resuming from epoch {start_epoch + 1}, epoch {epoch_in_iter + 1} in active learning iteration")
            else:
                logger.info(f"Resuming from epoch {start_epoch + 1}...")
        for epoch in range(start_epoch, max_epochs):
            if max_epoch_per_iter > 0 and epoch_in_iter >= max_epoch_per_iter:
                break
            model = model.train()
            start_time = time.time()
            batch_bar = tqdm(
                total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0, 
                desc='Train', ncols=5
            ) 
            trn_loss = []
            
            for i, batch in enumerate(train_dataloader):
                net_input, net_target = batch
                
                loss = 0
                with torch.set_grad_enabled(True):
                    output = model(net_input)
                    for loss_term in loss_terms.values():
                        loss += loss_term(output, net_target)
                trn_loss.append(float(loss.data))

                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
                )

                # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.max_norm > 0:
                    clip_grad_norm_(model.parameters(), self.max_norm)

                optimizer.step()

                if self.use_ema:
                    ema.update()
                
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)
            message = f'Epoch [{epoch + 1}/{max_epochs}]' + (f' ({epoch_in_iter + 1}/{max_epoch_per_iter})' if max_epoch_per_iter > 0 else '') + f', train_loss: {total_trn_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'

            if self.use_ema:
                cm = ema.average_parameters()
            else:
                cm = contextlib.nullcontext()

            if valid_dataset is not None:
                with cm:
                    predict_result = self.predict(
                        model=model, 
                        dataset=valid_dataset, 
                        loss_terms=loss_terms, 
                        dump_dir=dump_dir, 
                        transform=transform, 
                        epoch=epoch, 
                        load_model=False,
                    )
                    val_loss = predict_result["val_loss"]
                    metric_score = predict_result["metric_score"]
                    total_val_loss = np.mean(val_loss)
                    _score = metric_score["_judge_score"]
                    _metric = str(self.metrics)
                    save_handle = partial(self.save_state_dict, model=model, optimizer=optimizer, scheduler=scheduler, dump_dir=dump_dir, ema=ema, suffix="best", model_rank=model_rank)
                    is_early_stop, best_score, wait, saved = self._early_stop_choice(wait, best_score, metric_score, save_handle, self.patience, epoch)
                    if saved:
                        best_epoch = epoch
                    message += f', val_loss: {total_val_loss:.4f}, ' + \
                        ", ".join([f'val_{k}: {v:.4f}' for k, v in metric_score.items() if k != "_judge_score"]) + \
                        f', val_judge_score ({_metric}): {_score:.4f}' + \
                        (f', Patience [{wait}/{self.patience}], min_val_judge_score: {best_score:.4f}' if wait else '')
            else:
                is_early_stop = False

            epoch_in_iter += 1
            meta_state_dict.update({"epoch_in_iter": epoch_in_iter})   
            end_time = time.time()
            message += f', {(end_time - start_time):.1f}s'
            logger.info(message)
            self.save_state_dict(model, optimizer, scheduler, dump_dir, ema, "last", model_rank, epoch=epoch, best_score=best_score, best_epoch=best_epoch)
            if is_early_stop:
                break

        meta_state_dict.update({"model_rank": model_rank + 1 if model_rank is not None else 0, "epoch_in_iter": 0})
        if test_dataset is not None:
            if self.use_ema:
                cm = ema.average_parameters()
            else:
                cm = contextlib.nullcontext()
            with cm:
                predict_result = self.predict(
                    model=model, 
                    dataset=test_dataset, 
                    loss_terms=loss_terms, 
                    dump_dir=dump_dir,  
                    transform=transform, 
                    epoch=epoch, 
                    load_model=True,
                    model_rank=model_rank
                )
                y_pred = predict_result["y_pred"]
                y_truth = predict_result["y_truth"]
                metric_score = predict_result["metric_score"]
        else:
            y_pred = None
            y_truth = None
            metric_score = None
        return {"y_pred": y_pred, "y_truth": y_truth, "metric_score": metric_score}
    
    def _early_stop_choice(self, wait, min_loss, metric_score, save_handle, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, save_handle, patience, epoch)
    
    def predict(self, model: Module, dataset: Dataset, loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, epoch: int=1, load_model: bool=False, model_rank: Optional[str]=None) -> Dict[Literal["y_pred", "y_truth", "val_loss", "metric_score"], Any]:
        self._set_seed(self.seed)
        model = model.to(self.device).type(self.dtype)
        if load_model == True:
            from ..models import get_pretrain_path
            pretrain_path = get_pretrain_path(dump_dir, "best", model_rank)
            self.load_state_dict(model, pretrain_path=pretrain_path, inference=True)
            
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            collate_fn=self.decorate_batch_input
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = defaultdict(list)
        y_truths = defaultdict(list)
        for i, batch in enumerate(dataloader):
            net_input, net_target = batch
            output = model(net_input)
            # Get model outputs
            if self.monitor is not None:
                self.monitor.collect(output)
            loss = 0
            with torch.no_grad():
                if not load_model:
                    for loss_term in loss_terms.values():
                        loss += loss_term(output, net_target)
                    val_loss.append(float(loss.data))
            y_pred, y_truth = self.decorate_batch_output(output, net_input, net_target)
            for k, v in y_pred.items():
                y_preds[k].extend(v)
            for k, v in y_truth.items():
                y_truths[k].extend(v)
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1)))
                )
            batch_bar.update()

        if self.monitor is not None:
            self.monitor.summary()

        if transform is not None:
            transform.inverse_transform(y_preds)
            transform.inverse_transform(y_truths)
        metric_score = self.metrics.cal_metric(y_truths, y_preds)
        if load_model and "_judge_score" in metric_score:
            metric_score.pop("_judge_score")
        batch_bar.close()
        return {"y_pred": y_preds, "y_truth": y_truths, "val_loss": val_loss, "metric_score": metric_score}
