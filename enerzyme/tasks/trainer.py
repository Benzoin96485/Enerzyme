from ast import dump
from functools import partial
from typing import Iterable, Optional, Callable, Tuple, Dict, Any
from collections import defaultdict
import time, os, logging, contextlib
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
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


def _decorate_batch_output(output, features, targets):
    y_pred = dict()
    y_truth = dict()
    for k, v in output.items():
        if is_target(k):
            if is_atomic(k):
                y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
            else:
                y_pred[k] = v.detach().cpu().numpy()
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


def _load_state_dict(model: Module, device: torch.device, pretrain_path: Optional[str], ema: Optional[ExponentialMovingAverage]=None, inference: bool=False) -> None:
    if pretrain_path is None:
        return
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

    def decorate_batch_input(self, batch):
        return _decorate_batch_input(batch, self.dtype, self.device)
    
    def decorate_batch_output(self, output, features, targets):
        return _decorate_batch_output(output, features, targets)
    
    def _set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def load_state_dict(self, model: Module, pretrain_path: Optional[str], ema: Optional[ExponentialMovingAverage]=None, inference: bool=False) -> None:
        return _load_state_dict(model, self.device, pretrain_path, ema, inference)

    def save_state_dict(self, model: Module, dump_dir, ema: Optional[ExponentialMovingAverage]=None, suffix="last", model_rank=None):
        if model_rank is None:
            model_rank = ''
        os.makedirs(dump_dir, exist_ok=True)
        if ema is None:
            info = {'model_state_dict': model.state_dict()}
        else:
            info = {'ema_state_dict': ema.state_dict(), 'model_state_dict': model.state_dict()}
        torch.save(info, os.path.join(dump_dir, f'model{model_rank}_{suffix}.pth'))

    def fit_predict(self, 
        model: Module, pretrain_path: Optional[str],
        train_dataset: Dataset, valid_dataset: Dataset, 
        loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, 
        test_dataset: Optional[Dataset]=None, model_rank=None) -> Tuple[Optional[defaultdict[Any]], Dict]:
        self._set_seed(self.seed + (model_rank if model_rank is not None else 0))
        model = model.to(self.device).type(self.dtype)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.decorate_batch_input,
            drop_last=True 
        )
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-6, weight_decay=self.weight_decay, amsgrad=self.amsgrad)
        if self.use_ema:
            ema = ExponentialMovingAverage(model.parameters(), self.ema_decay, self.ema_use_num_updates)
        else:
            ema = None
        self.load_state_dict(model, pretrain_path, ema)
        scheduler = get_scheduler(self.schedule, optimizer, num_warmup_steps, num_training_steps)

        for epoch in range(self.max_epochs):
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
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
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
            message = f'Epoch [{epoch+1}/{self.max_epochs}] train_loss: {total_trn_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'

            if self.use_ema:
                cm = ema.average_parameters()
            else:
                cm = contextlib.nullcontext()
            y_preds = None
            if valid_dataset is not None:
                with cm:
                    _, val_loss, metric_score = self.predict(
                        model=model, 
                        dataset=valid_dataset, 
                        loss_terms=loss_terms, 
                        dump_dir=dump_dir, 
                        transform=transform, 
                        epoch=epoch, 
                        load_model=False,
                    )
                    total_val_loss = np.mean(val_loss)
                    _score = metric_score["_judge_score"]
                    _metric = str(self.metrics)
                    save_handle = partial(self.save_state_dict, model=model, dump_dir=dump_dir, ema=None, suffix="best", model_rank=model_rank)
                    is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, min_val_loss, metric_score, max_score, save_handle, self.patience, epoch)
                    message += f', val_loss: {total_val_loss:.4f}, ' + \
                        ", ".join([f'val_{k}: {v:.4f}' for k, v in metric_score.items() if k != "_judge_score"]) + \
                        f', val_judge_score ({_metric}): {_score:.4f}' + \
                        (f', Patience [{wait}/{self.patience}], min_val_judge_score: {min_val_loss:.4f}' if wait else '')
            else:
                is_early_stop = False
                
            end_time = time.time()
            message += f', {(end_time - start_time):.1f}s'
            logger.info(message)
            self.save_state_dict(model, dump_dir, ema, "last", model_rank)
            if is_early_stop:
                break

        if test_dataset is not None:
            y_preds, _, metric_score = self.predict(
                model=model, 
                dataset=test_dataset, 
                loss_terms=loss_terms, 
                dump_dir=dump_dir,  
                transform=transform, 
                epoch=epoch, 
                load_model=True,
                model_rank=model_rank
            )
        else:
            metric_score = None
        return y_preds, metric_score
    
    def _early_stop_choice(self, wait, min_loss, metric_score, max_score, save_handle, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, max_score, save_handle, patience, epoch)
    
    def predict(self, model, dataset, loss_terms, dump_dir, transform, epoch=1, load_model=False, model_rank=None):
        self._set_seed(self.seed)
        model = model.to(self.device).type(self.dtype)
        if load_model == True:
            from ..models import get_pretrain_path
            pretrain_path = get_pretrain_path(dump_dir, "best", model_rank)
            self.load_state_dict(model, pretrain_path, inference=True)
            
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
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
        return y_preds, val_loss, metric_score
