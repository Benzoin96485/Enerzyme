from collections import defaultdict
import time
import os, logging, sys
from enerzyme.data.datatype import is_target
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
try:
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
except:
    pass
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from .splitter import Splitter
from ..data import is_atomic, is_int, is_pair_idx, get_tensor_rank, requires_grad
from ..utils import logger
from .metrics import Metrics


DTYPE_MAPPING = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "double": torch.float64,
    "single": torch.float32
}


class Trainer(object):
    def __init__(self, out_dir=None, metric_config=None, **params):
        self.out_dir = out_dir
        self.metrics = Metrics(metric_config)    
        self.splitter = Splitter(**params["Splitter"])
        self.seed = params.get('seed', 114514)
        self.learning_rate = float(params.get('learning_rate', 1e-3))
        self.batch_size = params.get('batch_size', 8)
        self.max_epochs = params.get('max_epochs', 1000)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.weight_decay = float(params.get('weight_decay', 1e-4))
        self.amsgrad = params.get('amsgrad', False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.data_in_memory = params.get("data_in_memory", False)
        self.dtype = DTYPE_MAPPING[params.get('dtype', "float32")]

    def decorate_batch_input(self, batch):
        features, targets = zip(*batch)
        batch_features = dict()
        batch_targets = dict()
        
        for k in features[0]:
            if is_atomic(k):
                batch_features[k] = torch.tensor(
                    np.concatenate([feature[k][:feature["N"]] for feature in features]), 
                    dtype=torch.long if is_int(k) else self.dtype,
                    requires_grad=requires_grad(k)
                ).to(self.device)
            elif not is_pair_idx(k):
                batch_features[k] = torch.tensor(
                    np.array([feature[k] for feature in features]), 
                    dtype=torch.long if is_int(k) else self.dtype,
                ).to(self.device)

        batch_idx_i = []
        batch_idx_j = []
        batch_seg = []
        count = 0
        for i, feature in enumerate(features):
            batch_idx_i.append(feature["idx_i"][:feature["N_pair"]] + count)
            batch_idx_j.append(feature["idx_j"][:feature["N_pair"]] + count)
            batch_seg.append(np.full(feature["N"], i, dtype=int))
            count += feature["N"]
        batch_features["N"] = [feature["N"] for feature in features]
        batch_features["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long).to(self.device)
        batch_features["idx_i"] = torch.tensor(np.concatenate(batch_idx_i), dtype=torch.long).to(self.device)
        batch_features["idx_j"] = torch.tensor(np.concatenate(batch_idx_j), dtype=torch.long).to(self.device)

        if targets is not None:
            for k in targets[0]:
                if is_atomic(k): 
                    batch_targets[k] = torch.tensor(
                        np.concatenate([target[k][:feature["N"]] for target in targets]), 
                        dtype=torch.long if is_int(k) else self.dtype
                    ).to(self.device)
                else:
                    batch_targets[k] = torch.tensor(
                        np.array([target[k] for target in targets]), 
                        dtype=torch.long if is_int(k) else self.dtype,
                    ).to(self.device)
        
        return batch_features, batch_targets
    
    def decorate_batch_output(self, output, features, targets):
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
    
    def _set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fit_predict(self, model, train_dataset, valid_dataset, loss_terms, dump_dir, transform, test_dataset=None):
        self._set_seed(self.seed)
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
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )

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
                optimizer.zero_grad() # Zero gradients
                # if self.scaler and self.device.type == 'cuda':
                #     with torch.cuda.amp.autocast():
                #         outputs = model(task=self.task, **net_input)
                #         loss = loss_func(outputs, net_target)
                # else:
                loss = 0
                with torch.set_grad_enabled(True):
                    output = model(**net_input)
                    for loss_term in loss_terms.values():
                        loss += loss_term(output, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
                )
                # if self.scaler and self.device.type == 'cuda':
                #     self.scaler.scale(loss).backward() # This is a replacement for loss.backward()
                #     self.scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                #     clip_grad_norm_(model.parameters(), self.max_norm)  # Clip the norm of the gradients to max_norm.
                #     self.scaler.step(optimizer) # This is a replacement for optimizer.step()
                #     self.scaler.update()
                # else:
                loss.backward()
                clip_grad_norm_(model.parameters(), self.max_norm)
                optimizer.step()
                scheduler.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)
            y_preds, val_loss, metric_score = self.predict(
                model=model, 
                dataset=valid_dataset, 
                loss_terms=loss_terms, 
                dump_dir=dump_dir, 
                transform=transform, 
                epoch=epoch, 
                load_model=False,
            )
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = metric_score["_judge_score"]
            _metric = str(self.metrics)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, self.patience, epoch)
            message = f'Epoch [{epoch+1}/{self.max_epochs}] train_loss: {total_trn_loss:.4f}, ' + \
                f'val_loss: {total_val_loss:.4f}, ' + \
                ", ".join([f'val_{k}: {v:.4f}' for k, v in metric_score.items() if k != "_judge_score"]) + \
                f', val_judge_score ({_metric}): {_score:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}, ' + \
                f'{(end_time - start_time):.1f}s' + \
                (f', Patience [{wait}/{self.patience}], min_val_judge_score: {min_val_loss:.4f}' if wait else '')
            logger.info(message)
            if is_early_stop:
                break
        
        y_preds, _, _ = self.predict(
            model=model, 
            dataset=test_dataset, 
            loss_terms=loss_terms, 
            dump_dir=dump_dir,  
            transform=transform, 
            epoch=epoch, 
            load_model=True
        )
        return y_preds
    
    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, max_score, model, dump_dir, patience, epoch)
    
    # def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
    #     is_early_stop = False
    #     if loss <= min_loss :
    #         min_loss = loss
    #         wait = 0
    #         info = {'model_state_dict': model.state_dict()}
    #         os.makedirs(dump_dir, exist_ok=True)
    #         torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
    #     elif loss >= min_loss:
    #         wait += 1
    #         if wait == self.patience:
    #             logger.warning(f'Early stopping at epoch: {epoch+1}')
    #             is_early_stop = True
    #     return is_early_stop, min_loss, wait
    
    def predict(self, model, dataset, loss_terms, dump_dir, transform, epoch=1, load_model=False):
        self._set_seed(self.seed)
        model = model.to(self.device).type(self.dtype)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_best.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)["model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info(f"load model success from {load_model_path}!")
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
            output = model(**net_input)
            # Get model outputs
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

        if transform is not None:
            transform.inverse_transform(y_preds)
            transform.inverse_transform(y_truths)
        metric_score = self.metrics.cal_metric(y_truths, y_preds)
        if load_model and "_judge_score" in metric_score:
            metric_score.pop("_judge_score")
        batch_bar.close()
        return y_preds, val_loss, metric_score