import time
import os, logging, sys
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
from .split import Splitter
from ..utils import logger
from ..utils.metrics import Metrics


class Trainer(object):
    def __init__(self, task=None, metrics_str=None, metrics_weight=None, out_dir=None, **params):
        self.task = task
        self.out_dir = out_dir
        self.metrics = Metrics(metrics_str, metrics_weight)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        ### init common params ###
        self.split_method = params.get('Split').get('method', "fold_random")
        self.split_params = params.get('Split').get('params')
        self.seed = params.get('Common').get('seed', 114514)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, **self.split_params)
        self.logger_level = int(params.get('Common').get('logger_level'))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('FFtrainer').get('learning_rate', 1e-4))
        self.batch_size = params.get('FFtrainer').get('batch_size', 32)
        self.max_epochs = params.get('FFtrainer').get('max_epochs', 50)
        self.warmup_ratio = params.get('FFtrainer').get('warmup_ratio', 0.1)
        self.patience = params.get('FFtrainer').get('patience', 10)
        self.max_norm = params.get('FFtrainer').get('max_norm', 1.0)
        self.cuda = params.get('FFtrainer').get('cuda', False)
        self.amp = params.get('FFtrainer').get('amp', False)
        self.weight_decay = params.get('FFtrainer').get('weight_decay', 1e-4)
        self.amsgrad = params.get('FFtrainer').get('amsgrad', False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' and self.amp == True else None
    
    def decorate_batch(self, batch):
        batch_input, batch_target = batch
        for k, v in batch_input.items():
            if isinstance(v, torch.Tensor):
                batch_input[k] = v.to(self.device)
        for k, v in batch_target.items():
            if isinstance(v, torch.Tensor):
                batch_target[k] = v.to(self.device)
        return batch_input, batch_target
    
    def decorate_batch_output(self, output, target):
        y_pred = dict()
        y_truth = dict()
        for k in output:
            if k in ["Qa", "F", "P"]:
                y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), output[k]))
                y_truth[k] = list(map(lambda x: x.detach().cpu().numpy(), target[k]))
            elif k in ["E", "Q"]:
                y_pred[k] = output[k].detach().cpu().numpy()
                y_truth[k] = target[k].detach().cpu().numpy()
        y_pred["atom_type"] = target["atom_type"]
        y_truth["atom_type"] = target["atom_type"]
        return pd.DataFrame(y_pred), pd.DataFrame(y_truth)
    
    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, dump_dir, fold, target_scaler, feature_name, test_dataset=None, cv=True):
        model = model.to(self.device)
        collate_fn = model.batch_collate_fn
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
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
                net_input, net_target = self.decorate_batch(batch)
                optimizer.zero_grad() # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(task=self.task, **net_input)
                        loss = loss_func(outputs, net_target)
                else:
                    with torch.set_grad_enabled(True):
                        outputs = model(task=self.task, **net_input)
                        loss = loss_func(outputs, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
                )
                if self.scaler and self.device.type == 'cuda':
                    self.scaler.scale(loss).backward() # This is a replacement for loss.backward()
                    self.scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
                    clip_grad_norm_(model.parameters(), self.max_norm)  # Clip the norm of the gradients to max_norm.
                    self.scaler.step(optimizer) # This is a replacement for optimizer.step()
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    optimizer.step()
                scheduler.step()
                batch_bar.update()
            torch.cuda.empty_cache()
            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)
            y_preds, val_loss, metric_score = self.predict(
                model=model, 
                dataset=valid_dataset, 
                loss_func=loss_func, 
                dump_dir=dump_dir, 
                target_scaler=target_scaler, 
                epoch=epoch, 
                load_model=False,
                fold=fold
            )
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = metric_score["_judge_score"]
            _metric = str(self.metrics)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(wait, total_val_loss, min_val_loss, metric_score, max_score, model, dump_dir, fold, self.patience, epoch)
            message = f'Epoch [{epoch+1}/{self.max_epochs}] train_loss: {total_trn_loss:.4f}, ' + \
                f'val_loss: {total_val_loss:.4f}, ' + \
                ", ".join([f'val_{k}: {v:.4f}' for k, v in metric_score.items() if k != "_judge_score"]) + \
                f', val_judge_score ({_metric}): {_score:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}, ' + \
                f'{(end_time - start_time):.1f}s' + \
                (f', Patience [{wait}/{self.patience}], min_val_judge_score: {min_val_loss:.4f}' if wait else '')
            logger.info(message)
            if is_early_stop:
                break
        torch.cuda.empty_cache()   
        if cv:
            y_preds, _, _ = self.predict(
                model=model, 
                dataset=valid_dataset, 
                loss_func=loss_func, 
                dump_dir=dump_dir, 
                fold=fold, 
                target_scaler=target_scaler, 
                epoch=epoch, 
                load_model=True
            )
        else:
            y_preds, _, _ = self.predict(
                model=model, 
                dataset=test_dataset, 
                loss_func=loss_func, 
                dump_dir=dump_dir, 
                fold=fold, 
                target_scaler=target_scaler, 
                epoch=epoch, 
                load_model=True
            )
        return y_preds
    
    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
    
    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss :
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait
    
    def predict(self, model, dataset, loss_func, dump_dir, fold, target_scaler=None, epoch=1, load_model=False):
        model = model.to(self.device)
        collate_fn = model.batch_collate_fn
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            model_dict = torch.load(load_model_path, map_location=self.device)["model_state_dict"]
            model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch)
            outputs = model(task=self.task, **net_input)
            # Get model outputs
            with torch.no_grad():
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            outputs, net_target = model.batch_output_collate_fn(outputs, net_target)
            y_pred, y_truth = self.decorate_batch_output(outputs, net_target)
            y_preds.append(y_pred)
            y_truths.append(y_truth)
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1)))
                )
            batch_bar.update()
        y_preds = pd.concat(y_preds)
        y_truths = pd.concat(y_truths)
        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(inverse_y_truths, inverse_y_preds) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(y_truths, y_preds) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score