from collections import defaultdict
import os
from enerzyme import data
from enerzyme.models.physnet.loss import NHLoss
import joblib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .physnet import PhysNet, NHLoss
from .spookynet import SpookyNet
from .loss import MAELoss, MSELoss
from ..utils import logger


SEP = "-"
FF_REGISTER = {
    "PhysNet": PhysNet,
    "SpookyNet": SpookyNet
}
LOSS_REGISTER = {
    "nh_penalty": NHLoss,
    "mae": MAELoss,
    "mse": MSELoss
}


class FF:
    def __init__(self, 
        datahub, trainer, model_str, loss, architecture, build_params,
        pretrain_path=None, **params
    ):
        self.datahub = datahub
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.architecture = architecture
        self.build_params = build_params
        if pretrain_path is not None:
            if os.path.isdir(pretrain_path):
                self.pretrain_path = f"{pretrain_path}/model_best.pth"
            elif os.path.isfile(pretrain_path):
                self.pretrain_path = pretrain_path
            else:
                raise FileNotFoundError(f"Pretrained model not found at {self.pretrain_path}")
        else:
            self.pretrain_path = None
        self.metrics = self.trainer.metrics
        self.loss_terms = {k: LOSS_REGISTER[k](**v) for k, v in loss.items()}
        self.out_dir = self.trainer.out_dir
        self.dump_dir = os.path.join(self.out_dir, self.model_str)
        self.trainer._set_seed(self.trainer.seed)
        self.model = self._init_model(self.build_params)
        self.is_success = True
    
    def _init_model(self, build_params):
        if self.architecture in FF_REGISTER:
            model = FF_REGISTER[self.architecture](**build_params)
            if self.pretrain_path is not None:
                model_dict = torch.load(self.pretrain_path, map_location=self.trainer.device)["model_state_dict"]
                model.load_state_dict(model_dict, strict=False)
                logger.info(f"load model success from {self.pretrain_path}!")
            print(model.__str__())
        else:
            raise KeyError('Unknown model: {}'.format(self.architecture))
        return model
    
    def run(self):
        logger.info("start training FF:{}".format(self.model_str))
        X = self.datahub.features
        y = self.datahub.targets
        split = self.splitter.split(X, preload_path=self.datahub.preload_path)
        tr_idx = split["training"]
        vl_idx = split["validation"]
        te_idx = split["test"]
        train_dataset = FFDataset(X, y, tr_idx, self.trainer.data_in_memory)
        valid_dataset = FFDataset(X, y, vl_idx, True)
        if len(te_idx) > 0:
            test_dataset = FFDataset(X, y, te_idx, True)
            y_test = test_dataset.targets
            y_test["Za"] = test_dataset.features["Za"]
        try:
            y_pred = self.trainer.fit_predict(
                model=self.model, 
                train_dataset=train_dataset, 
                valid_dataset=valid_dataset, 
                test_dataset=test_dataset,
                loss_terms=self.loss_terms, 
                transform=self.datahub.transform,
                dump_dir=self.dump_dir
            )
        except RuntimeError as e:
            logger.info("FF {0} failed...".format(self.model_str))
            self.is_success = False
            raise e
        self.datahub.transform.inverse_transform(y_test)
        self.dump(y_pred, self.dump_dir, 'test.data')
        metric_score = self.metrics.cal_metric(y_test, y_pred)
        self.dump(metric_score, self.dump_dir, 'metric.result')
        logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
        logger.info("{} FF done!".format(self.model_str))
        logger.info("Metric result saved!")
        logger.info("{} Model saved!".format(self.model_str))

    def evaluate(self):
        logger.info("start predict FF:{}".format(self.model_str))
        X = self.datahub.features
        y = self.datahub.targets
        test_dataset = FFDataset(X, y, data_in_memory=True)
            # model_path = os.path.join(checkpoints_path, f'model_0.pth')
            # self.model.load_state_dict(torch.load(model_path, map_location=self.trainer.device)['model_state_dict'])
        y_pred, _, metric_score = self.trainer.predict(
            model=self.model, 
            dataset=test_dataset, 
            loss_terms=self.loss_terms, 
            dump_dir=self.dump_dir, 
            transform=self.datahub.transform, 
            epoch=1, 
            load_model=True
        )
        # self.dump(y_pred, self.dump_dir, 'test.data')
        # self.dump(metric_score, self.dump_dir, 'metric.result')
        logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
        logger.info("{} FF done!".format(self.model_str))
        logger.info("Metric result saved!")
        return y_pred, pd.DataFrame(metric_score, index=[self.model_str])

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)
        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

class FFDataset(Dataset):
    def __init__(self, features, targets, indices=None, data_in_memory=True):
        self.data_in_memory = data_in_memory
        self.indices = indices if indices is not None else np.arange(0, len(features["Za"]))
        if data_in_memory:
            self.features = {k: np.array([v[idx] for idx in self.indices]) for k, v in features.items()}
            self.targets = {k: np.array([v[idx] for idx in self.indices]) for k, v in targets.items()}
            self._getitem = lambda idx: ({k: v[idx] for k, v in self.features.items()}, {k: v[idx] for k, v in self.targets.items()})
        else:
            self.features = features
            self.targets = targets
            self._getitem = lambda idx: ({k: v[self.indices[idx]] for k, v in self.features.items()}, {k: v[self.indices[idx]] for k, v in self.targets.items()})

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self._getitem(idx)