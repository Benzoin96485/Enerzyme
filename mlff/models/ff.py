import os
import joblib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .physnet import PhysNet, MSE_nh_Loss
from ..utils import logger


SEP = "-"
FF_REGISTER = {
    "PhysNet": PhysNet
}
LOSS_REGISTER = {
    "mse_nh": MSE_nh_Loss
}


class FF:
    def __init__(self, data, features, trainer, model_str, loss_param, **params):
        self.data = data
        self.target_scaler = self.data['target_scaler']
        self.features = features
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.model_id, self.model_name, self.feature_name, self.task = self.model_str.split(SEP)[:4]
        self.model_params = params
        self.model_params['device'] = self.trainer.device
        self.cv_pretrain_path = self.model_params.get('cv_pretrain', None)
        if self.cv_pretrain_path:
            self.model_params["pretrain"] = f"{self.cv_pretrain_path}_0.pth"
        self.cv = dict()
        self.metrics = self.trainer.metrics
        self.loss_func = LOSS_REGISTER[loss_param["key"]](**loss_param["params"])
        self.out_dir = self.trainer.out_dir
        self.dump_dir = os.path.join(self.out_dir, self.model_str)
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(self.model_name, **self.model_params)
        self.is_success = True
    
    def _init_model(self, model_name, **params):
        if model_name in FF_REGISTER:
            model = FF_REGISTER[model_name](**params)
            print(model.__str__())
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model
    
    def run(self):
        logger.info("start training FF:{}".format(self.model_name))
        X = pd.DataFrame(self.features)
        y = pd.DataFrame(self.data['target'])
        y_pred = pd.DataFrame(columns=y.columns, index=y.index)
        for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(X)):
            X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
            X_valid, y_valid = X.iloc[te_idx], y.iloc[te_idx]
            train_dataset = FFDataset(X_train, y_train)
            valid_dataset = FFDataset(X_valid, y_valid)
            if fold > 0:
                ### need to initalize model for next fold training
                if self.cv_pretrain_path:
                    self.model_params["pretrain"] = f"{self.cv_pretrain_path}_{fold}.pth"
                self.model = self._init_model(self.model_name, **self.model_params)
            try:
                _y_pred = self.trainer.fit_predict(
                    model=self.model, 
                    train_dataset=train_dataset, 
                    valid_dataset=valid_dataset, 
                    loss_func=self.loss_func, 
                    dump_dir=self.dump_dir, 
                    fold=fold,
                    target_scaler=self.target_scaler, 
                    feature_name=self.feature_name
                )
            except RuntimeError as e:
                logger.info("FF {0} failed...".format(self.model_name))
                self.is_success = False
                raise e
                return

            y_pred.iloc[te_idx] = _y_pred
            logger.info ("fold {0}, result {1}".format(
                    fold,
                    self.metrics.cal_metric(
                        self.data['target_scaler'].inverse_transform(y_valid), self.data['target_scaler'].inverse_transform(_y_pred)
                    )
                )
            )

        self.cv['pred'] = y_pred
        self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(y), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        self.dump(self.cv['pred'], self.dump_dir, 'cv.data')
        self.dump(self.cv['metric'], self.dump_dir, 'metric.result')
        logger.info("{} FF metrics score: \n{}".format(self.model_str, self.cv['metric']))
        logger.info("{} FF done!".format(self.model_str))
        logger.info("Metric result saved!")
        logger.info("{} Model saved!".format(self.model_str))

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)
        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

class FFDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, idx):
        return self.feature.iloc[idx], self.label.iloc[idx]

    def __len__(self):
        return len(self.feature)