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
        if self.cv_pretrain_path is not None:
            self.model_params["pretrain"] = f"{self.cv_pretrain_path}/model_0.pth"
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
            if self.cv_pretrain_path is not None:
                model_dict = torch.load(self.model_params["pretrain"], map_location=self.trainer.device)["model_state_dict"]
                model.load_state_dict(model_dict, strict=False)
                logger.info(f"load model success from {self.model_params['pretrain']}!")
            print(model.__str__())
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model
    
    def run(self):
        logger.info("start training FF:{}".format(self.model_name))
        X = pd.DataFrame(self.features)
        y = pd.DataFrame(self.data['target'])
        y_pred = pd.DataFrame(columns=y.columns, index=y.index)
        for fold, idx in enumerate(self.splitter.split(X)):
            if self.splitter.cv:
                tr_idx, vl_idx = idx
            else:
                tr_idx, vl_idx, te_idx = idx
            X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
            X_valid, y_valid = X.iloc[vl_idx], y.iloc[vl_idx]
            train_dataset = FFDataset(X_train, y_train)
            valid_dataset = FFDataset(X_valid, y_valid)
            if self.splitter.cv:
                test_dataset = None
            else:
                X_test, y_test = X.iloc[te_idx], y.iloc[te_idx]
                test_dataset = FFDataset(X_test, y_test)
            if fold > 0:
                ### need to initalize model for next fold training
                if self.cv_pretrain_path:
                    self.model_params["pretrain"] = f"{self.cv_pretrain_path}/model_{fold}.pth"
                self.model = self._init_model(self.model_name, **self.model_params)
            try:
                _y_pred = self.trainer.fit_predict(
                    model=self.model, 
                    train_dataset=train_dataset, 
                    valid_dataset=valid_dataset, 
                    test_dataset=test_dataset,
                    loss_func=self.loss_func, 
                    dump_dir=self.dump_dir, 
                    fold=fold,
                    target_scaler=self.target_scaler, 
                    feature_name=self.feature_name,
                    cv=self.splitter.cv
                )
            except RuntimeError as e:
                logger.info("FF {0} failed...".format(self.model_name))
                self.is_success = False
                raise e
                return

            if self.splitter.cv:
                y_pred.iloc[vl_idx] = _y_pred
                logger.info ("fold {0}, result {1}".format(
                    fold,
                    self.metrics.cal_metric(
                        self.data['target_scaler'].inverse_transform(y_valid), self.data['target_scaler'].inverse_transform(_y_pred)
                        )
                    )
                )
            else:
                y_pred.iloc[te_idx] = _y_pred
        
        if self.splitter.cv:
            self.cv['pred'] = y_pred
            self.dump(self.cv['pred'], self.dump_dir, 'cv.data')
            self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(y), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        else:
            self.cv['pred'] = y_pred.iloc[te_idx]
            self.dump(self.cv['pred'], self.dump_dir, 'test.data')
            self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(y.iloc[te_idx]), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        self.dump(self.cv['metric'], self.dump_dir, 'metric.result')
        logger.info("{} FF metrics score: \n{}".format(self.model_str, self.cv['metric']))
        logger.info("{} FF done!".format(self.model_str))
        logger.info("Metric result saved!")
        logger.info("{} Model saved!".format(self.model_str))

    def evaluate(self, checkpoints_path=None):
        logger.info("start predict FF:{}".format(self.model_name))
        X = pd.DataFrame(self.features)
        y = pd.DataFrame(self.data['target'])
        test_dataset = FFDataset(X, y)
        if not self.splitter.cv:
            # model_path = os.path.join(checkpoints_path, f'model_0.pth')
            # self.model.load_state_dict(torch.load(model_path, map_location=self.trainer.device)['model_state_dict'])
            _y_pred, _, metric_score = self.trainer.predict(
                model=self.model, 
                dataset=test_dataset, 
                loss_func=self.loss_func, 
                dump_dir=self.dump_dir, 
                fold=0, 
                target_scaler=self.target_scaler, 
                epoch=1, 
                load_model=True
            )
        self.cv['test_pred'] = _y_pred
        self.cv['test_metrics'] = metric_score
        print(metric_score)

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