import os
import joblib
from .physnet import PhysNet
from ..utils import logger
import torch


SEP = "-"
FF_REGISTER = {
    "PhysNet": PhysNet
}
LOSS_REGISTER = {
    "q": torch.nn.MSELoss,
    "e": ...,
    "qe": ...
}


class FF:
    def __init__(self, data, features, trainer, model_str, loss_key, **params):
        self.data = data
        self.energy_scaler = self.data['energy_scaler']
        self.features = features
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.model_id, self.model_name, self.feature_name, self.task = self.model_str.split(SEP)[:4]
        self.model_params = params
        self.model_params['device'] = self.trainer.device
        self.metrics = self.trainer.metrics
        if loss_key is not None:
            self.loss_func = LOSS_REGISTER[self.task][loss_key]
        else:
            self.loss_func = LOSS_REGISTER[self.task]
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

    def collect_data(self, X, y, idx):
        pass
        # assert isinstance(y, np.ndarray), 'y must be numpy array'
        # if isinstance(X, np.ndarray):
        #     return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        # elif isinstance(X, list):
        #     return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        # else:
        #     raise ValueError('X must be numpy array or dict')
    
    def run(self):
        logger.info("start training NNModel:{}".format(self.model_name))
        # features_a, features_b = self.features
        # X_a = np.asarray(features_a)
        # X_b = np.asarray(features_b)
        # y = np.asarray(self.data['target'])
        # scaffold = np.asarray(self.data['scaffolds'])
        # y_pred = np.zeros_like(y.reshape(y.shape[0], self.num_classes)).astype(float)
        # for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(X_a, y, scaffold)):
        #     X_train_a, X_train_b, y_train = X_a[tr_idx], X_b[tr_idx], y[tr_idx]
        #     X_valid_a, X_valid_b, y_valid = X_a[te_idx], X_b[te_idx], y[te_idx]
        #     traindataset = NNDataset((X_train_a, X_train_b), y_train, self.feature_name, self.task)
        #     validdataset = NNDataset((X_valid_a, X_valid_b), y_valid, self.feature_name, self.task)
        #     if fold > 0:
        #         ### need to initalize model for next fold training
        #         if self.cv_pretrain_path:
        #             self.model_params["pretrain"] = f"{self.cv_pretrain_path}_{fold}.pth"
        #         self.model = self._init_model(self.model_name, **self.model_params)
        #     _y_pred = self.trainer.fit_predict(self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.dump_dir, fold, self.target_scaler, self.feature_name)
        #     # try:
        #     #     _y_pred = self.trainer.fit_predict(self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.dump_dir, fold, self.target_scaler, self.feature_name, pKa_mode=self.pKa_mode)
        #     # except:
        #     #     logger.info("NNModel {0} failed...".format(self.model_name))
        #     #     self.is_success = False
        #     #     return

        #     y_pred[te_idx] = _y_pred
        #     logger.info ("fold {0}, result {1}".format(
        #             fold,
        #             self.metrics.cal_metric(
        #                 self.data['target_scaler'].inverse_transform(y_valid), self.data['target_scaler'].inverse_transform(_y_pred)
        #             )
        #         )
        #     )

        # self.cv['pred'] = y_pred
        # self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(y), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        # self.dump(self.cv['pred'], self.dump_dir, 'cv.data')
        # self.dump(self.cv['metric'], self.dump_dir, 'metric.result')
        logger.info("{} NN model metrics score: \n{}".format(self.model_str, self.cv['metric']))
        logger.info("{} NN model done!".format(self.model_str))
        logger.info("Metric result saved!")
        logger.info("{} Model saved!".format(self.model_str))

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def evaluate(self, trainer=None,  checkpoints_path=None):
        pass
        # logger.info("start predict NNModel:{}".format(self.model_name))
        # testdataset = NNDataset(self.features, np.asarray(self.data['target']), self.feature_name, self.task)
        # for fold in range(self.splitter.n_splits):
        #     model_path = os.path.join(checkpoints_path, f'model_{fold}.pth')
        #     self.model.load_state_dict(torch.load(model_path, map_location=self.trainer.device)['model_state_dict'])
        #     _y_pred, _, __ = trainer.predict(self.model, testdataset, self.loss_func, self.activation_fn, self.dump_dir, fold, self.target_scaler, epoch=1, load_model=True, feature_name = self.feature_name)
        #     if fold == 0:
        #         y_pred = np.zeros_like(_y_pred)
        #     y_pred += _y_pred
        # y_pred /= self.splitter.n_splits
        # self.cv['test_pred'] = y_pred
        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)