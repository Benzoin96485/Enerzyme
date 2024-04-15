import numpy as np
import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .base_logger import logger


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.concatenate(y_true.to_list()), np.concatenate(y_pred.to_list())))


METRICS_REGISTER = {
    "rmse": rmse
}


class Metrics(object):
    def __init__(self, metrics_str, metrics_weight=None, task=None, **params):
        self.task = task
        self.threshold = np.arange(0, 1., 0.1)
        self.metrics_str = metrics_str
        self.metrics_weight = metrics_weight

    def __str__(self):
        pre_list = []
        for i, metric_str in enumerate(self.metrics_str):
            if self.metrics_weight[i] > 0:
                if self.metrics_weight[i] == 1:
                    pre_list.append(metric_str)
                else:
                    pre_list.append(f"{self.metrics_weight[i]:.2f} * {metric_str}")
        return " + ".join(pre_list)
       
    def cal_single_metric(self, label, predict, target_name, metric_str):
        return METRICS_REGISTER[metric_str](label[target_name], predict[target_name])
    
    def cal_judge_score(self, raw_metric_score):
        judge_score = 0
        for i, metric_str in enumerate(self.metrics_str):
            judge_score += self.metrics_weight[i] * raw_metric_score[metric_str]
        return judge_score

    def cal_metric(self, label, predict):
        res_dict = dict()
        for metric_str in self.metrics_str:
            res_dict[metric_str] = self.cal_single_metric(label, predict, *metric_str.split("_"))
        res_dict["_judge_score"] = self.cal_judge_score(res_dict)
        return res_dict

    def _early_stop_choice(self, wait, min_score, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        judge_score = metric_score.get("_judge_score", self.cal_judge_score(metric_score))
        is_early_stop, min_score, wait = self._judge_early_stop_decrease(wait, judge_score, min_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_score, wait, max_score

    def _judge_early_stop_decrease(self, wait, score, min_score, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if score <= min_score :
            min_score = score
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif score >= min_score:
            wait += 1
            if wait == patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait