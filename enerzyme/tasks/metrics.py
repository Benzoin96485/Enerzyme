import os
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ..data import is_atomic, get_tensor_rank
from ..utils.base_logger import logger


def rmse(label, prediction, target_name=None):
    y_true = label[target_name]
    y_pred = prediction[target_name]
    if is_atomic(target_name) or get_tensor_rank(target_name):
        score = mean_squared_error(np.concatenate(y_true), np.concatenate(y_pred), squared=False)
    else:
        score = mean_squared_error(y_true, y_pred, squared=False)
    return score


def mae(label, prediction, target_name=None):
    y_true = label[target_name]
    y_pred = prediction[target_name]
    if is_atomic(target_name) or get_tensor_rank(target_name):
        score = mean_absolute_error(np.concatenate(y_true), np.concatenate(y_pred), squared=False)
    else:
        score = mean_absolute_error(y_true, y_pred, squared=False)
    return score


METRICS_REGISTER = {
    "rmse": rmse,
    "mae": mae
}


class Metrics(object):
    def __init__(self, metric_config: Dict=dict()) -> None:
        self.metric_config = dict()
        for target, metrics in metric_config.items():
            for metric, weight in metrics.items():
                self.metric_config[f"{target}_{metric}"] = weight

    def __str__(self):
        terms = []
        for target_metric, weight in self.metric_config.items():
            if weight == 1:
                terms.append(target_metric)
            elif weight is not None and weight != 0:
                terms.append(f"{weight:.2f} * {target_metric}")
        return " + ".join(terms)

    def cal_single_metric(self, label, prediction, target_name, metric_name):
        return METRICS_REGISTER[metric_name](label, prediction, target_name)

    def cal_judge_score(self, raw_metric_score):
        judge_score = 0
        for target_metric, weight in self.metric_config.items():
            if weight is not None and weight != 0:
                judge_score += weight * raw_metric_score[target_metric]
        return judge_score

    def cal_metric(self, label, predict):
        raw_metric_score = dict()
        for target_metric in self.metric_config:
            raw_metric_score[target_metric] = self.cal_single_metric(label, predict, *target_metric.split("_"))
        raw_metric_score["_judge_score"] = self.cal_judge_score(raw_metric_score)
        return raw_metric_score

    def _early_stop_choice(self, wait, min_score, metric_score, max_score, save_handle, patience, epoch):
        judge_score = metric_score.get("_judge_score", self.cal_judge_score(metric_score))
        is_early_stop, min_score, wait = self._judge_early_stop_decrease(wait, judge_score, min_score, save_handle, patience, epoch)
        return is_early_stop, min_score, wait, max_score

    def _judge_early_stop_decrease(self, wait, score, min_score, save_handle, patience, epoch):
        is_early_stop = False
        if score <= min_score:
            min_score = score
            wait = 0
            save_handle()
        elif score >= min_score:
            wait += 1
            if wait == patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait
