from typing import Dict, Callable, Tuple, List, Union, Optional
import numpy as np
from ..data import is_atomic, get_tensor_rank
from ..utils.base_logger import logger


def build_single_metric(metric_str: str) -> Callable[[Dict[str, Union[List, np.ndarray]], Dict[str, Union[List, np.ndarray]], str], Optional[float]]:
    if metric_str == "rmse":
        try:
            from sklearn.metrics import root_mean_squared_error
        except ImportError:
            from sklearn.metrics import mean_squared_error
            metric_func = lambda x, y: mean_squared_error(x, y, squared=False)
        else:
            metric_func = lambda x, y: root_mean_squared_error(x, y)
    elif metric_str == "mae":
        from sklearn.metrics import mean_absolute_error
        metric_func = lambda x, y: mean_absolute_error(x, y)
    else:
        raise ValueError(f"Unknown metric: {metric_str}")
    
    def metric(label: Dict[str, Union[List, np.ndarray]], prediction: Dict[str, Union[List, np.ndarray]], target_name: str) -> Optional[float]:
        y_true = label.get(target_name, [])
        if not y_true:
            return 0
        y_pred = prediction[target_name]
        if is_atomic(target_name) or get_tensor_rank(target_name):
            y_trues, y_preds = np.concatenate(y_true), np.concatenate(y_pred)
            if y_preds.ndim == y_trues.ndim + 1:
                score = metric_func(y_trues, np.mean(y_preds, axis=-1))
            else:
                score = metric_func(y_trues, y_preds)
        else:
            y_trues, y_preds = np.array(y_true), np.array(y_pred)
            if y_preds.ndim == y_trues.ndim + 1:
                score = metric_func(y_trues, np.mean(y_preds, axis=-1))
            else:
                score = metric_func(y_true, y_pred)
        return score
    return metric


class Metrics(object):
    def __init__(self, metric_config: Dict=dict()) -> None:
        self.metric_config = dict()
        self.metrics_register = dict()
        for target, metrics in metric_config.items():
            for metric, weight in metrics.items():
                self.metric_config[f"{target}_{metric}"] = weight
                if metric not in self.metrics_register:
                    self.metrics_register[metric] = build_single_metric(metric)

    def __str__(self) -> str:
        terms = []
        for target_metric, weight in self.metric_config.items():
            if weight == 1:
                terms.append(target_metric)
            elif weight is not None and weight != 0:
                terms.append(f"{weight:.2f} * {target_metric}")
        return " + ".join(terms)

    def cal_single_metric(self, label: Dict[str, Union[List, np.ndarray]], prediction: Dict[str, Union[List, np.ndarray]], target_name: str, metric_name: str) -> float:
        return self.metrics_register[metric_name](label, prediction, target_name)

    def cal_judge_score(self, raw_metric_score: Dict[str, float]) -> float:
        judge_score = 0
        for target_metric, weight in self.metric_config.items():
            if weight is not None and weight != 0:
                judge_score += weight * raw_metric_score[target_metric]
        return judge_score

    def cal_metric(self, label: Dict[str, Union[List, np.ndarray]], predict: Dict[str, Union[List, np.ndarray]]) -> Dict[str, float]:
        raw_metric_score = dict()
        for target_metric in self.metric_config:
            raw_metric_score[target_metric] = self.cal_single_metric(label, predict, *target_metric.split("_"))
        raw_metric_score["_judge_score"] = self.cal_judge_score(raw_metric_score)
        return raw_metric_score

    def _early_stop_choice(self, wait: int, best_score: float, metric_score: Dict[str, float], save_handle: Callable, patience: int, epoch: int) -> Tuple[bool, float, int]:
        judge_score = metric_score.get("_judge_score", self.cal_judge_score(metric_score))
        return self._judge_early_stop_decrease(wait, judge_score, best_score, save_handle, patience, epoch)

    def _judge_early_stop_decrease(self, wait: int, score: float, min_score: float, save_handle: Callable, patience: int, epoch: int) -> Tuple[bool, float, int]:
        is_early_stop = False
        saved = False
        if score <= min_score:
            min_score = score
            wait = 0
            save_handle(best_score=score, best_epoch=epoch, epoch=epoch)
            saved = True
        elif score >= min_score:
            wait += 1
            if wait == patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait, saved
