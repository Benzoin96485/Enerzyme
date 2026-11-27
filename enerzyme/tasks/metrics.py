from functools import partial
from typing import Dict, Callable, Tuple, List, Union
import numpy as np
from ..data import is_atomic, get_tensor_rank
from ..utils.base_logger import logger


def _metric_arrays(label, prediction, target_name):
    y_true = label.get(target_name, [])
    if len(y_true) == 0:
        return None
    concatenate = is_atomic(target_name) or get_tensor_rank(target_name)
    stack = np.concatenate if concatenate else np.asarray
    y_true, y_pred = stack(y_true), stack(prediction[target_name])
    if y_pred.ndim == y_true.ndim + 1:
        y_pred = np.mean(y_pred, axis=-1)
    # sklearn treats [N] and [N, 1] as the same shape. Explicit columns keep
    # NumPy subtraction from broadcasting a scalar prediction to [N, N].
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Inconsistent label and prediction shapes for {target_name}: "
                         f"{y_true.shape} and {y_pred.shape}")
    return y_true, y_pred


def _single_metric(label, prediction, target_name, *, metric_name):
    arrays = _metric_arrays(label, prediction, target_name)
    if arrays is None:
        return 0.0
    if metric_name == "mae":
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(*arrays)
    try:
        from sklearn.metrics import root_mean_squared_error
    except ImportError:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(*arrays, squared=False)
    return root_mean_squared_error(*arrays)


def build_single_metric(metric_str: str):
    if metric_str not in ("mae", "rmse"):
        raise ValueError(f"Unknown metric: {metric_str}")
    # This callable also travels to DataLoader workers under forkserver/spawn.
    return partial(_single_metric, metric_name=metric_str)


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
            raw_metric_score[target_metric] = self.cal_single_metric(label, predict, *target_metric.rsplit("_", 1))
        raw_metric_score["_judge_score"] = self.cal_judge_score(raw_metric_score)
        return raw_metric_score

    def accumulate_partials(self, label, predict):
        """Preserve each output component until its global MAE or RMSE is known."""
        partials = {}
        for key in self.metric_config:
            target, metric = key.rsplit("_", 1)
            arrays = _metric_arrays(label, predict, target)
            if arrays is None:
                continue
            truth, pred = arrays
            error = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
            values = np.square(error) if metric == "rmse" else np.abs(error)
            partials[key] = (values.sum(axis=0), len(values))
        return partials

    @staticmethod
    def merge_partials(partials_list):
        merged = {}
        for partials in partials_list:
            for key, (sums, count) in partials.items():
                if key in merged:
                    previous, previous_count = merged[key]
                    if np.shape(sums) != np.shape(previous):
                        raise ValueError(f"Inconsistent output shape across shards for metric {key}")
                    sums, count = previous + sums, previous_count + count
                merged[key] = (sums, count)
        return merged

    def cal_metric_from_partials(self, partials):
        scores = {}
        for key in self.metric_config:
            sums, count = partials.get(key, (0.0, 0))
            mean = sums / count if count else 0.0
            if key.endswith("_rmse"):
                mean = np.sqrt(mean)
            scores[key] = float(np.mean(mean))
        scores["_judge_score"] = self.cal_judge_score(scores)
        return scores

    def distributed_metric(self, label, predict):
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self.cal_metric(label, predict)
        partials = self.accumulate_partials(label, predict)
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, partials)
        return self.cal_metric_from_partials(self.merge_partials(shards))

    def _early_stop_choice(self, wait: int, best_score: float, metric_score: Dict[str, float], save_handle: Callable, patience: int, epoch: int) -> Tuple[bool, float, int]:
        judge_score = (metric_score["_judge_score"] if "_judge_score" in metric_score
                       else self.cal_judge_score(metric_score))
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
            if wait >= patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait, saved
