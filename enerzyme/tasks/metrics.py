from typing import Dict, Callable, Tuple, List, Union, Optional, Iterable
import numpy as np
from ..data import is_atomic, get_tensor_rank
from ..utils.base_logger import logger


def _split_target_metric(target_metric: str) -> Tuple[str, str]:
    """Split ``Ea_rmse`` → ``("Ea", "rmse")`` (metric is the last ``_`` segment)."""
    target_name, metric_name = target_metric.rsplit("_", 1)
    return target_name, metric_name


def _arrays_for_metric(
    label: Dict[str, Union[List, np.ndarray]],
    prediction: Dict[str, Union[List, np.ndarray]],
    target_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Flatten label/prediction lists into comparable arrays, or None if empty."""
    y_true = label.get(target_name, [])
    if not y_true:
        return None
    y_pred = prediction[target_name]
    if is_atomic(target_name) or get_tensor_rank(target_name):
        y_trues, y_preds = np.concatenate(y_true), np.concatenate(y_pred)
    else:
        y_trues, y_preds = np.array(y_true), np.array(y_pred)
    if y_preds.ndim == y_trues.ndim + 1:
        y_preds = np.mean(y_preds, axis=-1)
    return y_trues, y_preds


def build_single_metric(metric_str: str) -> Callable[[Dict[str, Union[List, np.ndarray]], Dict[str, Union[List, np.ndarray]], str], Optional[float]]:
    return _ArrayMetric(metric_str)


class _ArrayMetric:
    """Top-level callable so Metrics survives forkserver pickle (no nested locals)."""

    def __init__(self, metric_str: str) -> None:
        if metric_str not in ("rmse", "mae"):
            raise ValueError(f"Unknown metric: {metric_str}")
        self.metric_str = metric_str

    def __call__(
        self,
        label: Dict[str, Union[List, np.ndarray]],
        prediction: Dict[str, Union[List, np.ndarray]],
        target_name: str,
    ) -> Optional[float]:
        arrays = _arrays_for_metric(label, prediction, target_name)
        if arrays is None:
            return 0
        y_trues, y_preds = arrays
        if self.metric_str == "rmse":
            try:
                from sklearn.metrics import root_mean_squared_error
            except ImportError:
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_trues, y_preds, squared=False)
            return root_mean_squared_error(y_trues, y_preds)
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_trues, y_preds)


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
            raw_metric_score[target_metric] = self.cal_single_metric(
                label, predict, *_split_target_metric(target_metric)
            )
        raw_metric_score["_judge_score"] = self.cal_judge_score(raw_metric_score)
        return raw_metric_score

    def accumulate_partials(
        self,
        label: Dict[str, Union[List, np.ndarray]],
        predict: Dict[str, Union[List, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        """Per-metric additive partials ``{error_sum, count}`` for DDP gather.

        For ``rmse``, ``error_sum`` is SSE; for ``mae``, SAE. Global metrics are
        recovered via :meth:`cal_metric_from_partials` after summing across ranks.
        """
        partials: Dict[str, Dict[str, float]] = {}
        for target_metric in self.metric_config:
            target_name, metric_name = _split_target_metric(target_metric)
            arrays = _arrays_for_metric(label, predict, target_name)
            if arrays is None:
                partials[target_metric] = {"error_sum": 0.0, "count": 0.0}
                continue
            y_trues, y_preds = arrays
            diff = y_trues.astype(np.float64) - y_preds.astype(np.float64)
            if metric_name == "rmse":
                error_sum = float(np.sum(diff * diff))
            elif metric_name == "mae":
                error_sum = float(np.sum(np.abs(diff)))
            else:
                raise ValueError(f"Unknown metric for partials: {metric_name}")
            partials[target_metric] = {
                "error_sum": error_sum,
                "count": float(diff.size),
            }
        return partials

    @staticmethod
    def merge_partials(
        partials_list: Iterable[Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, float]]:
        """Sum ``error_sum`` / ``count`` across rank-local partial dicts."""
        merged: Dict[str, Dict[str, float]] = {}
        for partials in partials_list:
            for key, stats in partials.items():
                if key not in merged:
                    merged[key] = {"error_sum": 0.0, "count": 0.0}
                merged[key]["error_sum"] += float(stats["error_sum"])
                merged[key]["count"] += float(stats["count"])
        return merged

    def cal_metric_from_partials(
        self, partials: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Build metric scores from gathered ``{error_sum, count}`` partials."""
        raw_metric_score: Dict[str, float] = {}
        for target_metric in self.metric_config:
            _, metric_name = _split_target_metric(target_metric)
            stats = partials.get(target_metric, {"error_sum": 0.0, "count": 0.0})
            count = float(stats["count"])
            error_sum = float(stats["error_sum"])
            if count <= 0:
                raw_metric_score[target_metric] = 0.0
            elif metric_name == "rmse":
                raw_metric_score[target_metric] = float(np.sqrt(error_sum / count))
            elif metric_name == "mae":
                raw_metric_score[target_metric] = float(error_sum / count)
            else:
                raise ValueError(f"Unknown metric for partials: {metric_name}")
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
