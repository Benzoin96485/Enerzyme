import os
from typing import Dict, Optional, List, Literal, Any
import pandas as pd
from .utils import YamlHandler, logger
from .data.datahub import DataHub
from .tasks.trainer import Trainer
from .models import ModelHub
from .models import BaseFFLauncher

class FFPredict:
    def __init__(self, model_dir: str, output_dir: str, config_path: Optional[str] = None, model_config_path: Optional[str] = None, simple_predict: bool=False) -> None:
        self.model_dir = model_dir
        self.output_dir = model_dir if output_dir is None else output_dir
        self.new_data = True
        self.simple_predict = simple_predict

        model_config_path = os.path.join(model_dir, 'config.yaml') if model_config_path is None else model_config_path
        config = YamlHandler(model_config_path).read_yaml()
        new_config = YamlHandler(config_path).read_yaml()
        if config_path is not None:
            new_datahub_config = new_config.get("Datahub", dict())
            if new_datahub_config:
                if "datasets" in config.Datahub and "datasets" not in new_datahub_config:
                    config.Datahub.pop("datasets")
                for k, v in new_datahub_config.items():
                    if k not in ["transforms", "neighbor_list", "global_transforms"]:
                        config.Datahub[k] = v
                if "targets" not in new_datahub_config:
                    config.Datahub.pop("targets")
            else:
                self.new_data = False
            new_metric_config = new_config.get("Metric", dict())
            if new_metric_config:
                config.Metric = new_metric_config
                for ff in config.Modelhub.internal_FFs.values():
                    if "Metric" in ff:
                        ff["Metric"] = new_metric_config
            for k, v in new_config.get("Trainer", dict()).items():
                if k in ["non_target_features", "inference_batch_size", "dtype", "device"]:
                    config.Trainer[k] = v
        else:
            config = new_config

        logger.info('Config: {}'.format(config))
        
        ### load from ckp will initialize the datahub and the modelhub
        self.load_from_ckp(**config)

    def load_from_ckp(self, **params) -> None:
        ## load test data
        self.datahub = DataHub(
            dump_dir=self.output_dir, 
            **params['Datahub']
        )
        self.trainer = Trainer(
            out_dir=self.model_dir, 
            metric_config=params['Metric'], 
            **params['Trainer']
        )
        self.modelhub = ModelHub(self.datahub, self.trainer, **params['Modelhub'])
        self.metrics = self.trainer.metrics

    def predict(self) -> None:
        if self.simple_predict:
            return self._simple_predict(non_target_features=self.trainer.non_target_features, save=True)
        
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        metrics = []
        for ff_name, ff in FFs.items():
            result = dict()
            if self.new_data:
                predict_result = ff.evaluate()
            else:
                partition = ff._init_partition()
                predict_result = ff._evaluate(partition["test"])
            y_pred = predict_result["y_pred"]
            y_truth = predict_result["y_truth"]
            metric_score = predict_result["metric_score"]
            result = pd.DataFrame({"data_key": y_pred["data_key"]})
            all_target_keys = set()
            for data_key, target in self.datahub.targets.items():
                all_target_keys.update(target.data.keys())
            for k in all_target_keys:
                if hasattr(ff, "size") and ff.size > 1:
                    for i, y_pred_single in enumerate(y_pred):
                        result[f"predict{i}_{k}"] = list(y_pred_single[k])
                else:
                    result[f"predict_{k}"] = list(y_pred[k])
                result[k] = list(y_truth[k])
            for k in self.trainer.non_target_features:
                if hasattr(ff, "size") and ff.size > 1:
                    for i, y_pred_single in enumerate(y_pred):
                        result[f"{k}{i}"] = list(y_pred_single[k])
                else:
                    result[k] = list(y_pred[k])
            os.makedirs(self.output_dir, exist_ok=True)
            for data_key, datahub in self.datahub.datahubs.items():
                result[result["data_key"] == data_key].to_pickle(os.path.join(datahub.preload_path, f"{ff_name}-prediction.pkl"))
            metrics.append(metric_score)
        metrics_df = pd.concat(metrics)
        metrics_df.to_csv(os.path.join(self.output_dir, 'metric.csv'))
        logger.info(f"final predict metrics score: \n{metrics_df.T}")
        logger.info("pipeline finish!")

    def _simple_predict(self, non_target_features: List[str], save: bool=True) -> Dict[str, Dict[Literal["y_pred", "y_truth", "metric_score"], Any]]:
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        predict_results = dict()
        for ff_name, ff in FFs.items():
            ff.trainer.non_target_features.extend(non_target_features)
            predict_results[ff_name] = ff.evaluate()
            if save:
                os.makedirs(self.output_dir, exist_ok=True)
                y_pred = predict_results[ff_name]["y_pred"]
                result = pd.DataFrame({k: list(v) for k, v in y_pred.items()})
                for data_key, datahub in self.datahub.datahubs.items():
                    result[result["data_key"] == data_key].to_pickle(os.path.join(datahub.preload_path, f"{ff_name}-prediction.pkl"))
        return predict_results

    def _simple_load_prediction(self):
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        predict_results = dict()
        for ff_name, ff in FFs.items():
            predict_result = dict()
            predict_result["y_pred"] = pd.read_pickle(os.path.join(self.datahub.preload_path, f"{ff_name}-prediction.pkl"))
            predict_results[ff_name] = predict_result
        return predict_results
