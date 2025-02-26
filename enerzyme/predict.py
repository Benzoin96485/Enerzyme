import os
import pickle
from typing import Dict, Optional, List, Literal, Any
import pandas as pd
from .utils import YamlHandler, logger
from .data import DataHub
from .tasks import Trainer
from .models import ModelHub
from .models import BaseFFLauncher

class FFPredict:
    def __init__(self, model_dir: str, output_dir: str, config_path: Optional[str] = None) -> None:
        self.model_dir = model_dir
        self.output_dir = model_dir if output_dir is None else model_dir
        self.new_data = True

        model_config_path = os.path.join(model_dir, 'config.yaml')
        config = YamlHandler(model_config_path).read_yaml()
        new_config = YamlHandler(config_path).read_yaml()
        if config_path is not None:
            new_datahub_config = new_config.get("Datahub", dict())
            if new_datahub_config:
                for k, v in new_datahub_config.items():
                    if k not in ["transforms", "neighbor_list"]:
                        config.Datahub[k] = v
            else:
                self.new_data = False
            new_metric_config = new_config.get("Metric", dict())
            if new_metric_config:
                config.Metric = new_metric_config
                for ff in config.Modelhub.internal_FFs.values():
                    if "Metric" in ff:
                        ff["Metric"] = new_metric_config
            for k, v in new_config.get("Trainer", dict()).items():
                if k in ["non_target_features", "inference_batch_size"]:
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
            for k in self.datahub.targets.keys():
                result[k] = y_truth[k]
                if hasattr(ff, "size") and ff.size > 1:
                    for i, y_pred_single in enumerate(y_pred):
                        result[f"predict{i}_{k}"] = y_pred_single[k]
                else:
                    result[f"predict_{k}"] = y_pred[k]
            for k in self.trainer.non_target_features:
                if hasattr(ff, "size") and ff.size > 1:
                    for i, y_pred_single in enumerate(y_pred):
                        result[f"{k}{i}"] = y_pred_single[k]
                else:
                    result[k] = y_pred[k]
            os.makedirs(self.output_dir, exist_ok=True)
            pd.DataFrame({k: [vi for vi in v] for k, v in result.items()}).to_pickle(os.path.join(self.datahub.preload_path, f"{ff_name}-prediction.pkl"))
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
                pd.DataFrame({k: [vi for vi in v] for k, v in predict_results[ff_name]["y_pred"].items()}).to_pickle(os.path.join(self.datahub.preload_path, f"{ff_name}-prediction.pkl"))
        return predict_results

    def _simple_load_prediction(self):
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        predict_results = dict()
        for ff_name, ff in FFs.items():
            predict_result = dict()
            predict_result["y_pred"] = pd.read_pickle(os.path.join(self.datahub.preload_path, f"{ff_name}-prediction.pkl"))
            predict_results[ff_name] = predict_result
        return predict_results
