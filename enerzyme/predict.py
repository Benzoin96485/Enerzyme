import os
from copy import copy
from typing import Dict, Optional, List, Literal, Any
import pandas as pd
from .utils import YamlHandler, logger
from .utils.util import atomic_output
from .tasks.distributed import run_rank0_exclusive
from .train import validate_parallel_config
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
        if config_path is not None:
            new_config = YamlHandler(config_path).read_yaml()
            new_datahub_config = new_config.get("Datahub", dict())
            if new_datahub_config:
                if "datasets" in config.Datahub and "datasets" not in new_datahub_config:
                    config.Datahub.pop("datasets")
                for k, v in new_datahub_config.items():
                    if k not in ["transforms", "neighbor_list", "global_transforms"]:
                        config.Datahub[k] = v
                if "targets" not in new_datahub_config:
                    config.Datahub.pop("targets", None)
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
            self.new_data = False

        validate_parallel_config(config, training=False)
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
            if self.new_data:
                predict_result = ff.evaluate()
            else:
                # The original training run owns the saved test indices, even
                # when prediction writes to a different output directory.
                splitter = copy(ff.splitter)
                splitter.preload, splitter.save = True, False
                paths = {key: os.path.join(self.model_dir, os.path.basename(path))
                         for key, path in self.datahub.preload_path.items()}
                split = splitter.get_split(self.datahub.features, paths)
                partition = ff._init_partition(split)
                predict_result = ff._evaluate(partition["test"])
            def save_prediction():
                predictions = predict_result["y_pred"]
                committee = isinstance(predictions, list)
                members = predictions if committee else [predictions]
                result = pd.DataFrame({"data_key": members[0]["data_key"]})
                targets = {key for fields in self.datahub.targets.values() for key in fields.keys()}
                for key in targets:
                    for i, member in enumerate(members):
                        column = f"predict{i}_{key}" if committee else f"predict_{key}"
                        result[column] = list(member[key])
                    result[key] = list(predict_result["y_truth"][key])
                for key in ff.trainer.non_target_features:
                    for i, member in enumerate(members):
                        result[f"{key}{i}" if committee else key] = list(member[key])
                self._write_predictions(ff_name, result)
                metrics.append(predict_result["metric_score"])

            run_rank0_exclusive(save_prediction, sync_dir=self.output_dir, name=f"prediction_{ff_name}")

        def save_metrics():
            if metrics:
                scores = pd.concat(metrics)
                with atomic_output(os.path.join(self.output_dir, "metric.csv")) as path:
                    scores.to_csv(path)
                logger.info(f"Final prediction metrics:\n{scores.T}")

        run_rank0_exclusive(save_metrics, sync_dir=self.output_dir, name="prediction_metrics")

    def _write_predictions(self, ff_name, frame):
        for data_key, hub in self.datahub.datahubs.items():
            with atomic_output(os.path.join(hub.preload_path, f"{ff_name}-prediction.pkl")) as path:
                frame[frame["data_key"] == data_key].to_pickle(path)

    def _simple_predict(self, non_target_features: List[str], save: bool=True) -> Dict[str, Dict[Literal["y_pred", "y_truth", "metric_score"], Any]]:
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        predict_results = dict()
        for ff_name, ff in FFs.items():
            for key in non_target_features:
                if key not in ff.trainer.non_target_features:
                    ff.trainer.non_target_features.append(key)
            predict_results[ff_name] = ff.evaluate()
            if save:
                def save_prediction():
                    predictions = predict_results[ff_name]["y_pred"]
                    if isinstance(predictions, list):
                        # Keep member zero's ordinary columns for existing simple-prediction readers.
                        result = pd.DataFrame({k: list(v) for k, v in predictions[0].items()})
                        for i, member in enumerate(predictions[1:], 1):
                            for key, values in member.items():
                                if key != "data_key":
                                    result[f"{key}{i}"] = list(values)
                    else:
                        result = pd.DataFrame({k: list(v) for k, v in predictions.items()})
                    self._write_predictions(ff_name, result)

                run_rank0_exclusive(save_prediction, sync_dir=self.output_dir, name=f"simple_prediction_{ff_name}")
        return predict_results

    def _simple_load_prediction(self):
        FFs: Dict[str, BaseFFLauncher] = self.modelhub.models.get('FF', dict())
        predict_results = dict()
        for ff_name, ff in FFs.items():
            predict_result = dict()
            predict_result["y_pred"] = pd.concat([pd.read_pickle(os.path.join(datahub.preload_path, f"{ff_name}-prediction.pkl")) for data_key, datahub in self.datahub.datahubs.items()])
            predict_results[ff_name] = predict_result
        return predict_results
