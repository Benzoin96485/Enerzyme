import os
from inspect import signature
from typing import Dict, Tuple, List, Any, Callable, Literal, Optional, Iterable, Union
from functools import partial
from abc import ABC, abstractmethod
import joblib
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset, Subset

from ..data import DataHub, FieldDataset
from ..tasks import Trainer
from .loss import LOSS_REGISTER
from ..utils import logger
from . import layers as Layers


SEP = "-"


def get_ff_core(architecture: str) -> Tuple[Layers.BaseFFCore, Dict[str, Any], List[Dict[str, Any]]]:
    global LOSS_REGISTER
    if architecture.lower() == "physnet":
        from .physnet import PhysNetCore as Core
        from .physnet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        from .physnet import LOSS_REGISTER as special_loss
    elif architecture.lower() == "spookynet":
        from .spookynet import SpookyNetCore as Core
        from .spookynet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        special_loss = {}
    elif architecture.lower() == "leftnet":
        from .leftnet import LEFTNet as Core
        from .leftnet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        special_loss = {}
    elif architecture.lower() == "mace":
        from .mace import MACEWrapper as Core
        from .mace import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        special_loss = {}
    elif architecture.lower() == "nequip":
        from .nequip import NequIPWrapper as Core
        from .nequip import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        special_loss = {}
    elif architecture.lower() == "xpainn":
        from .xpainn import XPaiNNWrapper as Core
        from .xpainn import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        special_loss = {}
    LOSS_REGISTER.update(special_loss)
    return Core, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS


def build_layer(layer: Callable, layer_params: Dict[str, Any], build_params: Dict[str, Any], built_layers: Optional[Dict[str, Module]]=None) -> Module:
    final_params = dict()
    for name, attr in signature(layer).parameters.items():
        if name in layer_params:
            final_params[name] = layer_params[name]
        elif name in build_params:
            final_params[name] = build_params[name]
        elif name == "built_layers" and built_layers is not None:
            final_params[name] = built_layers
        elif attr.default is attr.empty:
            raise TypeError(f"{name} value should be provided")
    return layer(**final_params)


def build_model(
    architecture: str, 
    layer_params: Optional[List[Dict[Literal["name", "params"], Any]]]=None, 
    build_params: Optional[Dict]=None,
    verbose: int=1
) -> Module:
    Core, default_build_params, default_layer_params = get_ff_core(architecture)
    if layer_params is None:
        layer_params = default_layer_params
    if build_params is None:
        build_params = default_build_params
    built_layers = []
    core = None
    for layer_param in layer_params:
        layer_name = layer_param["name"]
        params = layer_param.get("params", dict())
        if layer_name == "Core":
            Layer = Core
        elif hasattr(Layers, layer_name + "Layer"):
            Layer = getattr(Layers, layer_name + "Layer")
        elif hasattr(Layers, layer_name):
            Layer = getattr(Layers, layer_name)
        else:
            raise NotImplementedError(f"Layer {layer_name} not implemented")
        
        layer = build_layer(Layer, params, build_params, built_layers)
        if layer_name == "Core":
            core = layer
        if verbose:
            logger.info(f"built {layer_name}")
        built_layers.append(layer)
    
    if hasattr(core, "build"):
        core.build(built_layers)
    return core


class FFDataset(Dataset):
    def __init__(self, features: FieldDataset, targets: FieldDataset, indices: Optional[Iterable[int]]=None, data_in_memory: bool=True) -> None:
        if indices is None:
            indices = np.arange(0, len(features["Ra"]))
        self.data_in_memory = data_in_memory
        self.full_features = features
        self.full_targets = targets
        if data_in_memory:
            self.features = features.load_subset(indices)
            self.targets = targets.load_subset(indices)
            self.indices = np.arange(0, len(indices))
        else:
            self.features = features
            self.targets = targets
            self.indices = indices
        self.raw_indices = np.array(indices)

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:
        return self.features.loc(self.indices[idx]), self.targets.loc(self.indices[idx])
    
    def expand_with_indices(self, new_indices: List[int]) -> None:
        self.raw_indices = np.concatenate((self.raw_indices, new_indices))
        if self.data_in_memory:
            self.features = self.full_features.load_subset(self.raw_indices)
            self.targets = self.full_targets.load_subset(self.raw_indices)
            self.indices = np.arange(0, len(self.raw_indices))
        else:
            self.indices = self.raw_indices


class MetaStateDict(dict):
    def __init__(self, dump_path) -> None:
        super().__init__()
        self.dump_path = dump_path

    def load(self) -> None:
        if os.path.exists(self.dump_path):
            self.update(joblib.load(self.dump_path))

    def dump(self) -> None:
        joblib.dump({k: v for k, v in self.items()}, self.dump_path)

    def update(self, d: Dict) -> None:
        super().update(d)
        self.dump()


class BaseFFLauncher(ABC):
    def __init__(self, 
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params: Optional[Dict]=None, layers: Optional[List[Dict]]=None,
        pretrain_path: Optional[str]=None
    ) -> None:
        self.datahub = datahub
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.architecture = architecture
        self.build_params = build_params
        self.layer_params = layers
        self.pretrain_path = pretrain_path
        self.metrics = self.trainer.metrics
        self.out_dir = self.trainer.out_dir
        self.dump_dir = os.path.join(self.out_dir, self.model_str)
        self.loss_params = loss
        self.loss_terms = {}
        self.trainer._set_seed(self.trainer.seed)
        self.uq_mode = None

    def _init_model(self, verbose=1) -> Module:
        model = build_model(self.architecture, self.layer_params, self.build_params, verbose)
        self.loss_terms = {k: LOSS_REGISTER[k](**v) for k, v in self.loss_params.items()}
        if verbose:
            print(model.__str__())
        return model

    def _init_partition(self, split: Optional[Dict[str, Iterable[int]]]=None) -> Dict[str, FFDataset]:
        X = self.datahub.features
        y = self.datahub.targets
        partitions = dict()
        if split is None:
            split = self.splitter.split(X, preload_path=self.datahub.preload_path)
        for k, v in split.items():
            if len(v) > 0:
                if k == "training":
                    if "withheld" in split:
                        partitions[k] = FFDataset(X, y, v, False)
                    else:
                        partitions[k] = FFDataset(X, y, v, self.trainer.data_in_memory)
                elif k == "withheld":
                    partitions[k] = FFDataset(X, y, v, False)
                else:
                    partitions[k] = FFDataset(X, y, v, True)
        return partitions
    
    def _init_default_partition(self) -> Tuple[FFDataset, Optional[FFDataset], Optional[FFDataset]]:
        partitions = self._init_partition()
        return partitions["training"], partitions.get("validation", None), partitions.get("test", None)
    
    @abstractmethod
    def _init_pretrain_path(self) -> None:
        ...
    
    @abstractmethod
    def _train(
        self, 
        train_dataset: FFDataset, 
        valid_dataset: Optional[FFDataset]=None, 
        test_dataset: Optional[FFDataset]=None,
        max_epoch_per_iter: int=-1,
        **kwargs
    ) -> None:
        ...

    @abstractmethod
    def _evaluate(self, dataset: FFDataset) -> Tuple[Union[Dict, List[Dict]], pd.DataFrame]:
        ...

    def train(self) -> None:
        self._train(*self._init_default_partition())

    def evaluate(self) -> Dict[Literal["y_pred", "y_truth", "metric_score"], Any]:
        X = self.datahub.features
        y = self.datahub.targets
        test_dataset = FFDataset(X, y, data_in_memory=True)
        return self._evaluate(test_dataset)

    def dump(self, data: Any, dump_dir: str, name: str) -> None:
        path = os.path.join(dump_dir, name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        joblib.dump(data, path)
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def active_learn(self) -> None:
        assert self.uq_mode is not None
        from ..tasks.active_learning import PICKING_REGISTER

        active_learning_params = self.trainer.active_learning_params
        picking_method = active_learning_params.get("picking_method", "max_Fa_norm_std")
        max_epoch_per_iter = active_learning_params.get("max_epoch_per_iter", -1)
        picking_params_ = active_learning_params.get("picking_params", {})
        relative_bound_on_validation = picking_params_.get("relative_bound_on_validation", False)
        data_source = active_learning_params.get("data_source", "withheld")
        sample_size = active_learning_params["sample_size"]
        checkpoint_name = active_learning_params.get("checkpoint_name", "al_ckp.data")
        refresh_best_score = active_learning_params.get("refresh_best_score", False)

        resume = active_learning_params.get("resume", False)
        al_state_dict = MetaStateDict(os.path.join(self.dump_dir, checkpoint_name))
        if resume and self.trainer.resume:
            al_state_dict.load()
            partitions = self._init_partition(al_state_dict.get("new_indices", None))
        else:
            self.dump({}, self.dump_dir, checkpoint_name)
            partitions = self._init_partition()
        
        training_set = partitions["training"]
        validation_set = partitions.get("validation", None)
        test_set = partitions.get("test", None)
        withheld_set = partitions["withheld"]

        len_training = len(training_set)
        len_validation = len(validation_set) if validation_set is not None else 0
        ratio_training = len_training / (len_training + len_validation)
        max_iter = active_learning_params.get("max_iter", len(withheld_set))

        picking_func_ = PICKING_REGISTER[picking_method]
        picking_params = {}
        for name, attr in signature(picking_func_).parameters.items():
            if name in picking_params_:
                picking_params[name] = picking_params_[name]
            elif name != "y_preds" and attr.default is attr.empty:
                raise ValueError(f"Parameter {name} is missing in picking_params!")
        
        picking_func = partial(picking_func_, mode=self.uq_mode)
        
        if data_source == "withheld":
            withheld_size = len(withheld_set)
            withheld_mask = np.full(withheld_size, True)

            iter_count = al_state_dict.get("iter_count", 0)
            if iter_count > 0:
                logger.info(f"Active learning iteration {iter_count + 1} resumed!")
            while iter_count < max_iter:   
                unmasked_relative_indices = withheld_mask.nonzero()[0]
                unmasked_size = len(unmasked_relative_indices)
                if unmasked_size == 0:
                    logger.info(f"Withheld set is exhausted and active learning stops at iteration {iter_count}!")
                    break

                if iter_count > 0:
                    self.trainer.resume = 2
                    self._init_pretrain_path(self.dump_dir)

                if al_state_dict.get("stage", 0) < 1: # training in this iteration unfinished
                    self._train(training_set, validation_set, test_set, max_epoch_per_iter, meta_state_dict=al_state_dict, refresh_patience=True, refresh_best_score=refresh_best_score)
                    al_state_dict.update({"stage": 1, "epoch_in_iter": 0})
                else:
                    logger.info(f"Model training in active learning iteration {iter_count + 1} has finished, start evaluating!")

                if relative_bound_on_validation:
                    if "relative_error_lower_bound" not in picking_params_ or "relative_error_upper_bound" not in picking_params_:
                        raise ValueError("Relative error lower bound and upper bound are required for relative bound on validation set!")
                    if "estimated_error_mean" in al_state_dict:
                        estimated_error_mean = al_state_dict["estimated_error_mean"]
                    else:
                        if validation_set is None:
                            raise ValueError("Validation set is required for relative bound on validation set!")
                        validation_result = self._evaluate(validation_set)
                        y_pred = validation_result["y_pred"]
                        logger.info(f"Estimating error mean on validation set...")
                        estimated_error_mean = picking_func(y_pred, stat_only=True, **picking_params)["estimated_error_mean"]
                        al_state_dict.update({"estimated_error_mean": estimated_error_mean})
                    if al_state_dict.get("relative_error_lower_bound", None) != picking_params_["relative_error_lower_bound"] or al_state_dict.get("relative_error_upper_bound", None) != picking_params_["relative_error_upper_bound"]:
                        al_state_dict.update({
                            "relative_error_lower_bound": picking_params_["relative_error_lower_bound"],
                            "relative_error_upper_bound": picking_params_["relative_error_upper_bound"]
                        })
                    picking_params["error_lower_bound"] = picking_params_["relative_error_lower_bound"] * estimated_error_mean
                    picking_params["error_upper_bound"] = picking_params_["relative_error_upper_bound"] * estimated_error_mean
                
                n_part = (unmasked_size + sample_size - 1) // sample_size
                masked_relative_indices = []
                for j in range(n_part):
                    part_relative_indices = unmasked_relative_indices[j * sample_size: (j + 1) * sample_size]
                    withheld_part = Subset(withheld_set, part_relative_indices)
                    predict_result = self._evaluate(withheld_part)
                    y_pred = predict_result["y_pred"]
                    masked_relative_indices += [part_relative_indices[idx] for idx in picking_func(y_pred, **picking_params)["picked_indices"]]
                    if len(masked_relative_indices) >= sample_size:
                        break
                
                if len(masked_relative_indices) == 0:
                    logger.info(f"No uncertain samples are found and active learning stops at iteration {iter_count + 1}!")
                    break

                masked_relative_indices = masked_relative_indices[:sample_size]
                expand_absolute_indices = withheld_set.raw_indices[masked_relative_indices]
                len_expanded = len(expand_absolute_indices)
                len_expanded_training = int(len_expanded * ratio_training + 0.5)
                training_set.expand_with_indices(expand_absolute_indices[:len_expanded_training])
                if validation_set is not None:
                    validation_set.expand_with_indices(expand_absolute_indices[len_expanded_training:])
                withheld_mask[masked_relative_indices] = False
                new_indices = {
                    "training": training_set.raw_indices,
                    "withheld": withheld_set.raw_indices
                }
                if validation_set is not None:
                    new_indices["validation"] = validation_set.raw_indices
                if test_set is not None:
                    new_indices["test"] = test_set.raw_indices
                iter_count += 1
                al_state_dict.update({"new_indices": new_indices, "iter_count": iter_count, "stage": 0, "model_rank": 0})
                logger.info(f"Active learning iteration {iter_count} finished!")
        else:
            raise NotImplementedError

class FF_single(BaseFFLauncher):
    def __init__(self, 
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params=None, layers=None,
        pretrain_path=None, **params
    ) -> None:
        super().__init__(datahub, trainer, model_str, loss, architecture, build_params, layers, pretrain_path)
        self.base_pretrain_path = pretrain_path
        self._init_pretrain_path()
        self.model = self._init_model()
        self.uq_mode = "single"
    
    def _init_pretrain_path(self, base_pretrain_path: Optional[str]=None) -> None:
        from .modelhub import get_pretrain_path
        if self.trainer.resume:
            if base_pretrain_path is None:
                try:
                    self.pretrain_path = get_pretrain_path(self.dump_dir, "last", None)
                except FileNotFoundError:
                    self.pretrain_path = None
            else:
                self.pretrain_path = get_pretrain_path(base_pretrain_path, "last", None)
        else:
            self.pretrain_path = get_pretrain_path(self.base_pretrain_path if base_pretrain_path is None else base_pretrain_path, "best", None)

    def _train(
        self, 
        train_dataset: FFDataset, 
        valid_dataset: Optional[FFDataset]=None, 
        test_dataset: Optional[FFDataset]=None,
        max_epoch_per_iter=-1,
        meta_state_dict: MetaStateDict=dict(),
        refresh_patience: bool=False,
        refresh_best_score: bool=False,
        **kwargs
    ) -> None:
        logger.info("start training FF: {}".format(self.model_str))
        predict_result = self.trainer.fit_predict(
            model=self.model, 
            pretrain_path=self.pretrain_path,
            train_dataset=train_dataset, 
            valid_dataset=valid_dataset, 
            test_dataset=test_dataset,
            loss_terms=self.loss_terms, 
            transform=self.datahub.transform,
            dump_dir=self.dump_dir,
            max_epoch_per_iter=max_epoch_per_iter,
            refresh_patience=refresh_patience,
            refresh_best_score=refresh_best_score
        )
        y_pred = predict_result["y_pred"]
        metric_score = predict_result["metric_score"]
        logger.info("{} FF done!".format(self.model_str))
        logger.info("{} Model saved!".format(self.model_str))
        if test_dataset is not None:
            self.dump(y_pred, self.dump_dir, 'test.data')
            self.dump(metric_score, self.dump_dir, 'metric.result')
            logger.info("{} FF metrics score on test set: \n{}".format(self.model_str, metric_score))
            logger.info("Metric result saved!")

    def _evaluate(self, dataset: FFDataset) -> Dict[Literal["y_pred", "y_truth", "metric_score"], Any]:
        logger.info("start evaluate FF:{}".format(self.model_str))
        predict_result = self.trainer.predict(
            model=self.model, 
            dataset=dataset, 
            loss_terms=self.loss_terms, 
            dump_dir=self.dump_dir, 
            transform=self.datahub.transform, 
            epoch=1, 
            load_model=True
        )
        y_pred = predict_result["y_pred"]
        y_truth = predict_result["y_truth"]
        metric_score = predict_result["metric_score"]
        logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
        logger.info("{} FF done!".format(self.model_str))
        print(metric_score)
        return {"y_pred": y_pred, "y_truth": y_truth, "metric_score": pd.DataFrame(metric_score, index=[self.model_str])}


class FF_committee(BaseFFLauncher):
    def __init__(self, 
        committee_size: int,
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params: Optional[Dict]=None, layers: Optional[List[Dict]]=None,
        pretrain_path: Optional[str]=None, **params
    ) -> None:
        super().__init__(
            datahub, trainer, model_str, loss, architecture, build_params, layers, 
            pretrain_path=None
        )
        self.size = committee_size
        self.base_pretrain_path = pretrain_path
        self.verbose = 1
        self._init_pretrain_path()
        self.uq_mode = "committee"

    def _init_pretrain_path(self, base_pretrain_path: Optional[str]=None) -> None:
        from .modelhub import get_pretrain_path
        if self.trainer.resume:
            if base_pretrain_path is None:
                try:
                    self.pretrain_path = [get_pretrain_path(self.dump_dir, "last", i) for i in range(self.size)]
                except FileNotFoundError:
                    self.pretrain_path = None
            else:
                self.pretrain_path = [
                    get_pretrain_path(base_pretrain_path, "last", i) 
                    for i in range(self.size)
                ]
        else:
            self.pretrain_path = [
                get_pretrain_path(self.base_pretrain_path if base_pretrain_path is None else base_pretrain_path, "best", i) 
                for i in range(self.size)
            ]

    def _train(self, 
        train_dataset: FFDataset, 
        valid_dataset: Optional[FFDataset]=None, 
        test_dataset: Optional[FFDataset]=None, 
        max_epoch_per_iter: int=-1, 
        meta_state_dict: MetaStateDict=dict(),
        refresh_patience: bool=False,
        refresh_best_score: bool=False,
        **kwargs
    ) -> None:
        model_rank = meta_state_dict.get("model_rank", 0)
        if model_rank > 0 and model_rank < self.size:
            logger.info(f"Training from model rank {model_rank} resumed!")
        elif model_rank >= self.size:
            logger.info(f"Training skipped!")
        for i in range(model_rank, self.size):
            logger.info(f"start training FF: {self.model_str} ({i})")
            self.model = self._init_model(self.verbose)
            self.verbose = 0
            predict_result = self.trainer.fit_predict(
                model=self.model, 
                pretrain_path=self.pretrain_path[i],
                train_dataset=train_dataset, 
                valid_dataset=valid_dataset, 
                test_dataset=test_dataset,
                loss_terms=self.loss_terms, 
                transform=self.datahub.transform,
                dump_dir=self.dump_dir,
                model_rank=i,
                max_epoch_per_iter=max_epoch_per_iter,
                meta_state_dict=meta_state_dict,
                refresh_patience=refresh_patience,
                refresh_best_score=refresh_best_score
            )
            y_pred = predict_result["y_pred"]
            metric_score = predict_result["metric_score"]
            logger.info(f"{self.model_str} FF ({i + 1} / {self.size}) done!")
            logger.info(f"{self.model_str} Model ({i}) saved!")
            delattr(self, "model")
            if test_dataset is not None:
                self.dump(y_pred, self.dump_dir, f'test{i}.data')
                logger.info(f"{self.model_str} FF ({i}) metrics score on test set: \n{metric_score}")
                self.dump(metric_score, self.dump_dir, f'metric{i}.result')
                logger.info(f"Metric result ({i}) saved!")

    def _evaluate(self, dataset: FFDataset) -> Dict[Literal["y_pred", "y_truth", "metric_score"], Any]:
        y_preds = []
        y_truth = None
        metric_scores = []
        for i in range(self.size):
            self.model = self._init_model(self.verbose)
            logger.info(f"start evaluate FF: {self.model_str} ({i})")
            predict_result = self.trainer.predict(
                model=self.model, 
                dataset=dataset, 
                loss_terms=self.loss_terms, 
                dump_dir=self.dump_dir, 
                transform=self.datahub.transform, 
                epoch=1, 
                load_model=True,
                model_rank=i
            )
            y_pred = predict_result["y_pred"]
            if y_truth is None:
                y_truth = predict_result["y_truth"]
            metric_score = predict_result["metric_score"]
            delattr(self, "model")
            logger.info(f"{self.model_str} FF ({i}) metrics score: \n{metric_score}")
            logger.info(f"{self.model_str} FF ({i}) done!")
            y_preds.append(y_pred)
            metric_scores.append(metric_score)
        return {"y_pred": y_preds, "y_truth": y_truth, "metric_score": pd.DataFrame(metric_scores, index=[self.model_str + str(i) for i in range(self.size)])}
