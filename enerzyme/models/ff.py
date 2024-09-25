import os
from inspect import signature
from typing import Dict, Tuple, List, Any, Callable, Literal, Optional, Iterable
import joblib
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset
from ..data import DataHub, FieldDataset
from ..tasks import Trainer
from .loss import LOSS_REGISTER
from ..utils import logger
from . import layers as Layers


SEP = "-"
FF_REGISTER = {}


def get_ff_core(architecture: str) -> Tuple[Layers.BaseFFCore, List]:
    global LOSS_REGISTER
    if architecture.lower() == "physnet":
        from .physnet import PhysNetCore as Core
        from .physnet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
        from .physnet import LOSS_REGISTER as special_loss
    elif architecture.lower() == "spookynet":
        from .spookynet import SpookyNetCore as Core
        from .spookynet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS
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
    build_params: Optional[Dict]=None
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
        logger.info(f"built {layer_name}")
        built_layers.append(layer)
    
    core.build(built_layers)
    return core


class FFDataset(Dataset):
    def __init__(self, features: FieldDataset, targets: FieldDataset, indices: Optional[Iterable[int]]=None, data_in_memory: bool=True) -> None:
        if indices is None:
            indices = np.arange(0, len(features["Ra"]))
        if data_in_memory:
            self.features = features.load_subset(indices)
            self.targets = targets.load_subset(indices)
            self.indices = np.arange(0, len(indices))
        else:
            self.features = features
            self.targets = targets
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:
        return self.features.loc(self.indices[idx]), self.targets.loc(self.indices[idx])


class BaseFFLauncher:
    def __init__(self, 
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params, layers=None,
        pretrain_path=None
    ) -> None:
        self.datahub = datahub
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.architecture = architecture
        self.build_params = build_params
        self.layer_params = layers
        if pretrain_path is not None:
            self.pretrain_path = pretrain_path
        else:
            self.pretrain_path = None
        self.metrics = self.trainer.metrics
        self.out_dir = self.trainer.out_dir
        self.dump_dir = os.path.join(self.out_dir, self.model_str)
        self.loss_params = loss
        self.loss_terms = {}
        self.trainer._set_seed(self.trainer.seed)

    def _init_model(self, build_params: Dict[str, Any], pretrain_path: Optional[str]=None) -> Module:
        if self.architecture in FF_REGISTER:
            model = FF_REGISTER[self.architecture](**build_params)
        else:
            model = build_model(self.architecture, self.layer_params, self.build_params)
        self.loss_terms = {k: LOSS_REGISTER[k](**v) for k, v in self.loss_params.items()}
        print(model.__str__())
        return model

    def _init_partition(self) -> Tuple[FFDataset, Optional[FFDataset], Optional[FFDataset]]:
        X = self.datahub.features
        y = self.datahub.targets
        split = self.splitter.split(X, preload_path=self.datahub.preload_path)
        tr_idx = split["training"]
        train_dataset = FFDataset(X, y, tr_idx, self.trainer.data_in_memory)
        if "validation" in split and len(split["validation"]) > 0:
            vl_idx = split["validation"]
            valid_dataset = FFDataset(X, y, vl_idx, True)
        else:
            valid_dataset = None
        if "test" in split and len(split["test"]) > 0:
            te_idx = split["test"]
            test_dataset = FFDataset(X, y, te_idx, True)
            y_test = test_dataset.targets
            y_test["Za"] = test_dataset.features["Za"]
        else:
            test_dataset = None
        return train_dataset, valid_dataset, test_dataset
    
    def dump(self, data: Any, dump_dir: str, name: str) -> None:
        path = os.path.join(dump_dir, name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        joblib.dump(data, path)
        
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

class FF_single(BaseFFLauncher):
    def __init__(self, 
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params, layers=None,
        pretrain_path=None, **params
    ) -> None:
        super().__init__(datahub, trainer, model_str, loss, architecture, build_params, layers, pretrain_path)
        if pretrain_path is not None and os.path.isdir(pretrain_path):
            self.pretrain_path = f"{pretrain_path}/model_best.pth"
            if self.pretrain_path is None:
                raise FileNotFoundError(f"Pretrained model not found at {pretrain_path}")
        self.model = self._init_model(self.build_params)
    
    def train(self) -> None:
        logger.info("start training FF:{}".format(self.model_str))
        train_dataset, valid_dataset, test_dataset = self._init_partition()
        y_pred, metric_score = self.trainer.fit_predict(
            model=self.model, 
            train_dataset=train_dataset, 
            valid_dataset=valid_dataset, 
            test_dataset=test_dataset,
            loss_terms=self.loss_terms, 
            transform=self.datahub.transform,
            dump_dir=self.dump_dir
        )
        logger.info("{} FF done!".format(self.model_str))
        logger.info("{} Model saved!".format(self.model_str))
        if test_dataset is not None:
            self.dump(y_pred, self.dump_dir, 'test.data')
            self.dump(metric_score, self.dump_dir, 'metric.result')
            logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
            logger.info("Metric result saved!")

    def evaluate(self):
        logger.info("start evaluate FF:{}".format(self.model_str))
        X = self.datahub.features
        y = self.datahub.targets
        test_dataset = FFDataset(X, y, data_in_memory=True)
        y_pred,_ , metric_score = self.trainer.predict(
            model=self.model, 
            dataset=test_dataset, 
            loss_terms=self.loss_terms, 
            dump_dir=self.dump_dir, 
            transform=self.datahub.transform, 
            epoch=1, 
            load_model=True
        )
        logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
        logger.info("{} FF done!".format(self.model_str))
        return y_pred, pd.DataFrame(metric_score, index=[self.model_str])


class FF_committee(BaseFFLauncher):
    def __init__(self, 
        committee_size: int,
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params, layers=None,
        pretrain_path=None, **params
    ) -> None:
        super().__init__(
            datahub, trainer, model_str, loss, architecture, build_params, layers, 
            pretrain_path=None
        )
        self.size = committee_size
        self.pretrain_path = []
        if pretrain_path is not None:
            if os.path.isdir(pretrain_path):
                for i in range(committee_size):
                    single_pretrain_path = f"{pretrain_path}/model{i}_best.pth"
                    if os.path.isfile(single_pretrain_path):
                        self.pretrain_path.append(single_pretrain_path)
                    else:
                        raise FileNotFoundError(f"Pretrained model {i} not found at {pretrain_path}")
            else:
                raise FileNotFoundError(f"Pretrained model not found at {pretrain_path}")
        else:
            self.pretrain_path = [None] * committee_size

    def train(self) -> None:
        train_dataset, valid_dataset, test_dataset = self._init_partition()
        for i in range(self.size):
            self.model = self._init_model(self.build_params)
            y_pred, metric_score = self.trainer.fit_predict(
                model=self.model, 
                pretrain_path=self.pretrain_path[i],
                train_dataset=train_dataset, 
                valid_dataset=valid_dataset, 
                test_dataset=test_dataset,
                loss_terms=self.loss_terms, 
                transform=self.datahub.transform,
                dump_dir=self.dump_dir,
                model_rank=i
            )
            logger.info(f"{self.model_str} FF ({i + 1} / {self.size}) done!")
            logger.info(f"{self.model_str} Model ({i}) saved!")
            delattr(self, "model")
            if test_dataset is not None:
                self.dump(y_pred, self.dump_dir, f'test{i}.data')
                logger.info(f"{self.model_str} FF ({i}) metrics score: \n{metric_score}")
                self.dump(metric_score, self.dump_dir, f'metric{i}.result')
                logger.info(f"Metric result ({i}) saved!")

    def evaluate(self) -> None:
        X = self.datahub.features
        y = self.datahub.targets
        test_dataset = FFDataset(X, y, data_in_memory=True)
        y_preds = []
        metric_scores = []
        for i in range(self.size):
            self.model = self._init_model(self.build_params)
            logger.info(f"start evaluate FF:{self.model_str} ({i})")
            y_pred, _, metric_score = self.trainer.predict(
                model=self.model, 
                dataset=test_dataset, 
                loss_terms=self.loss_terms, 
                dump_dir=self.dump_dir, 
                transform=self.datahub.transform, 
                epoch=1, 
                load_model=True,
                model_rank=i
            )
            delattr(self, "model")
            logger.info(f"{self.model_str} FF ({i}) metrics score: \n{metric_score}")
            logger.info(f"{self.model_str} FF ({i}) done!")
            y_preds.append(y_pred)
            metric_scores.append(metric_score)
        return y_preds, pd.DataFrame(metric_scores, index=[self.model_str + str(i) for i in range(self.size)])
