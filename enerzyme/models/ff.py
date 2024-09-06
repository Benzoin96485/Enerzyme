import os
from inspect import signature
from typing import Dict, Tuple, List, Any, Callable, Literal, Optional
import joblib
import torch
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset
from ..data import DataHub
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


class FF:
    def __init__(self, 
        datahub: DataHub, 
        trainer: Trainer, 
        model_str: str, 
        loss: Dict, 
        architecture: str, 
        build_params, layers=None,
        pretrain_path=None, **params
    ) -> None:
        self.datahub = datahub
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_str = model_str
        self.architecture = architecture
        self.build_params = build_params
        self.layer_params = layers
        if pretrain_path is not None:
            if os.path.isdir(pretrain_path):
                self.pretrain_path = f"{pretrain_path}/model_best.pth"
            elif os.path.isfile(pretrain_path):
                self.pretrain_path = pretrain_path
            else:
                raise FileNotFoundError(f"Pretrained model not found at {self.pretrain_path}")
        else:
            self.pretrain_path = None
        self.metrics = self.trainer.metrics
        self.out_dir = self.trainer.out_dir
        self.dump_dir = os.path.join(self.out_dir, self.model_str)
        self.trainer._set_seed(self.trainer.seed)
        self.model = self._init_model(self.build_params)
        self.loss_terms = {k: LOSS_REGISTER[k](**v) for k, v in loss.items()}
        self.is_success = True
    
    def _init_model(self, build_params) -> Module:
        if self.architecture in FF_REGISTER:
            model = FF_REGISTER[self.architecture](**build_params)
        else:
            model = build_model(self.architecture, self.layer_params, self.build_params)
        print(model.__str__())
        if self.pretrain_path is not None:
            model_dict = torch.load(self.pretrain_path, map_location=self.trainer.device)["model_state_dict"]
            model.load_state_dict(model_dict, strict=False)
            logger.info(f"load model success from {self.pretrain_path}!")
        return model
    
    def train(self) -> None:
        logger.info("start training FF:{}".format(self.model_str))
        X = self.datahub.features
        y = self.datahub.targets
        split = self.splitter.split(X, preload_path=self.datahub.preload_path)
        tr_idx = split["training"]
        vl_idx = split["validation"]
        if "test" in split and len(split["test"]) > 0:
            te_idx = split["test"]
            test_dataset = FFDataset(X, y, te_idx, True)
            y_test = test_dataset.targets
            y_test["Za"] = test_dataset.features["Za"]
        else:
            test_dataset = None
        train_dataset = FFDataset(X, y, tr_idx, self.trainer.data_in_memory)
        valid_dataset = FFDataset(X, y, vl_idx, True)
        y_pred = self.trainer.fit_predict(
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
            self.datahub.transform.inverse_transform(y_test)
            self.dump(y_pred, self.dump_dir, 'test.data')
            metric_score = self.metrics.cal_metric(y_test, y_pred)
            self.dump(metric_score, self.dump_dir, 'metric.result')
            logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
            logger.info("Metric result saved!")

    def evaluate(self):
        logger.info("start evaluate FF:{}".format(self.model_str))
        X = self.datahub.features
        y = self.datahub.targets
        test_dataset = FFDataset(X, y, data_in_memory=True)
            # model_path = os.path.join(checkpoints_path, f'model_0.pth')
            # self.model.load_state_dict(torch.load(model_path, map_location=self.trainer.device)['model_state_dict'])
        y_pred, _, metric_score = self.trainer.predict(
            model=self.model, 
            dataset=test_dataset, 
            loss_terms=self.loss_terms, 
            dump_dir=self.dump_dir, 
            transform=self.datahub.transform, 
            epoch=1, 
            load_model=True
        )
        # self.dump(y_pred, self.dump_dir, 'test.data')
        # self.dump(metric_score, self.dump_dir, 'metric.result')
        logger.info("{} FF metrics score: \n{}".format(self.model_str, metric_score))
        logger.info("{} FF done!".format(self.model_str))
        logger.info("Metric result saved!")
        return y_pred, pd.DataFrame(metric_score, index=[self.model_str])

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)
        
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

class FFDataset(Dataset):
    def __init__(self, features, targets, indices=None, data_in_memory=True) -> None:
        self.data_in_memory = data_in_memory
        self.indices = indices if indices is not None else np.arange(0, len(features["Za"]))
        if data_in_memory:
            self.features = {k: np.array([v[idx] for idx in self.indices]) for k, v in features.items()}
            self.targets = {k: np.array([v[idx] for idx in self.indices]) for k, v in targets.items()}
            self._getitem = lambda idx: ({k: v[idx] for k, v in self.features.items()}, {k: v[idx] for k, v in self.targets.items()})
        else:
            self.features = features
            self.targets = targets
            self._getitem = lambda idx: ({k: v[self.indices[idx]] for k, v in self.features.items()}, {k: v[self.indices[idx]] for k, v in self.targets.items()})

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self._getitem(idx)