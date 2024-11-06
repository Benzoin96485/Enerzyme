import os.path as osp
from collections import defaultdict
from typing import Literal, Optional
from ..utils import logger
from ..data import DataHub
from ..tasks import Trainer
from .ff import FF_single, FF_committee


def get_model_str(model_id, model_params):
    return f"{model_id}-{model_params.architecture}" + (f"-{model_params.suffix}" if model_params.suffix else "")


def get_pretrain_path(pretrain_path: Optional[str]=None, preference: Literal["best", "last"]="best", model_rank: Optional[int]=None):
    if pretrain_path is not None:
        if osp.isfile(pretrain_path):
            return pretrain_path
        elif osp.isdir(pretrain_path):
            if model_rank == None:
                model_rank = ''
            found_path = None
            best_path = osp.join(pretrain_path, f"model{model_rank}_best.pth")
            last_path = osp.join(pretrain_path, f"model{model_rank}_last.pth")
            if preference == "best":
                if osp.isfile(best_path):
                    found_path = best_path
                elif osp.isfile(last_path):
                    found_path = last_path
            elif preference == "last":
                if osp.isfile(last_path):
                    found_path = last_path
                elif osp.isfile(best_path):
                    found_path = best_path
            if found_path is None:
                if model_rank is None:
                    return get_pretrain_path(pretrain_path, preference, 0)
                raise FileNotFoundError(f"Pretrained model{' ' if model_rank is None else 'ranked ' + str(model_rank)} not found in {pretrain_path}")
            return found_path
        else:
            raise FileNotFoundError(f"Pretrained model not found at {pretrain_path}")
    else:
        return None


class ModelHub:
    def __init__(self, datahub: DataHub, trainer: Trainer, **params) -> None:
        self.datahub = datahub
        self.models = defaultdict(dict)
        self.default_trainer = trainer
        self.default_metrics = trainer.metrics
        self.out_dir = self.default_trainer.out_dir
        self._init_models(**params)
    
    def _init_models(self, **params) -> None:
        for model_id, model_params in {**params.get('internal_FFs', {}), **params.get('external_FFs', {})}.items():
            model_str = get_model_str(model_id, model_params)
            new_trainer_config = model_params.get('Trainer', None)
            new_metric_config = model_params.get('Metric', None)
            if new_trainer_config is None and new_metric_config is None:
                trainer = self.default_trainer
            else:
                trainer_config = self.default_trainer.config.copy()
                if new_metric_config is not None:
                    logger.info("init {} custom metrics".format(model_str))
                    metric_config = new_metric_config
                else:
                    metric_config = self.default_trainer.metric_config.copy()
                if new_trainer_config is not None:
                    logger.info("init {} custom train parameters".format(model_str))
                    for k, v in new_trainer_config.items():
                        trainer_config[k] = v
                trainer = Trainer(self.out_dir, metric_config=metric_config, **trainer_config)   
                
            if model_params['active']:
                if trainer.committee_size > 1:
                    logger.info(f"Initiate {model_str} Force Field Committee of {trainer.committee_size}")
                    self.models['FF'][model_str] = FF_committee(trainer.committee_size, self.datahub, trainer, model_str, **model_params)
                else:
                    logger.info(f"Initiate {model_str} Force Field")
                    self.models['FF'][model_str] = FF_single(self.datahub, trainer, model_str, **model_params)
