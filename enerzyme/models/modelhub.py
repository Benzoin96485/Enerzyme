import os.path as osp
from functools import partial, cmp_to_key
from glob import glob
from collections import defaultdict
from typing import Literal, Optional
from ..utils import logger
from ..data.datahub import DataHub
from ..tasks.trainer import Trainer
from .ff import FF_single, FF_committee


def get_model_str(model_id, model_params):
    return f"{model_id}-{model_params.architecture}" + (f"-{model_params.suffix}" if model_params.suffix is not None else "")


def get_all_possible_paths(pretrain_path: str, prefix: str):
    return glob(osp.join(pretrain_path, f"{prefix}_best-v*.pth")) + glob(osp.join(pretrain_path, f"{prefix}_last-v*.pth")) + glob(osp.join(pretrain_path, f"{prefix}_best.pth")) + glob(osp.join(pretrain_path, f"{prefix}_last.pth"))


def get_all_possible_paths_with_rank(pretrain_path: str, model_rank: Optional[int]=None):
    if model_rank is None:
        return get_all_possible_paths(pretrain_path, "model") + get_all_possible_paths(pretrain_path, "model0")
    else:
        return get_all_possible_paths(pretrain_path, f"model{model_rank}")


def path_resolve(path, target_preference: Literal["best", "last"]="best", target_model_rank: Optional[int]=None):
    basename = osp.basename(path).split(".")[0]
    model_prefix, model_suffix = basename.split("_")
    if len(model_prefix) == 5:
        if target_model_rank is None:
            model_rank = 1
        else:
            model_rank = -1
    else:
        model_rank = int(model_prefix[5:])
    version_split = model_suffix.split("-")
    if len(version_split) == 1:
        version = 0
    else:
        version = int(version_split[1][1:])
    return {
        "model_rank": model_rank,
        "preference": int(target_preference == version_split[0]),
        "version": version,
    }


def compare_path(path1, path2, preference: Literal["best", "last"]="best", model_rank: Optional[int]=None) -> int:
    '''
    Compare two paths in the order of

    - preference: if the preference is "best", then the path of the best checkpoint is prioritized; otherwise, the path of the last checkpoint is prioritized.
    - model_rank: explicit model rank is prioritized.
    - version: latest version is prioritized.

    Params:
    ----------
    path1: str
        The first path to compare.
    path2: str
        The second path to compare.
    preference: Literal["best", "last"]
        The preference between the best checkpoint and the last checkpoint.
    model_rank: Optional[int]
        The model rank. Only used for deep ensemble to make sure that the model with the correct rank is selected.

    Returns:
    ----------
    int
        The difference between the two paths. When the difference is positive, path1 prioritized over path2.
    '''
    path1_info = path_resolve(path1, preference, model_rank)
    path2_info = path_resolve(path2, preference, model_rank)
    if path1_info["preference"] != path2_info["preference"]:
        return path1_info["preference"] - path2_info["preference"]
    elif path1_info["model_rank"] != path2_info["model_rank"]:
        return path1_info["model_rank"] - path2_info["model_rank"]
    else:
        return path1_info["version"] - path2_info["version"]
    

def get_pretrain_path(pretrain_path: Optional[str]=None, preference: Literal["best", "last"]="best", model_rank: Optional[int]=None):
    if pretrain_path is not None:
        if osp.isfile(pretrain_path):
            return pretrain_path
        elif osp.isdir(pretrain_path):
            if model_rank == None:
                model_rank = ''
            all_possible_paths = sorted(
                get_all_possible_paths_with_rank(pretrain_path, model_rank), 
                key=cmp_to_key(partial(
                    compare_path, 
                    preference=preference, 
                    model_rank=model_rank
                )), reverse=True
            )
            if len(all_possible_paths) == 0:
                raise FileNotFoundError(f"Pretrained model{' ' if model_rank is None else 'ranked ' + str(model_rank)} not found in {pretrain_path}")
            return all_possible_paths[0]
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
            if model_params['active']:
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

                if trainer.committee_size > 1:
                    logger.info(f"Initiate {model_str} Force Field Committee of {trainer.committee_size}")
                    self.models['FF'][model_str] = FF_committee(trainer.committee_size, self.datahub, trainer, model_str, **model_params)
                else:
                    logger.info(f"Initiate {model_str} Force Field")
                    self.models['FF'][model_str] = FF_single(self.datahub, trainer, model_str, **model_params)
