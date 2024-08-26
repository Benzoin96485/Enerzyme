from collections import defaultdict
from ..utils import logger
from ..data import DataHub
from ..tasks import Trainer
from .ff import FF


def get_model_str(model_id, model_params):
    return f"{model_id}-{model_params.architecture}" + (f"-{model_params.suffix}" if model_params.suffix else "")


class ModelHub:
    def __init__(self, datahub: DataHub, trainer: Trainer, **params) -> None:
        self.datahub = datahub
        self.models = defaultdict(dict)
        self.default_trainer = trainer
        self.default_metrics = trainer.metrics
        self.out_dir = self.default_trainer.out_dir
        self._init_models(**params)
    
    def _init_models(self, **params) -> None:
        internal_FFs = params['internal_FFs']

        for model_id, model_params in internal_FFs.items():
            # WARNING: please carefully update the model_str function due to
            # need to call each model one by one.
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
                self.models['FF'][model_str] = self._init_ff(trainer, model_str, **model_params)
    
    def _init_ff(self, trainer, model_str, **model_params):
        logger.info("Initiate {} Force Field".format(model_str))
        model = FF(self.datahub, trainer, model_str, **model_params)
        return model