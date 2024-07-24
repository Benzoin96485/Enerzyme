from collections import defaultdict
from ..utils import hash_model_name, model_name_generation, logger
from ..tasks import Trainer
from .ff import FF


def get_model_str(model_id, model_params):
    return f"{model_id}-{model_params.architecture}" + (f"-{model_params.suffix}" if model_params.suffix else "")


class ModelHub:
    def __init__(self, datahub, trainer, **params):
        self.datahub = datahub
        self.models = defaultdict(dict)
        self.default_trainer = trainer
        self.out_dir = self.default_trainer.out_dir
        self._init_models(**params)
    
    def _init_models(self, **params):
        internal_FFs = params['internal_FFs']

        for model_id, model_params in internal_FFs.items():
            # WARNING: please carefully update the model_str function due to
            # need to call each model one by one.
            model_str = get_model_str(model_id, model_params)
            trainer_params = model_params.get('Trainer', None)
            if trainer_params is None:
                trainer = self.default_trainer
            else:
                logger.info("init {} custom train parameters".format(model_str))
                trainer = Trainer(self.task, self.out_dir, **trainer_params)
            if model_params['active']:
                self.models['FF'][model_str] = self._init_ff(trainer, model_str, **model_params)
    
    def _init_ff(self, trainer, model_str, **model_params):
        logger.info("Initiate {} Force Field".format(model_str))
        model = FF(self.datahub, trainer, model_str, **model_params)
        return model