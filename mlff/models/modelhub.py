from collections import defaultdict
from ..utils import hash_model_name, model_name_generation, logger
from ..tasks import Trainer
from .ff import FF


class ModelHub:
    def __init__(self, featurehub, trainer, task, **params):
        self.data = featurehub.data
        self.features = featurehub.features
        self.models = defaultdict(dict)
        self.task = task
        self.default_trainer = trainer
        self.out_dir = self.default_trainer.out_dir
        self._init_models(**params)
    
    def _init_models(self, **params):
        ffparams = params['FF']

        for model_id, single_params in ffparams.items():
            # WARNING: please carefully update the model_str function due to
            # need to call each model one by one.
            model_str = model_name_generation(model_id, single_params['model'],
                    single_params['feature'], self.task)
            feature_name = single_params['feature']
            model_params = single_params['params']
            trainer_params = single_params.get('trainer', None)
            if trainer_params is None:
                trainer = self.default_trainer
            else:
                logger.info("init {} custom train parameters".format(model_str))
                trainer = Trainer(self.task, self.metrics_str, self.out_dir, **trainer_params)
            loss_key = single_params.get('loss', None)
            if single_params['active']:
                self.models['FF'][model_str] = self._init_ff(self.data, self.features[feature_name], trainer, model_str, loss_key, **model_params)
    
    def _init_ff(self, data, feature, trainer, model_str, loss_key=None, **params):
        logger.info("init {} NN model".format(model_str))
        model = FF(data, feature, trainer, model_str, loss_key, **params)
        return model