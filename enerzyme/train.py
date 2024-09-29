import os
import argparse
from .utils import YamlHandler, logger
from .data import DataHub
from .models import ModelHub
from .tasks import Trainer


class FFTrain(object):
    def __init__(self, config_path=None, out_dir=None, **params):
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.config_path = config_path
        self.out_dir = out_dir
        self.task = config.Base.task
        self.datahub = DataHub(dump_dir=self.out_dir, **config.Datahub)
        logger.info('Config: {}'.format(config))
        self.trainer = Trainer(out_dir=self.out_dir, metric_config=config.Metric, **config.Trainer)
        self.modelhub = ModelHub(self.datahub, self.trainer, **config.Modelhub)

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                logger.info('Create output directory: {}'.format(self.out_dir))
                os.makedirs(self.out_dir)
            else:
                logger.info('Output directory already exists: {}'.format(self.out_dir))
                logger.warning('Overwrite output directory: {}'.format(self.out_dir))
            out_path = os.path.join(self.out_dir, 'config.yaml')
            self.yamlhandler.write_yaml(data = config, out_file_path = out_path)
        
    def train_all(self):
        FFs = self.modelhub.models.get('FF', dict())
        for ff in FFs.values():
            if ff.trainer.active_learning:
                ff.active_learn()
            else:
                ff.train()
