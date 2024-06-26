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
        self._init_datahub(**config.Datahub)
        self.config = self.update_config(config, **params)
        logger.info('Config: {}'.format(self.config))
        self._init_trainer(**config.Trainer, **self.config.Base.Metric)
        self._init_modelhub(**config.Modelhub)

        if self.out_dir is not None:
            if not os.path.exists(self.out_dir):
                logger.info('Create output directory: {}'.format(self.out_dir))
                os.makedirs(self.out_dir)
            else:
                logger.info('Output directory already exists: {}'.format(self.out_dir))
                logger.info('Warning: Overwrite output directory: {}'.format(self.out_dir))
            out_path = os.path.join(self.out_dir, 'config.yaml')
            self.yamlhandler.write_yaml(data = self.config, out_file_path = out_path)

    def _init_baseconfig(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def _init_datahub(self, **params):
        self.datahub = DataHub(task=self.task, is_train=True, dump_dir=self.out_dir, **params)

    def _init_trainer(self, **params):
        self.trainer = Trainer(
            task=self.task,
            out_dir=self.out_dir,
            **params
        )

    def _init_modelhub(self, **params):
        self.modelhub = ModelHub(self.datahub, self.trainer, self.task, **params)

    def update_config(self, config, **params):
        for key, value in params.items():
            if value is not None:
                config[key] = value
        return config
        
    def train_all(self):
        FFs = self.modelhub.models.get('FF', None)
        for ff in FFs.values():
            ff.run()
        

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default='', 
        help='training config'
    )
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                    help='the output directory for saving artifact') 
    parser.add_argument('-m', '--model_name', type=str,
                        help='training single model name')      
    args = parser.parse_args()
    return args