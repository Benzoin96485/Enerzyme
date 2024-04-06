import os
import argparse
from .utils import YamlHandler, logger
from .data import DataHub
from .models import ModelHub
from .tasks import Trainer


class FFTrain(object):
    def __init__(self, data_path=None, task='q', config_path=None, out_dir=None, **params):
        if not config_path:
            parent = os.path.dirname(__file__)
            if task == "q":
                config_path = os.path.join(parent, 'config/q_default.yaml')
            elif task == "e":
                config_path = os.path.join(parent, 'config/e_default.yaml')
            elif task == "qe":
                config_path = os.path.join(parent, 'config/qe_default.yaml')
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.data_path = data_path
        self.task = task
        self.config_path = config_path
        self.out_dir = out_dir
        self._init_datahub(**config.Datahub)
        self.config = self.update_config(config, task=self.task, **params)
        logger.info('Config: {}'.format(self.config))

        self._init_trainer(**config.Trainer)
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

    def _init_datahub(self, **params):
        self.datahub = DataHub(data_path = self.data_path, task = self.task, is_train = True, dump_dir = self.out_dir, **params)

    def _init_trainer(self, **params):
        self.trainer = Trainer(self.task, self.metrics_str, self.out_dir, **params)

    def _init_modelhub(self, **params):
        self.modelhub = ModelHub(self.datahub, self.trainer, self.task, **params)

    def update_config(self, config, **params):
        for key, value in params.items():
            if value is not None:
                config[key] = value
        return config
        
    def train_single(self, model_str):
        NNModels = self.modelhub.models.get('NNModel', None)
        if NNModels is not None and model_str in NNModels.keys():
            NNModels[model_str].run()
        

def get_parser(desc, default_task='train'):
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='training phase')
    parser.add_argument('--task', type=str, metavar='N', default='q', help='task type')
    parser.add_argument('--single_model_param', type=str, metavar='N', default='', 
        help='training single model param'
    )
    parser.add_argument('--config_path', type=str, metavar='N', default='', 
        help='training config'
    )
    parser.add_argument('--data_path', type=str, metavar='N', help='training data path')
    parser.add_argument('--output_dir', type=str, metavar='N', default='../results',
                    help='the output directory for saving artifact')          
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser("training")

    moltrain = FFTrain(
        data_path=args.data_path,
        task=args.task,
        out_dir=args.output_dir,
        config_path=args.config_path
    )
    if args.train:
        if args.single_model_param:
            model_param = args.single_model_param
            moltrain.train_single(model_str=model_param)
        else:
            moltrain.fit()
    else:
        pass

    logger.info("train complete")