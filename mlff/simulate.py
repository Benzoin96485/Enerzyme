import os
import torch
from .models import SEP, FF_REGISTER
from .tasks import Simulation
from .utils import YamlHandler, logger


class FFSimulate:
    def __init__(self, model_dir=None, config_path=None, save_dir=None):
        if not model_dir:
            raise ValueError("model_dir is None")
        if not config_path:
            raise ValueError("config_path is None")
        # if not save_dir:
        #     raise ValueError("save_dir is None")
        self.model_dir = model_dir
        self.config = YamlHandler(config_path).read_yaml()
        self._init_model(self.config.Simulation.model_str)
        self.simulation = Simulation(self.config, self.model)
    
    def _init_model(self, model_str, fold=0):
        model_config_path = os.path.join(self.model_dir, 'config.yaml')
        self.model_config = YamlHandler(model_config_path).read_yaml()
        logger.info('Model Config: {}'.format(self.model_config))
        self.model_id, self.model_name, _, _ = model_str.split(SEP)[:4]
        single_param = self.model_config.Modelhub.FF[self.model_id]["params"]
        self.model = FF_REGISTER[self.model_name](**single_param)
        load_model_path = os.path.join(self.model_dir, model_str, f'model_{fold}.pth')
        model_dict = torch.load(load_model_path)["model_state_dict"]
        self.model.load_state_dict(model_dict)
        logger.info(f"load model success from {load_model_path}")

    def run(self):
        self.simulation.run()