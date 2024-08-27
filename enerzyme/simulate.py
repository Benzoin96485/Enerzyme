import os
import torch
from .models import FF_REGISTER, get_model_str, build_model
from .tasks import Simulation
from .utils import YamlHandler, logger
from .data import Transform


class FFSimulate:
    def __init__(self, model_dir=None, config_path=None, out_dir=None):
        if not model_dir:
            raise ValueError("model_dir is None")
        if not config_path:
            raise ValueError("config_path is None")
        # if not save_dir:
        #     raise ValueError("save_dir is None")
        self.model_dir = model_dir
        self.config = YamlHandler(config_path).read_yaml()
        self.out_dir = out_dir
        self.simulations = []
        model_config_path = os.path.join(self.model_dir, 'config.yaml')
        model_config = YamlHandler(model_config_path).read_yaml()
        logger.info('Model Config: {}'.format(model_config))
        self.transform = Transform(model_config.Datahub.transforms)
        for FF_key, FF_params in model_config.Modelhub.internal_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
    
    def _init_model(self, FF_key, FF_params):
        model_str = get_model_str(FF_key, FF_params)
        if FF_params.architecture in FF_REGISTER:
            model = FF_REGISTER[FF_params.architecture](**FF_params.build_params)
        else:
            model = build_model(FF_params.architecture, FF_params.layers, FF_params.build_params)
        model_path = os.path.join(self.model_dir, model_str, f'model_best.pth')
        model_dict = torch.load(model_path)["model_state_dict"]
        model.load_state_dict(model_dict)
        logger.info(f"load model success from {model_path}")
        self.simulations.append(Simulation(self.config, model, self.out_dir, self.transform))
        
    def run(self):
        for simulation in self.simulations:
            simulation.run()