import os, pathlib, importlib, sys
from typing import Optional
from .models import get_model_str, build_model, get_pretrain_path
from .tasks.simulator import Simulation
from .utils import YamlHandler, logger
from .data.transform import Transform


class FFSimulate:
    def __init__(self, model_dir: str, config_path: str, out_dir: str, patch: Optional[str] = None):
        self.model_dir = model_dir
        self.config = YamlHandler(config_path).read_yaml()
        self.out_dir = out_dir
        self.patch = patch
        self.patch_module = None
        self.simulations = []
        model_config_path = os.path.join(self.model_dir, 'config.yaml')
        model_config = YamlHandler(model_config_path).read_yaml()
        logger.info('Model Config: {}'.format(model_config))
        self.transform = Transform(model_config.Datahub.transforms, simulation_mode=True)
        if self.patch is not None:
            self._init_patch_module()
        for FF_key, FF_params in model_config.Modelhub.internal_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        for FF_key, FF_params in model_config.Modelhub.external_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        

    def _init_patch_module(self) -> None:
        p = pathlib.Path(self.patch).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix != ".py":
            raise ValueError(f"Plugin must be a .py file, got: {p}")

        # Hash the patch file to get a unique module name
        module_name = f"plugin_{p.stem}_{abs(hash(str(p)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {self.patch}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.patch_module = module
        logger.info(f"Initialized patch module: {self.patch}")
    
    def _init_model(self, FF_key, FF_params):
        model_str = get_model_str(FF_key, FF_params)
        model = build_model(FF_params.architecture, FF_params.layers, FF_params.build_params)
        model_path = get_pretrain_path(os.path.join(self.model_dir, model_str), "best")
        self.simulations.append(Simulation(self.config, model, model_path, self.out_dir, self.transform, self.patch_module))
        
    def run(self):
        for simulation in self.simulations:
            simulation.run()
