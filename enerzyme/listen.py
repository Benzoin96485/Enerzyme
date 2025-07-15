import os
from flask import Flask, request, jsonify
import waitress
import logging
from .utils import YamlHandler, logger
from .data import Transform
from .models import get_model_str, build_model, get_pretrain_path
from .tasks.server import Server


app = Flask("enerzyme")


class FFListen:
    def __init__(self, config_path=str, model_dir=str, out_dir=str, bind=str):
        self.model_dir = model_dir
        self.config = YamlHandler(config_path).read_yaml()
        self.out_dir = out_dir
        self.calculators = dict()
        model_config_path = os.path.join(self.model_dir, 'config.yaml')
        model_config = YamlHandler(model_config_path).read_yaml()
        logger.info('Model Config: {}'.format(model_config))
        self.transform = Transform(model_config.Datahub.get("global_transforms", model_config.Datahub.get("transforms", None)), simulation_mode=True)
        for FF_key, FF_params in model_config.Modelhub.internal_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        for FF_key, FF_params in model_config.Modelhub.external_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        self.bind = bind
    
    def _init_model(self, FF_key, FF_params):
        model_str = get_model_str(FF_key, FF_params)
        model = build_model(FF_params.architecture, FF_params.layers, FF_params.build_params)
        model_path = get_pretrain_path(os.path.join(self.model_dir, model_str), "best")
        self.servers[FF_key] = Server(self.config, model=model, model_path=model_path, out_dir=self.out_dir, transform=self.transform)

    @app.route('/calculate', methods=['POST'])
    def run_calculate(self):
        info = request.get_json()
        outputs = dict()
        for k, v in self.servers.items():
            outputs[k] = v.calculate(info)
        return jsonify(outputs)
    
    def run(self):
        # Configure waitress logger to use our logger's handlers
        waitress_logger = logging.getLogger('waitress')
        waitress_logger.handlers = []  # Clear existing handlers
        waitress_logger.addHandler(logger.handlers[0])  # Add console handler
        if len(logger.handlers) > 1:
            waitress_logger.addHandler(logger.handlers[1])  # Add file handler
        waitress_logger.setLevel(logging.INFO)
        waitress_logger.propagate = False  # Prevent duplicate messages
        waitress.serve(app, listen=self.bind)

    def stop(self):
        pass
