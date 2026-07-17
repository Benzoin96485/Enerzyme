import os
import time
from typing import Optional

from flask import Flask, request, jsonify
import waitress
import logging

from .utils import YamlHandler, logger
from .data.transform import Transform
from .models import get_model_str, build_model, get_pretrain_path
from .tasks.server import Server
from .tasks.calculator import _init_patch_module, get_patched_calculator


app = Flask("enerzyme")
_ff_listen_instance = None


class FFListen:
    def __init__(
        self,
        config_path=str,
        model_dir: Optional[str] = None,
        out_dir=str,
        bind=str,
        model_config_path: Optional[str] = None,
        calculator_patch: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.config = YamlHandler(config_path).read_yaml()
        self.out_dir = out_dir
        self.servers = dict()
        self.bind = bind
        self.calculator_patch = calculator_patch
        self.calculator_patch_module = None

        server_cfg = self.config.get("Server", dict())
        self.external_calculator_config = server_cfg.get("external_calculator", dict())
        self.uncertainty_calculator_config = server_cfg.get("uncertainty_calculator", None)
        self.internal_calculator_weight = server_cfg.get("internal_calculator_weight", 1.0)

        self.shell_mode = (
            self.internal_calculator_weight == 0
            and self.uncertainty_calculator_config is None
        )

        self.external_calculator = None
        if self.calculator_patch is not None:
            self.calculator_patch_module = _init_patch_module(self.calculator_patch)
            logger.info(f"Initialized calculator patch module: {self.calculator_patch}")
            self.external_calculator = get_patched_calculator(
                self.calculator_patch_module,
                self.external_calculator_config,
            )
            logger.info(
                "Initialized external calculator: "
                f"{self.external_calculator_config.get('name', None)}"
            )

        if self.shell_mode:
            if self.external_calculator is None:
                raise ValueError(
                    "Shell mode (internal_calculator_weight=0 without uncertainty_calculator) "
                    "requires -cp / --calculator_patch and Server.external_calculator.name"
                )
            self.servers["external"] = Server(
                self.config,
                model=None,
                model_path=None,
                out_dir=self.out_dir,
                transform=None,
                external_calculator=self.external_calculator,
                external_calculator_config=self.external_calculator_config,
                internal_calculator_weight=self.internal_calculator_weight,
                uncertainty_calculator_config=self.uncertainty_calculator_config,
            )
            logger.info("Listening in external-calculator shell mode (no internal MLFF)")
            return

        if self.model_dir is None:
            raise ValueError(
                "model_dir (-m) is required when internal calculator is active "
                "(internal_calculator_weight != 0 or uncertainty_calculator is set)"
            )

        model_config_path = (
            os.path.join(self.model_dir, "config.yaml")
            if model_config_path is None
            else model_config_path
        )
        model_config = YamlHandler(model_config_path).read_yaml()
        logger.info("Model Config: {}".format(model_config))
        self.transform = Transform(
            model_config.Datahub.get(
                "global_transforms",
                model_config.Datahub.get("transforms", None),
            ),
            simulation_mode=True,
        )
        for FF_key, FF_params in model_config.Modelhub.internal_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        for FF_key, FF_params in model_config.Modelhub.external_FFs.items():
            if FF_params.get("active", False):
                self._init_model(FF_key, FF_params)
        if not self.servers:
            raise ValueError(
                "No active force fields found in Modelhub; set active: true or use shell mode"
            )

    def _init_model(self, FF_key, FF_params):
        model_str = get_model_str(FF_key, FF_params)
        model = build_model(FF_params.architecture, FF_params.layers, FF_params.build_params)
        model_path = get_pretrain_path(os.path.join(self.model_dir, model_str), "best")
        self.servers[FF_key] = Server(
            self.config,
            model=model,
            model_path=model_path,
            out_dir=self.out_dir,
            transform=self.transform,
            external_calculator=self.external_calculator,
            external_calculator_config=self.external_calculator_config,
            internal_calculator_weight=self.internal_calculator_weight,
            uncertainty_calculator_config=self.uncertainty_calculator_config,
        )

    def run_calculate(self):
        info = request.get_json()
        model_key = info.get("model_key", list(self.servers.keys())[0])
        start_time = time.time()
        raw_outputs = self.servers[model_key].calculate(info)
        end_time = time.time()
        new_outputs = {
            k: v[0].tolist() if hasattr(v[0], "tolist") else v[0]
            for k, v in raw_outputs.items()
        }
        results = {"outputs": new_outputs}
        results["units"] = {
            "Hartree_in_E": self.servers[model_key].Hartree_in_E,
            "Bohr_in_R": self.servers[model_key].Bohr_in_R,
        }
        logger.info(
            f"Serving results for {info.get('input_file', '<unknown>')} "
            f"with model {model_key} in {end_time - start_time} seconds"
        )
        return jsonify(results)

    def listen(self):
        # Configure waitress logger to use our logger's handlers
        waitress_logger = logging.getLogger("waitress")
        waitress_logger.handlers = []  # Clear existing handlers
        waitress_logger.addHandler(logger.handlers[0])  # Add console handler
        if len(logger.handlers) > 1:
            waitress_logger.addHandler(logger.handlers[1])  # Add file handler
        waitress_logger.setLevel(logging.INFO)
        waitress_logger.propagate = False  # Prevent duplicate messages
        global _ff_listen_instance
        _ff_listen_instance = self
        waitress.serve(app, listen=self.bind)

    def stop_signal(self):
        logger.info("Stop signal received")


@app.route("/calculate", methods=["POST"])
def run():
    return _ff_listen_instance.run_calculate()


@app.route("/shutdown", methods=["POST"])
def shutdown():
    import threading
    import os

    def shutdown_server():
        _ff_listen_instance.stop_signal()
        time.sleep(1)  # Give time for the response to be sent
        os._exit(0)  # Force exit the process

    # Start shutdown in a separate thread to allow response to be sent
    threading.Thread(target=shutdown_server).start()
    return jsonify({"message": "Server shutdown initiated"}), 200
