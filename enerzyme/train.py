import os
from .utils import YamlHandler, logger
from .utils.util import atomic_output
from .data.datahub import DataHub
from .models import ModelHub
from .tasks.trainer import Trainer
from .tasks.distributed import detect_launch_env, is_lightning_subprocess, run_rank0_exclusive, validate_distributed_launch


def validate_parallel_config(config, *, training):
    """Check each active model's effective settings before preparing shared data."""
    launch = detect_launch_env()
    models = {**config.Modelhub.get("internal_FFs", {}), **config.Modelhub.get("external_FFs", {})}
    for name, model in models.items():
        if not model.get("active"):
            continue
        settings = {**config.Trainer, **(model.get("Trainer") or {})}
        lightning = settings.get("lightning", False)
        if not lightning:
            validate_distributed_launch(launch)
        committee = settings.get("committee_size", 1) > 1
        active_learning = training and (settings.get("active_learning_params") or {}).get("active", False)
        if committee or active_learning:
            distributed = launch.world_size > 1 or is_lightning_subprocess()
            if lightning and not distributed:
                import torch
                distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1
            if distributed:
                mode = "committee" if committee else "active learning"
                raise ValueError(f"{name}: {mode} requires a single process")


class FFTrain(object):
    def __init__(self, config_path=None, out_dir=None, **params):
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.config_path = config_path
        self.out_dir = out_dir
        self.task = config.Base.task
        validate_parallel_config(config, training=True)
        self.datahub = DataHub(dump_dir=self.out_dir, **config.Datahub)
        logger.info('Config: {}'.format(config))
        self.trainer = Trainer(out_dir=self.out_dir, metric_config=config.Metric, **config.Trainer)
        self.modelhub = ModelHub(self.datahub, self.trainer, **config.Modelhub)

        if self.out_dir is not None:
            def save_config():
                with atomic_output(os.path.join(self.out_dir, "config.yaml")) as path:
                    self.yamlhandler.write_yaml(data=config, out_file_path=path)

            run_rank0_exclusive(save_config, sync_dir=self.out_dir, name="train_config")

    def train_all(self):
        FFs = self.modelhub.models.get('FF', dict())
        for ff in FFs.values():
            if ff.trainer.active_learning:
                ff.active_learn()
            else:
                ff.train()
