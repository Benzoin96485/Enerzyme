import os
from .utils import YamlHandler, logger
from .data.datahub import DataHub
from .models import ModelHub
from .tasks.trainer import Trainer
from .tasks.distributed import (
    detect_launch_env,
    run_rank0_exclusive,
    validate_distributed_launch,
)


class FFTrain(object):
    def __init__(self, config_path=None, out_dir=None, **params):
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        self.config_path = config_path
        self.out_dir = out_dir
        self.task = config.Base.task
        # Fail fast before any DataHub I/O if a multi-task SLURM job lacks srun/torchrun.
        validate_distributed_launch()
        self.datahub = DataHub(dump_dir=self.out_dir, **config.Datahub)
        logger.info('Config: {}'.format(config))
        self.trainer = Trainer(out_dir=self.out_dir, metric_config=config.Metric, **config.Trainer)
        self.modelhub = ModelHub(self.datahub, self.trainer, **config.Modelhub)

        if self.out_dir is not None:
            launch = detect_launch_env()

            def _write_config():
                # Only rank 0 creates the output dir and writes the resolved config.
                if not os.path.exists(self.out_dir):
                    logger.info('Create output directory: {}'.format(self.out_dir))
                    os.makedirs(self.out_dir)
                else:
                    logger.info('Output directory already exists: {}'.format(self.out_dir))
                    logger.warning('Overwrite output directory: {}'.format(self.out_dir))
                out_path = os.path.join(self.out_dir, 'config.yaml')
                self.yamlhandler.write_yaml(data = config, out_file_path = out_path)

            run_rank0_exclusive(
                _write_config,
                env=launch,
                sync_dir=os.path.abspath(self.out_dir),
                name="fftrain_config",
            )
        
    def train_all(self):
        FFs = self.modelhub.models.get('FF', dict())
        for ff in FFs.values():
            if ff.trainer.active_learning:
                ff.active_learn()
            else:
                ff.train()
