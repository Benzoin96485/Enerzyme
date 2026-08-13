from functools import partial
from typing import Iterable, Optional, Callable, Dict, Any, Literal, List
from collections import defaultdict
import time, os, logging, contextlib, warnings
from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch_ema import ExponentialMovingAverage
try:
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
except:
    pass
from transformers.optimization import get_scheduler
import numpy as np
from .splitter import Splitter
from .batch import _decorate_batch_input, _decorate_batch_output, _to_device, _decorate_pyg_batch_input, _pyg_to_device
from .generator_ode import generator_ode_predict_enabled, generator_predict_forward
from .monitor import Monitor
from .optimizer import get_optimizer, get_optimizer_config
from ..data.transform import Transform
from ..utils import logger
from .metrics import Metrics
from .distributed import (
    LaunchEnv,
    bind_single_visible_gpu,
    destroy_process_group,
    detect_launch_env,
    export_torchrun_env,
    infer_num_workers,
    init_process_group,
    is_distributed_runtime,
    is_global_zero,
    validate_distributed_launch,
)


DTYPE_MAPPING = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float": torch.float32,
    "double": torch.float64,
    "single": torch.float32
}

_OBSOLETE_TRAINER_KEYS = ("accelerator", "devices", "num_nodes", "strategy", "precision")


class _NoOpBar:
    """Stand-in for tqdm on non-rank0 processes."""

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def update(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def _unwrap_ddp(model: Module) -> Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _require_nonempty_train_loader(
    n_batches: int,
    n_samples: int,
    batch_size: int,
    world_size: int = 1,
) -> None:
    """Fail fast when drop_last would yield zero optimizer steps."""
    if n_batches > 0:
        return
    if world_size > 1:
        raise RuntimeError(
            f"DDP training would run zero optimizer steps: {n_samples} train "
            f"samples, world_size={world_size}, batch_size={batch_size}, "
            f"drop_last=True. Need at least {batch_size * world_size} samples "
            f"(one full batch per rank). Use a single GPU for tiny/smoke "
            f"splits, or lower batch_size / GPU count."
        )
    raise RuntimeError(
        f"Training would run zero optimizer steps: {n_samples} samples with "
        f"batch_size={batch_size} and drop_last=True. Lower batch_size or "
        f"add data."
    )


def _convert_lightning_model_state_dict(lightning_model_state_dict: Dict) -> Dict:
    model_state_dict = dict()
    for k, v in lightning_model_state_dict.items():
        if k.startswith("model."):
            model_state_dict[k[6:]] = v
        else:
            model_state_dict[k] = v
    return model_state_dict


def _convert_lightning_state_dict(lightning_loaded_info: Dict) -> Dict:
    loaded_info = dict()
    if "state_dict" in lightning_loaded_info:
        loaded_info["model_state_dict"] = _convert_lightning_model_state_dict(lightning_loaded_info["state_dict"])
    if "epoch" in lightning_loaded_info:
        loaded_info["epoch"] = lightning_loaded_info["epoch"]
    if "lr_schedulers" in lightning_loaded_info:
        loaded_info["scheduler_state_dict"] = lightning_loaded_info["lr_schedulers"][0]
    if 'optimizer_states' in lightning_loaded_info:
        loaded_info["optimizer_state_dict"] = lightning_loaded_info["optimizer_states"][0]
    if "callbacks" in lightning_loaded_info:
        for k, v in lightning_loaded_info["callbacks"].items():
            if k == "EMACallback":
                loaded_info["ema_state_dict"] = v
            elif k.startswith("EarlyStopping"):
                if 'best_score' in v:
                    loaded_info["best_score"] = v['best_score']
                if 'wait_count' in v and "epoch" in loaded_info:
                    loaded_info["best_epoch"] = loaded_info["epoch"] - v['wait_count']
    return loaded_info


def _load_state_dict(model: Module, device: Optional[torch.device]=None, pretrain_path: Optional[str]=None, ema: Optional[ExponentialMovingAverage]=None, inference: bool=False, optimizer: Optional[Optimizer]=None, scheduler: Optional[LRScheduler]=None, strict: bool=True) -> Dict:
    other_info = dict()
    if pretrain_path is None:
        return other_info
    loaded_info = torch.load(pretrain_path, map_location=device, weights_only=False)
    if 'pytorch-lightning_version' in loaded_info:
        loaded_info = _convert_lightning_state_dict(loaded_info)
    if ema is not None and "ema_state_dict" in loaded_info:
        model.load_state_dict(loaded_info["model_state_dict"])
        ema.load_state_dict(loaded_info["ema_state_dict"])
        logger.info(f"loading ema state dict from {pretrain_path}...")
    else:
        if inference and "ema_state_dict" in loaded_info:
            model.load_state_dict(loaded_info["model_state_dict"])
            tmp_ema = ExponentialMovingAverage(model.parameters(), decay=1, use_num_updates=True)
            tmp_ema.load_state_dict(loaded_info["ema_state_dict"])
            tmp_ema.copy_to(model.parameters())
            logger.info(f"loading averaged model state dict from {pretrain_path}...")
        else:
            model.load_state_dict(loaded_info["model_state_dict"], strict=strict)
            other_info["model_state_dict"] = loaded_info["model_state_dict"]
            logger.info(f"loading model state dict from {pretrain_path}...")
    if not inference:
        if optimizer is not None and "optimizer_state_dict" in loaded_info:
            optimizer.load_state_dict(loaded_info["optimizer_state_dict"])
            logger.info(f"loading optimizer state dict from {pretrain_path}...")
        if scheduler is not None and "scheduler_state_dict" in loaded_info:
            scheduler.load_state_dict(loaded_info["scheduler_state_dict"])
            logger.info(f"loading scheduler state dict from {pretrain_path}...")
        if "epoch" in loaded_info:
            other_info["epoch"] = loaded_info["epoch"]
        if "best_epoch" in loaded_info:
            other_info["best_epoch"] = loaded_info["best_epoch"]
        if "best_score" in loaded_info:
            other_info["best_score"] = loaded_info["best_score"]
    # if "torch_rng_state" in loaded_info:
    #     torch.random.set_rng_state(loaded_info["torch_rng_state"])
    #     logger.info(f"loading torch random generator state from {pretrain_path}...")
    # if "torch_cuda_rng_state_all" in loaded_info:
    #     torch.cuda.random.set_rng_state_all(loaded_info["torch_cuda_rng_state_all"])
    #     logger.info(f"loading torch cuda random generator state from {pretrain_path}...")
    # if "np_rng_state" in loaded_info:
    #     np.random.set_state(loaded_info["np_rng_state"])
    #     logger.info(f"loading numpy random generator state from {pretrain_path}...")
    return other_info


class Trainer:
    def __init__(self, out_dir: str=None, metric_config: Dict=dict(), **params) -> None:
        '''
        The trainer class for training and evaluating the model.

        Params:
        ----------
        out_dir: str
            The directory to save the model.

        metric_config: dict
            The configuration for the :doc:`Metrics <enerzyme.tasks.metrics.Metrics>` class.

        **params: dict
            The configuration for the trainer.
        
        '''
        self.batch_size = params.get('batch_size', 8)
        self.pyg = params.get("pyg", False)
        if "lightning" in params:
            warnings.warn(
                "Trainer.lightning is deprecated and ignored. Multi-GPU training "
                "is enabled automatically when launched with srun/torchrun "
                "(one process per GPU).",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(
                "Trainer.lightning is deprecated and ignored. DDP follows the "
                "external launcher (srun/torchrun), not a YAML flag."
            )
        for key in _OBSOLETE_TRAINER_KEYS:
            if key in params:
                warnings.warn(
                    f"Trainer.{key} is deprecated and ignored. DDP world size "
                    "comes from srun/torchrun (one process per GPU).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                logger.warning(
                    f"Trainer.{key} is deprecated and ignored. Use srun/torchrun "
                    "with one process per GPU; Enerzyme does not spawn."
                )
        self.patience = params.get('patience', 50)
        self.max_norm = params.get('max_norm', -1)
        self.config = params
        self.out_dir = out_dir
        self.metric_config = metric_config
        self.metrics = Metrics(metric_config)
        self.optimizer_name, self.optimizer_hyper_params = get_optimizer_config(**params)
        self.splitter = Splitter(**params["Splitter"])
        if "Monitor" in params:
            self.monitor = Monitor(**params["Monitor"])
        else:
            self.monitor = None
        self.seed = params.get('seed', 114514)
        self.inference_batch_size = params.get('inference_batch_size', self.batch_size)
        self.max_epochs = params.get('max_epochs', 1000)
        self.warmup_ratio = params.get('warmup_ratio', 0.01)
        self.cuda = params.get('cuda', False)
        self.schedule = params.get('schedule', "linear")
        self.data_in_memory = params.get("data_in_memory", True)
        self.use_ema = params.get("use_ema", True)
        self.ema_decay = params.get("ema_decay", 0.999)
        self.ema_use_num_updates = params.get("ema_use_num_updates", True)
        self.dtype = DTYPE_MAPPING[params.get('dtype', "float32")]
        self.committee_size = params.get("committee_size", 1)
        self.dump_interval = params.get("dump_interval", -1)
        self.active_learning_params = params.get("active_learning_params", None)
        if self.active_learning_params is not None and self.active_learning_params.get("active", False):
            self.active_learning = True
        else:
            self.active_learning = False
        self.resume = params.get("resume", 1)
        self.refresh_best_score = params.get("refresh_best_score", None)
        self.refresh_patience = params.get("refresh_patience", None)
        self.freeze_pretrain_weights = params.get("freeze_pretrain_weights", False)
        self.otf_graph = params.get("otf_graph", True)
        self.generator_config = params.get("Generator")
        non_target_features = params.get("non_target_features", [])
        self.reset_parameters = params.get("reset_parameters", False)
        self.find_unused_parameters = params.get("find_unused_parameters", True)
        self.ddp_timeout_minutes = params.get("ddp_timeout_minutes", 30)
        self.tensorboard = params.get("tensorboard", True)
        self.tensorboard_log_interval = max(1, int(params.get("tensorboard_log_interval", 1)))

        self._launch: LaunchEnv = detect_launch_env()
        validate_distributed_launch(self._launch)
        self._use_ddp = (
            self._launch.world_size > 1
            and self._launch.mode in ("slurm_srun", "torchrun")
        )
        self._is_rank0 = is_global_zero(self._launch)
        self.num_workers = infer_num_workers(params.get("num_workers", -1), env=self._launch)
        if self._is_rank0:
            logger.info(f"using {self.num_workers} workers for dataloader")

        if isinstance(non_target_features, list):
            self.non_target_features = non_target_features
        elif isinstance(non_target_features, str):
            self.non_target_features = [non_target_features]
        else:
            raise ValueError(f"non_target_features must be a list or a string, but got {type(non_target_features)}")

        if self._use_ddp:
            export_torchrun_env(self._launch)
            bind_single_visible_gpu(self._launch, cuda=self.cuda)
            if torch.cuda.is_available() and self.cuda:
                torch.cuda.set_device(0)
                self.device = torch.device("cuda:0")
                if self._is_rank0:
                    logger.info(
                        f"DDP: world_size={self._launch.world_size}, "
                        f"mode={self._launch.mode}, device=cuda:0, "
                        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, "
                        f"MASTER_PORT={os.environ.get('MASTER_PORT')}"
                    )
            else:
                self.device = torch.device("cpu")
                if self._is_rank0:
                    logger.info(
                        f"DDP: world_size={self._launch.world_size}, "
                        f"mode={self._launch.mode}, device=cpu"
                    )
        elif torch.cuda.is_available():
            if self._is_rank0:
                logger.info("GPU found!")
            self.device = torch.device("cuda:0" if self.cuda else "cpu")
        else:
            if self._is_rank0:
                logger.info("GPU not found, turn to CPU!")
            self.device = torch.device("cpu")

    def _decorate_batch_input_impl(self, batch, generator_training: bool):
        if self.pyg:
            return _decorate_pyg_batch_input(
                batch,
                self.dtype,
                self.device,
                self.otf_graph,
                self.generator_config,
                generator_training,
            )
        return _decorate_batch_input(
            batch,
            self.dtype,
            self.device,
            self.otf_graph,
            self.generator_config,
            generator_training,
        )

    def decorate_batch_input(self, batch):
        """Training collate: random ``flow_t`` and linear interpolation for CFM."""
        return self._decorate_batch_input_impl(batch, generator_training=True)

    def decorate_eval_batch_input(self, batch):
        """Eval / predict collate: ``flow_t=0``, flow state at init (ODE start)."""
        return self._decorate_batch_input_impl(batch, generator_training=False)
        
    def to_device(self, batch):
        if self.pyg:
            return _pyg_to_device(batch, self.device)
        else:
            return _to_device(batch, self.device)
    
    def decorate_batch_output(self, output, features, targets):
        return _decorate_batch_output(output, features, targets, self.non_target_features)
    
    def _set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _dataloader_kwargs(
        self,
        num_workers: int,
        *,
        pin_memory: bool = False,
        distributed: bool = False,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"num_workers": num_workers}
        if pin_memory:
            kwargs["pin_memory"] = True
        if num_workers > 0 and (pin_memory or distributed):
            # HDF5: avoid fork workers sharing open handles under DDP.
            kwargs["persistent_workers"] = True
            kwargs["multiprocessing_context"] = "forkserver"
        return kwargs

    def _progress_bar(self, **kwargs):
        if self._is_rank0:
            return tqdm(**kwargs)
        return _NoOpBar()

    def _reduce_mean_scalar(self, value: float, count: float = 1.0) -> float:
        """All-reduce a scalar mean across ranks (no-op when not DDP).

        ``value`` is a local mean; ``count`` is the number of local samples
        (or batches) that went into it.
        """
        if not is_distributed_runtime():
            return 0.0 if count <= 0 else float(value)
        import torch.distributed as dist

        total = torch.tensor(
            [float(value) * float(max(count, 0.0)), float(max(count, 0.0))],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if float(total[1].item()) <= 0:
            return 0.0
        return float(total[0].item() / total[1].item())

    def _reduce_metric_scores(self, partials: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not is_distributed_runtime():
            return self.metrics.cal_metric_from_partials(partials)
        import torch.distributed as dist

        keys = list(self.metrics.metric_config.keys())
        if not keys:
            return {"_judge_score": 0.0}
        error_sums = torch.tensor(
            [float(partials.get(k, {}).get("error_sum", 0.0)) for k in keys],
            dtype=torch.float64,
            device=self.device,
        )
        counts = torch.tensor(
            [float(partials.get(k, {}).get("count", 0.0)) for k in keys],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(error_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        merged = {
            key: {
                "error_sum": float(error_sums[i].item()),
                "count": float(counts[i].item()),
            }
            for i, key in enumerate(keys)
        }
        return self.metrics.cal_metric_from_partials(merged)

    def _gather_prediction_dicts(
        self,
        y_preds: Dict[str, List],
        y_truths: Dict[str, List],
    ) -> tuple:
        if not is_distributed_runtime():
            return y_preds, y_truths
        import torch.distributed as dist

        gathered_preds = [None] * dist.get_world_size()
        gathered_truths = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_preds, dict(y_preds))
        dist.all_gather_object(gathered_truths, dict(y_truths))
        merged_preds = defaultdict(list)
        merged_truths = defaultdict(list)
        for rank_preds, rank_truths in zip(gathered_preds, gathered_truths):
            for key, values in rank_preds.items():
                merged_preds[key].extend(values)
            for key, values in rank_truths.items():
                merged_truths[key].extend(values)
        return dict(merged_preds), dict(merged_truths)

    def _summarize_monitor(self) -> None:
        """Log Monitor stats over the full validation set (gather under DDP)."""
        if self.monitor is None:
            return
        if is_distributed_runtime():
            import torch.distributed as dist

            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self.monitor.collection)
            if self._is_rank0:
                merged = {key: [] for key in self.monitor.terms}
                for shard in gathered:
                    if not shard:
                        continue
                    for key, values in shard.items():
                        merged.setdefault(key, []).extend(values)
                self.monitor.collection = merged
                self.monitor.summary()
            else:
                self.monitor._reset()
            return
        self.monitor.summary()

    def _open_tensorboard(self, dump_dir: str):
        if not self.tensorboard or not self._is_rank0:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning(
                "tensorboard is not installed; skipping SummaryWriter. "
                "Install the tensorboard package or set Trainer.tensorboard: false."
            )
            return None
        tb_dir = os.path.join(dump_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        logger.info(f"TensorBoard logdir: {tb_dir}")
        return SummaryWriter(log_dir=tb_dir)

    def load_state_dict(
        self, 
        model: Module, 
        optimizer: Optional[Optimizer]=None,
        scheduler: Optional[LRScheduler]=None,
        pretrain_path: Optional[str]=None, 
        ema: Optional[ExponentialMovingAverage]=None, 
        inference: bool=False,
        strict: bool=True
    ) -> None:
        return _load_state_dict(
            _unwrap_ddp(model),
            self.device,
            pretrain_path,
            ema,
            inference,
            optimizer,
            scheduler,
            strict,
        )

    def save_state_dict(self, model: Module, optimizer: Optimizer, scheduler: LRScheduler, dump_dir: str, ema: Optional[ExponentialMovingAverage]=None, suffix="last", model_rank=None, epoch: Optional[int]=None, best_score: Optional[float]=None, best_epoch: Optional[int]=None):
        if self._is_rank0:
            if model_rank is None:
                model_rank = ''
            os.makedirs(dump_dir, exist_ok=True)
            raw_model = _unwrap_ddp(model)
            info = {"model_state_dict": raw_model.state_dict()}
            if ema is not None:
                info["ema_state_dict"] = ema.state_dict()
            info["optimizer_state_dict"] = optimizer.state_dict()
            info["scheduler_state_dict"] = scheduler.state_dict()
            if epoch is not None:
                info["epoch"] = epoch
            if best_score is not None:
                info["best_score"] = best_score
            if best_epoch is not None:
                info["best_epoch"] = best_epoch
            torch.save(info, os.path.join(dump_dir, f'model{model_rank}_{suffix}.pth'))
        if is_distributed_runtime():
            import torch.distributed as dist
            dist.barrier()

    def fit_predict(self, 
        model: Module, pretrain_path: Optional[str],
        train_dataset: Dataset, valid_dataset: Optional[Dataset], 
        loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, 
        test_dataset: Optional[Dataset]=None, model_rank: Optional[int]=None, max_epoch_per_iter: int=-1,
        meta_state_dict: Dict=dict(), refresh_patience: bool=False, refresh_best_score: bool=False
    ) -> Dict[Literal["y_pred", "y_truth", "metric_score"], Any]:
        '''
        Train the model on the training set, validate it on the validation set, and test the model on the test set.

        Params:
        ----------
        model: Module
            The model to train

        pretrain_path: Optional[str]
            The path to the pretrained model or the checkpoint of the model

        train_dataset: Dataset
            The training dataset.

        valid_dataset: Optional[Dataset]
            The validation dataset. If not provided, the model will not be validated.

        loss_terms: Iterable[Callable]
            The loss functions with multiple terms to use.

        dump_dir: str
            The directory to save the model

        transform: Transform
            The data transform in preprocessing. The inverse transform will be applied to the prediction results when calculating the metrics during validation and testing.

        test_dataset: Optional[Dataset]
            The test dataset. If not provided, the model will not be tested.

        model_rank: Optional[int]
            Only used in deep ensemble training. The rank of the model.

        max_epoch_per_iter: int
            Only used in active learning. The maximum number of epochs per active learning iteration.

        meta_state_dict: Dict
            Only used in active learning checkpointing. The meta state dictionary.

        refresh_patience: bool
            Whether to refresh the patience when loading the checkpoint.

        refresh_best_score: bool
            Whether to refresh the best score when loading the checkpoint.

        Returns:
        ----------
        The prediction results on the test set. A dictionary with the following keys:

        y_pred
            The predicted results.

        y_truth
            The true results.
            
        metric_score
            The metrics score based on the predicted and true results.

        '''
        if self.refresh_best_score is not None:
            refresh_best_score = self.refresh_best_score
        if self.refresh_patience is not None:
            refresh_patience = self.refresh_patience
        self._set_seed(self.seed + (model_rank if model_rank is not None else 0))
        validate_distributed_launch(self._launch)

        pg_initialized = False
        tb_writer = None
        try:
            if self._use_ddp:
                pg_initialized = init_process_group(
                    self._launch,
                    timeout_minutes=self.ddp_timeout_minutes,
                    backend="nccl" if self.device.type == "cuda" else "gloo",
                )

            train_sampler = None
            if self._use_ddp:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self._launch.world_size,
                    rank=self._launch.global_rank,
                    shuffle=True,
                    drop_last=True,
                    seed=self.seed + (model_rank if model_rank is not None else 0),
                )
            pin_memory = self._use_ddp and self.device.type == "cuda"
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=self.decorate_batch_input,
                drop_last=True,
                **self._dataloader_kwargs(
                    self.num_workers,
                    pin_memory=pin_memory,
                    distributed=self._use_ddp,
                ),
            )
            _require_nonempty_train_loader(
                len(train_dataloader),
                len(train_dataset),
                self.batch_size,
                world_size=self._launch.world_size if self._use_ddp else 1,
            )
            num_training_steps = len(train_dataloader) * self.max_epochs
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            optimizer = get_optimizer(self.optimizer_name, model, self.optimizer_hyper_params)
            scheduler = get_scheduler(self.schedule, optimizer, num_warmup_steps, num_training_steps)

            if self.reset_parameters:
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        logger.info(f"Resetting parameters for {m.__class__.__name__}")
                        m.reset_parameters()

            model = model.to(device=self.device, dtype=self.dtype)
            if self.use_ema:
                ema = ExponentialMovingAverage(
                    model.parameters(),
                    self.ema_decay,
                    self.ema_use_num_updates,
                )
            else:
                ema = None
            if self.resume > 1:
                other_info = self.load_state_dict(model, optimizer, scheduler, pretrain_path, ema)
            elif self.resume == 1:
                other_info = self.load_state_dict(model, pretrain_path=pretrain_path, ema=ema)
            else:
                other_info = self.load_state_dict(model, pretrain_path=pretrain_path, inference=True, strict=False)

            if self.freeze_pretrain_weights:
                for name, param in model.named_parameters():
                    if name in other_info.get("model_state_dict", dict()):
                        param.requires_grad = False
                        logger.info(f"Freezing pretrained weights for {name}")

            if self._use_ddp:
                ddp_device_ids = [self.device.index] if self.device.type == "cuda" else None
                model = DistributedDataParallel(
                    model,
                    device_ids=ddp_device_ids,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    find_unused_parameters=self.find_unused_parameters,
                )

            if self.resume > 1 and "best_epoch" in other_info and "epoch" in other_info:
                wait = other_info["epoch"] - other_info["best_epoch"]
                if refresh_best_score:
                    best_score = float("inf")
                else:
                    best_score = other_info.get("best_score", float("inf"))
                start_epoch = other_info["epoch"] + 1
                if (wait >= self.patience and refresh_patience) or refresh_best_score:
                    wait = 0
            else:
                wait = 0
                if self.resume > 1:
                    start_epoch = other_info.get("epoch", -1) + 1
                else:
                    start_epoch = 0
                if valid_dataset is not None:
                    if self.resume > 1 and not refresh_best_score:
                        best_score = other_info.get("best_score", float("inf"))
                    else:
                        best_score = float("inf")
                else:
                    best_score = None

            if self.resume > 1:
                max_epochs = self.max_epochs
            else:
                max_epochs = start_epoch + self.max_epochs
            
            if valid_dataset is not None:
                if self.resume > 1:
                    best_epoch = other_info.get("best_epoch", start_epoch)
                else:
                    best_epoch = None
            else:
                best_epoch = None

            tb_writer = self._open_tensorboard(dump_dir)
            global_step = start_epoch * max(1, len(train_dataloader))

            epoch = start_epoch
            epoch_in_iter = meta_state_dict.get("epoch_in_iter", 0)
            if start_epoch > 0 and self._is_rank0:
                if epoch_in_iter > 0:
                    logger.info(f"Resuming from epoch {start_epoch + 1}, epoch {epoch_in_iter + 1} in active learning iteration")
                else:
                    logger.info(f"Resuming from epoch {start_epoch + 1}...")
            for epoch in range(start_epoch, max_epochs):
                if max_epoch_per_iter > 0 and epoch_in_iter >= max_epoch_per_iter:
                    break
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                model = model.train()
                start_time = time.time()
                batch_bar = self._progress_bar(
                    total=len(train_dataloader), dynamic_ncols=True, leave=False, position=0,
                    desc='Train', ncols=5
                )
                trn_loss = []
                
                for i, batch in enumerate(train_dataloader):
                    net_input, net_target = self.to_device(batch)
                    
                    loss = 0
                    with torch.set_grad_enabled(True):
                        output = model(net_input)
                        for loss_term in loss_terms.values():
                            loss += loss_term(output, net_target)
                    trn_loss.append(float(loss.data))

                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, max_epochs),
                        loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                        lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
                    )

                    # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    if self.max_norm > 0:
                        clip_grad_norm_(model.parameters(), self.max_norm)

                    optimizer.step()
                    global_step += 1
                    if tb_writer is not None and global_step % self.tensorboard_log_interval == 0:
                        # Rank0 batch loss only (no per-step all_reduce). lr is the value used this step.
                        tb_writer.add_scalar("train_loss", float(loss.detach().item()), global_step)
                        tb_writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), global_step)

                    if self.use_ema:
                        ema.update()
                    
                    scheduler.step()
                    batch_bar.update()

                batch_bar.close()
                total_trn_loss = np.mean(trn_loss) if trn_loss else 0.0
                total_trn_loss_epoch = self._reduce_mean_scalar(
                    float(total_trn_loss), count=float(len(trn_loss))
                )
                message = f'Epoch [{epoch + 1}/{max_epochs}]' + (f' ({epoch_in_iter + 1}/{max_epoch_per_iter})' if max_epoch_per_iter > 0 else '') + f', train_loss: {total_trn_loss_epoch:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'

                if self.use_ema:
                    cm = ema.average_parameters()
                else:
                    cm = contextlib.nullcontext()

                if valid_dataset is not None:
                    with cm:
                        predict_result = self.predict(
                            model=model, 
                            dataset=valid_dataset, 
                            loss_terms=loss_terms, 
                            dump_dir=dump_dir, 
                            transform=transform, 
                            epoch=epoch, 
                            load_model=False,
                        )
                        val_loss = predict_result["val_loss"]
                        metric_score = predict_result["metric_score"]
                        total_val_loss = float(np.mean(val_loss)) if len(val_loss) else 0.0
                        _score = metric_score["_judge_score"]
                        _metric = str(self.metrics)
                        save_handle = partial(self.save_state_dict, model=model, optimizer=optimizer, scheduler=scheduler, dump_dir=dump_dir, ema=ema, suffix="best", model_rank=model_rank)
                        is_early_stop, best_score, wait, saved = self._early_stop_choice(wait, best_score, metric_score, save_handle, self.patience, epoch)
                        if saved:
                            best_epoch = epoch
                        message += f', val_loss: {total_val_loss:.4f}, ' + \
                            ", ".join([f'val_{k}: {v:.4f}' for k, v in metric_score.items() if k != "_judge_score"]) + \
                            f', val_judge_score ({_metric}): {_score:.4f}' + \
                            (f', Patience [{wait}/{self.patience}], min_val_judge_score: {best_score:.4f}' if wait else '')
                        if tb_writer is not None:
                            tb_writer.add_scalar("train_loss_epoch", total_trn_loss_epoch, epoch + 1)
                            tb_writer.add_scalar("val_loss", total_val_loss, epoch + 1)
                            for key, value in metric_score.items():
                                tb_writer.add_scalar(key, float(value), epoch + 1)
                            if best_score is not None:
                                tb_writer.add_scalar("best_score", float(best_score), epoch + 1)
                            tb_writer.add_scalar("patience_wait", float(wait), epoch + 1)
                else:
                    is_early_stop = False
                    if tb_writer is not None:
                        tb_writer.add_scalar("train_loss_epoch", total_trn_loss_epoch, epoch + 1)

                epoch_in_iter += 1
                meta_state_dict.update({"epoch_in_iter": epoch_in_iter})   
                end_time = time.time()
                message += f', {(end_time - start_time):.1f}s'
                if self._is_rank0:
                    logger.info(message)
                self.save_state_dict(model, optimizer, scheduler, dump_dir, ema, "last", model_rank, epoch=epoch, best_score=best_score, best_epoch=best_epoch)
                if self.dump_interval > 0 and (epoch + 1) % self.dump_interval == 0:
                    self.save_state_dict(model, optimizer, scheduler, dump_dir, ema, f"epoch={epoch}", model_rank, epoch=epoch, best_score=best_score, best_epoch=best_epoch)
                if is_early_stop:
                    break

            meta_state_dict.update({"model_rank": model_rank + 1 if model_rank is not None else 0, "epoch_in_iter": 0})
            if test_dataset is not None:
                if self.use_ema:
                    cm = ema.average_parameters()
                else:
                    cm = contextlib.nullcontext()
                with cm:
                    predict_result = self.predict(
                        model=model, 
                        dataset=test_dataset, 
                        loss_terms=loss_terms, 
                        dump_dir=dump_dir,  
                        transform=transform, 
                        epoch=epoch, 
                        load_model=True,
                        model_rank=model_rank
                    )
                    y_pred = predict_result["y_pred"]
                    y_truth = predict_result["y_truth"]
                    metric_score = predict_result["metric_score"]
            else:
                y_pred = None
                y_truth = None
                metric_score = None
            return {"y_pred": y_pred, "y_truth": y_truth, "metric_score": metric_score}
        finally:
            if tb_writer is not None:
                tb_writer.flush()
                tb_writer.close()
            if pg_initialized:
                destroy_process_group()
    
    def _early_stop_choice(self, wait, min_loss, metric_score, save_handle, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, save_handle, patience, epoch)
    
    def predict(self, model: Module, dataset: Dataset, loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, epoch: int=1, load_model: bool=False, model_rank: Optional[str]=None, test_mode: bool=False) -> Dict[Literal["y_pred", "y_truth", "val_loss", "metric_score"], Any]:
        self._set_seed(self.seed)
        raw_model = _unwrap_ddp(model)
        raw_model = raw_model.to(device=self.device, dtype=self.dtype)
        if load_model == True:
            from ..models import get_pretrain_path
            pretrain_path = get_pretrain_path(dump_dir, "best", model_rank)
            if pretrain_path is not None:
                self.load_state_dict(raw_model, pretrain_path=pretrain_path, inference=True)
            else:
                if self._is_rank0:
                    logger.warning(f"Pretrained model not found at {dump_dir}, using random weights")
            
        use_ddp = is_distributed_runtime()
        if use_ddp:
            shard_indices = list(range(self._launch.global_rank, len(dataset), self._launch.world_size))
            eval_dataset = Subset(dataset, shard_indices) if shard_indices else Subset(dataset, [])
        else:
            eval_dataset = dataset
        pin_memory = use_ddp and self.device.type == "cuda"
        dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            collate_fn=self.decorate_eval_batch_input,
            **self._dataloader_kwargs(
                self.num_workers,
                pin_memory=pin_memory,
                distributed=use_ddp,
            ),
        )
        raw_model = raw_model.eval()
        raw_model.test_mode = test_mode
        batch_bar = self._progress_bar(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = defaultdict(list)
        y_truths = defaultdict(list)
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.to_device(batch)
            if generator_ode_predict_enabled(self.generator_config):
                with torch.no_grad():
                    output = generator_predict_forward(
                        raw_model, net_input, self.generator_config
                    )
            else:
                output = raw_model(net_input)
            # Get model outputs
            if self.monitor is not None:
                self.monitor.collect(output)
            loss = 0
            with torch.no_grad():
                if not load_model:
                    for loss_term in loss_terms.values():
                        loss += loss_term(output, net_target)
                    val_loss.append(float(loss.data))
            y_pred, y_truth = self.decorate_batch_output(output, net_input, net_target)
            for k, v in y_pred.items():
                y_preds[k].extend(v)
            for k, v in y_truth.items():
                y_truths[k].extend(v)
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))) if val_loss else "n/a"
                )
            batch_bar.update()

        if self.monitor is not None:
            self._summarize_monitor()

        if transform is not None:
            transform.inverse_transform(y_preds)
            transform.inverse_transform(y_truths)
        if use_ddp:
            partials = self.metrics.accumulate_partials(y_truths, y_preds)
            metric_score = self._reduce_metric_scores(partials)
            if not load_model:
                local_count = float(len(val_loss))
                local_mean = float(np.mean(val_loss)) if local_count else 0.0
                total_val_loss = self._reduce_mean_scalar(local_mean, local_count)
                val_loss = [total_val_loss]
            if load_model:
                y_preds, y_truths = self._gather_prediction_dicts(y_preds, y_truths)
        else:
            metric_score = self.metrics.cal_metric(y_truths, y_preds)
        if load_model and "_judge_score" in metric_score:
            metric_score.pop("_judge_score")
        batch_bar.close()
        raw_model.test_mode = False
        return {"y_pred": y_preds, "y_truth": y_truths, "val_loss": val_loss, "metric_score": metric_score}
