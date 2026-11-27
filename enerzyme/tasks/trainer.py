from functools import partial
from typing import Iterable, Optional, Callable, Dict, Any, Literal, List
from collections import defaultdict
import time, os, logging, contextlib, random
import torch.distributed as dist
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
from .batch import (
    CollateBatch, _decorate_batch_input, _decorate_pyg_batch_input, _decorate_batch_output,
    _to_device, _pyg_to_device,
)
from .generator_ode import generator_ode_predict_enabled, generator_predict_forward
from .monitor import Monitor
from .optimizer import get_optimizer, get_optimizer_config
from ..data.transform import Transform
from ..utils import logger
from ..utils.util import atomic_output
from .metrics import Metrics
from .distributed import (
    LaunchEnv,
    native_device_for_launch,
    run_rank0_exclusive,
    destroy_process_group,
    detect_launch_env,
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

def _unwrap_ddp(model: Module) -> Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _strided_shard_indices(rank: int, world_size: int, n_samples: int) -> List[int]:
    """Dataset indices owned by ``rank`` under ``range(rank, n, world_size)`` sharding."""
    return list(range(rank, n_samples, world_size))


def _merge_sharded_prediction_dicts(
    gathered_preds: List[Dict[str, List]],
    gathered_truths: List[Dict[str, List]],
    gathered_indices: List[List[int]],
) -> tuple:
    """Restore original dataset order after rank-strided eval sharding.

    Concatenating rank-order lists would scramble ``test.data`` row alignment
    even though pooled metrics stay correct.
    """
    order = []
    for rank, indices in enumerate(gathered_indices):
        if not indices:
            continue
        for local_i, global_i in enumerate(indices):
            order.append((global_i, rank, local_i))
    order.sort(key=lambda item: item[0])

    pred_keys: List[str] = []
    truth_keys: List[str] = []
    seen_pred = set()
    seen_truth = set()
    for rank_preds, rank_truths in zip(gathered_preds, gathered_truths):
        if rank_preds:
            for key in rank_preds:
                if key not in seen_pred:
                    seen_pred.add(key)
                    pred_keys.append(key)
        if rank_truths:
            for key in rank_truths:
                if key not in seen_truth:
                    seen_truth.add(key)
                    truth_keys.append(key)

    merged_preds = {key: [] for key in pred_keys}
    merged_truths = {key: [] for key in truth_keys}
    for _, rank, local_i in order:
        rank_preds = gathered_preds[rank]
        rank_truths = gathered_truths[rank]
        for key in pred_keys:
            merged_preds[key].append(rank_preds[key][local_i])
        for key in truth_keys:
            merged_truths[key].append(rank_truths[key][local_i])
    return merged_preds, merged_truths


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


def _modify_lightning_state_dict(lightning_ckpt_path: str, patience: int) -> None:
    '''
    Modify the patience of the EarlyStopping callback in the Lightning checkpoint.

    .. danger::

    This function should be only used in the world rank 0 during distributed training. Otherwise, the checkpoint will be corrupted.

    Params:
    ----------
    lightning_ckpt_path: str
        The path to the Lightning checkpoint.
        
    patience: int
        The patience to set.

    Returns:
    ----------
    None
    '''
    loaded_info = torch.load(lightning_ckpt_path, map_location="cpu", weights_only=False)
    modified = False
    for key, state in loaded_info.get("callbacks", {}).items():
        if key.startswith("EarlyStopping") and state.get("patience", patience) < patience:
            state["patience"] = patience
            modified = True
    if modified:
        logger.info(f"Increasing EarlyStopping patience to {patience}")
        with atomic_output(lightning_ckpt_path) as temporary:
            torch.save(loaded_info, temporary)


def _capture_rng(device):
    kind, keys, index, has_gauss, cached = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "numpy": (kind, torch.from_numpy(keys.astype(np.int64)), index, has_gauss, cached),
        "python": random.getstate(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device):
    kind, keys, index, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, keys.cpu().numpy().astype(np.uint32), index, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


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
    if "enerzyme_rank_rng" in lightning_loaded_info:
        loaded_info["rank_states"] = lightning_loaded_info["enerzyme_rank_rng"]
    if "global_step" in lightning_loaded_info:
        loaded_info["global_step"] = lightning_loaded_info["global_step"]
    if "epoch" in lightning_loaded_info:
        loaded_info["epoch"] = lightning_loaded_info["epoch"]
    if lightning_loaded_info.get("lr_schedulers"):
        loaded_info["scheduler_state_dict"] = lightning_loaded_info["lr_schedulers"][0]
    if lightning_loaded_info.get("optimizer_states"):
        loaded_info["optimizer_state_dict"] = lightning_loaded_info["optimizer_states"][0]
    if "callbacks" in lightning_loaded_info:
        for k, v in lightning_loaded_info["callbacks"].items():
            if k == "EMACallback" and v:
                loaded_info["ema_state_dict"] = v
            elif k.startswith("EarlyStopping"):
                if 'best_score' in v:
                    loaded_info["best_score"] = v['best_score']
                if 'wait_count' in v and "epoch" in loaded_info:
                    loaded_info["best_epoch"] = loaded_info["epoch"] - v['wait_count']
                    loaded_info["wait"] = v['wait_count']
    return loaded_info


def _load_state_dict(model: Module, device: Optional[torch.device]=None, pretrain_path: Optional[str]=None, ema: Optional[ExponentialMovingAverage]=None, inference: bool=False, optimizer: Optional[Optimizer]=None, scheduler: Optional[LRScheduler]=None, strict: bool=True) -> Dict:
    other_info = dict()
    if pretrain_path is None:
        return other_info
    loaded_info = torch.load(pretrain_path, map_location=device, weights_only=False)
    if 'pytorch-lightning_version' in loaded_info:
        loaded_info = _convert_lightning_state_dict(loaded_info)
    from ..models.compat import upgrade_checkpoint
    loaded_info = upgrade_checkpoint(model, loaded_info, optimizer)
    model.load_state_dict(loaded_info["model_state_dict"], strict=strict)
    other_info["model_state_dict"] = loaded_info["model_state_dict"]
    if not inference and optimizer is not None and "rank_states" not in loaded_info:
        # Older native best checkpoints were saved inside ema.average_parameters().
        # The weights paired with the optimizer moments live in collected_params there.
        legacy_ema = loaded_info.get("ema_state_dict", {})
        collected = legacy_ema.get("collected_params")
        if collected is not None:
            params = list(model.parameters())
            shadows = legacy_ema["shadow_params"]
            if len(params) == len(collected) == len(shadows) and all(
                torch.equal(param, shadow) for param, shadow in zip(params, shadows)
            ):
                with torch.no_grad():
                    for param, raw in zip(params, collected):
                        param.copy_(raw)
    if "ema_state_dict" in loaded_info and (ema is not None or inference):
        if ema is None:
            ema = ExponentialMovingAverage(model.parameters(), decay=1)
        ema.load_state_dict(loaded_info["ema_state_dict"])
        if inference:
            ema.copy_to()
        logger.info(f"Loading EMA weights from {pretrain_path}")
    elif ema is not None:
        with torch.no_grad():
            for shadow, param in zip(ema.shadow_params, model.parameters()):
                shadow.copy_(param)
    logger.info(f"Loading model state from {pretrain_path}")
    if not inference:
        if optimizer is not None and "optimizer_state_dict" in loaded_info:
            optimizer.load_state_dict(loaded_info["optimizer_state_dict"])
            logger.info(f"loading optimizer state dict from {pretrain_path}...")
        if scheduler is not None and "scheduler_state_dict" in loaded_info:
            scheduler.load_state_dict(loaded_info["scheduler_state_dict"])
            logger.info(f"loading scheduler state dict from {pretrain_path}...")
        for key in ("epoch", "best_epoch", "best_score", "global_step", "rank_states", "sampler_seed", "wait"):
            if key in loaded_info:
                other_info[key] = loaded_info[key]
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

        ``**params``: dict
            The configuration for the trainer.
        
        '''
        self.batch_size = params.get('batch_size', 8)
        self.pyg = params.get("pyg", False)
        self.lightning = params.get("lightning", False)
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
        self.find_unused_parameters = params.get("find_unused_parameters", False)
        self.ddp_timeout_minutes = params.get("ddp_timeout_minutes", 30)
        self.tensorboard = params.get("tensorboard")
        self.tensorboard_log_interval = max(1, int(params.get("tensorboard_log_interval", 1)))

        self._launch: LaunchEnv = detect_launch_env()
        if not self.lightning:
            validate_distributed_launch(self._launch)
        self._distributed_launch = (
            self._launch.world_size > 1
            and self._launch.mode in ("slurm_srun", "torchrun")
        )
        self._use_ddp = self._distributed_launch and not self.lightning
        if self.tensorboard is None:
            self.tensorboard = self._use_ddp
        self._is_rank0 = is_global_zero()
        self.num_workers = infer_num_workers(params.get("num_workers", -1), env=self._launch)
        if self._is_rank0:
            logger.info(f"using {self.num_workers} workers for dataloader")

        if isinstance(non_target_features, list):
            self.non_target_features = non_target_features
        elif isinstance(non_target_features, str):
            self.non_target_features = [non_target_features]
        else:
            raise ValueError(f"non_target_features must be a list or a string, but got {type(non_target_features)}")

        self.device = None
        if self.lightning:
            import lightning as L
            from lightning.pytorch.callbacks import EarlyStopping
            from lightning.pytorch.strategies import DDPStrategy

            # Historically Lightning auto-selected the accelerator when cuda was omitted.
            self.lightning_cuda = params.get("cuda", True)
            accelerator = "auto" if self.lightning_cuda else "cpu"
            strategy = "auto"
            if self._distributed_launch:
                count = self._launch.local_world_size
                if self.lightning_cuda and torch.cuda.is_available():
                    # srun can expose only the assigned GPU to each local process.
                    devices = [torch.device("cuda", i if torch.cuda.device_count() > 1 else 0) for i in range(count)]
                else:
                    devices = [torch.device("cpu")] * count
                strategy = DDPStrategy(parallel_devices=devices, find_unused_parameters=self.find_unused_parameters)
            self.lightning_trainer = L.Trainer(
                default_root_dir=out_dir, accelerator=accelerator, devices="auto",
                strategy=strategy, num_nodes=self._launch.num_nodes,
                callbacks=[EarlyStopping(monitor="_judge_score", mode="min", patience=self.patience)],
                gradient_clip_val=self.max_norm if self.max_norm > 0 else None,
                inference_mode=False, max_epochs=self.max_epochs,
            )
        else:
            self.device = native_device_for_launch(self._launch, self.cuda)
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            logger.info(f"Training device {self.device}, rank={self._launch.global_rank}/{self._launch.world_size}, "
                        f"batch_size={self.batch_size} per process, global_batch_size={self.batch_size * self._launch.world_size}")

    def _collate_batch(self, *, generator_training: bool) -> CollateBatch:
        return CollateBatch(
            pyg=self.pyg,
            dtype=self.dtype,
            device=self.device,
            otf_graph=self.otf_graph,
            generator_config=self.generator_config,
            generator_training=generator_training,
        )

    def _decorate_batch_input_impl(self, batch, generator_training: bool):
        return self._collate_batch(generator_training=generator_training)(batch)

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
        random.seed(seed)

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
            # fork copies open HDF5 handles into workers (unsafe). forkserver
            # pickles dataset/collate into a fresh interpreter instead.
            kwargs["multiprocessing_context"] = "forkserver"
        return kwargs

    def _progress_bar(self, **kwargs):
        return tqdm(disable=not self._is_rank0, **kwargs)

    @contextlib.contextmanager
    def _distributed_session(self):
        owns_group = self._distributed_launch and not dist.is_initialized()
        if owns_group:
            init_process_group(
                self._launch, timeout_minutes=self.ddp_timeout_minutes,
                backend="nccl" if self.device.type == "cuda" else "gloo",
            )
        try:
            yield
        finally:
            if owns_group:
                destroy_process_group()

    def _reduce_mean_scalar(self, value: float, count: float = 1.0) -> float:
        """All-reduce a scalar mean across ranks (no-op when not DDP).

        ``value`` is a local mean; ``count`` is the number of local samples
        (or batches) that went into it.
        """
        if not is_distributed_runtime():
            return 0.0 if count <= 0 else float(value)
        total = torch.tensor(
            [float(value) * float(max(count, 0.0)), float(max(count, 0.0))],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        if float(total[1].item()) <= 0:
            return 0.0
        return float(total[0].item() / total[1].item())

    def _gather_prediction_dicts(self, y_preds, y_truths, indices):
        gathered = [None] * dist.get_world_size() if self._is_rank0 else None
        dist.gather_object((dict(y_preds), dict(y_truths), indices), gathered, dst=0)
        if gathered is None:
            return {}, {}
        return _merge_sharded_prediction_dicts(*zip(*gathered))

    def _summarize_monitor(self) -> None:
        """Log Monitor stats over the full validation set (gather under DDP)."""
        if self.monitor is None:
            return
        if is_distributed_runtime():
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

    def save_state_dict(self, model, optimizer, scheduler, dump_dir, ema=None, suffix="last",
                        model_rank=None, epoch=None, best_score=None, best_epoch=None, **training_state):
        from ..models.compat import CHECKPOINT_FORMAT, FORMAT_KEY

        def write():
            info = {
                FORMAT_KEY: CHECKPOINT_FORMAT, "model_state_dict": _unwrap_ddp(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                **training_state,
            }
            if best_score is not None:
                info["best_score"] = float(best_score)
            for key, value in (("epoch", epoch), ("best_epoch", best_epoch)):
                if value is not None:
                    info[key] = value
            if ema is not None:
                # torch-ema retains the temporary evaluation copy after restore().
                info["ema_state_dict"] = {**ema.state_dict(), "collected_params": None}
            os.makedirs(dump_dir, exist_ok=True)
            filename = f"model{'' if model_rank is None else model_rank}_{suffix}.pth"
            with atomic_output(os.path.join(dump_dir, filename)) as temporary:
                torch.save(info, temporary)

        run_rank0_exclusive(write, self._launch, sync_dir=dump_dir, name="checkpoint")

    def _fit_lightning(self, model, pretrain_path, train_dataset, valid_dataset, test_dataset,
                       loss_terms, dump_dir, transform):
        world = self.lightning_trainer.num_devices * self.lightning_trainer.num_nodes
        # Lightning's DistributedSampler pads training indices before local drop_last.
        steps = ((len(train_dataset) + world - 1) // world) // self.batch_size
        _require_nonempty_train_loader(steps, len(train_dataset), self.batch_size, world)
        num_steps = steps * self.max_epochs
        model = model.to(dtype=self.dtype)
        if self.reset_parameters:
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        initial_ema = None
        if pretrain_path is not None and self.resume <= 1:
            if self.use_ema and self.resume == 1:
                initial_ema = ExponentialMovingAverage(model.parameters(), self.ema_decay, self.ema_use_num_updates)
            loaded = _load_state_dict(model, torch.device("cpu"), pretrain_path, ema=initial_ema,
                                      inference=self.resume == 0, strict=self.resume != 0)
            if self.freeze_pretrain_weights:
                for name, parameter in model.named_parameters():
                    if name in loaded["model_state_dict"]:
                        parameter.requires_grad_(False)
            del loaded
        optimizer = get_optimizer(self.optimizer_name, model, self.optimizer_hyper_params)
        scheduler = get_scheduler(self.schedule, optimizer, int(num_steps * self.warmup_ratio), num_steps)
        logger.info("Using Lightning Trainer")
        from .lightning_utils import LightningModel, evaluation_loader
        lightning_model = LightningModel(
            model, loss_terms, dump_dir,
            optimizer, scheduler,
            monitor=self.monitor,
            transform=transform,
            metrics=self.metrics,
            use_ema=self.use_ema,
            ema_decay=self.ema_decay,
            ema_use_num_updates=self.ema_use_num_updates,
            dump_interval=self.dump_interval,
            generator_config=self.generator_config,
        )
        if initial_ema is not None:
            lightning_model.initial_ema_state = initial_ema.state_dict()
            del initial_ema
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch(generator_training=True),
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers
        )
        if valid_dataset is not None:
            valid_dataloader = evaluation_loader(
                dataset=valid_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                collate_fn=self._collate_batch(generator_training=False),
                pin_memory=True,
                num_workers=self.num_workers
            )
        else:
            valid_dataloader = None
        if self.resume > 1 and pretrain_path is not None:
            # if the world rank is 0, modify the state dict
            run_rank0_exclusive(
                lambda: _modify_lightning_state_dict(pretrain_path, self.patience),
                sync_dir=dump_dir, name="lightning_resume",
            )
            logger.info(f"Resuming from {pretrain_path}...")
        self.lightning_trainer.fit(lightning_model, train_dataloader, valid_dataloader, ckpt_path=pretrain_path if self.resume > 1 else None)

        if test_dataset is not None:
            test_dataloader = evaluation_loader(
                dataset=test_dataset,
                batch_size=self.inference_batch_size,
                shuffle=False,
                collate_fn=self._collate_batch(generator_training=False),
                pin_memory=True,
                num_workers=self.num_workers
            )
            self.lightning_trainer.test(lightning_model, test_dataloader)
            return lightning_model.test_result or {
                "y_pred": None,
                "y_truth": None,
                "metric_score": None,
            }
        return {"y_pred": None, "y_truth": None, "metric_score": None}

    def fit_predict(self, 
        model: Module, pretrain_path: Optional[str],
        train_dataset: Dataset, valid_dataset: Optional[Dataset], 
        loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, 
        test_dataset: Optional[Dataset]=None, model_rank: Optional[int]=None, max_epoch_per_iter: int=-1,
        meta_state_dict: Optional[Dict]=None, refresh_patience: bool=False, refresh_best_score: bool=False
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
        from ..models.loss import training_loss

        world = (self.lightning_trainer.num_devices * self.lightning_trainer.num_nodes
                 if self.lightning else self._launch.world_size)
        if world > 1 and (self.active_learning or self.committee_size > 1):
            raise ValueError("Active learning and committee training require a single process")
        if meta_state_dict is None:
            meta_state_dict = {}
        if self.refresh_best_score is not None:
            refresh_best_score = self.refresh_best_score
        if self.refresh_patience is not None:
            refresh_patience = self.refresh_patience
        self._set_seed(self.seed + (model_rank if model_rank is not None else 0))
        if self.lightning:
            return self._fit_lightning(model, pretrain_path, train_dataset, valid_dataset, test_dataset,
                                       loss_terms, dump_dir, transform)
        with self._distributed_session(), contextlib.ExitStack() as resources:
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
                collate_fn=self._collate_batch(generator_training=True),
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

            model = model.to(device=self.device, dtype=self.dtype)
            optimizer = get_optimizer(self.optimizer_name, model, self.optimizer_hyper_params)
            scheduler = get_scheduler(self.schedule, optimizer, num_warmup_steps, num_training_steps)

            if self.reset_parameters:
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        logger.info(f"Resetting parameters for {m.__class__.__name__}")
                        m.reset_parameters()

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
                if ema is not None:
                    ema = ExponentialMovingAverage(model.parameters(), self.ema_decay, self.ema_use_num_updates)

            if self.freeze_pretrain_weights:
                for name, param in model.named_parameters():
                    if name in other_info.get("model_state_dict", dict()):
                        param.requires_grad = False
                        logger.info(f"Freezing pretrained weights for {name}")
            other_info.pop("model_state_dict", None)

            if self._use_ddp:
                ddp_device_ids = [self.device.index] if self.device.type == "cuda" else None
                model = DistributedDataParallel(
                    model,
                    device_ids=ddp_device_ids,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    find_unused_parameters=self.find_unused_parameters,
                )
                if ema is not None:
                    # DDP synchronizes model parameters on construction. Match its initial EMA on every rank.
                    for shadow in ema.shadow_params:
                        dist.broadcast(shadow, src=0)

            if self.resume > 1 and "best_epoch" in other_info and "epoch" in other_info:
                wait = other_info.get("wait", other_info["epoch"] - other_info["best_epoch"])
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
            if tb_writer is not None:
                resources.callback(tb_writer.close)
            global_step = other_info.get("global_step", start_epoch * len(train_dataloader)) if self.resume > 1 else 0
            if self._use_ddp:
                self._set_seed(self.seed + (model_rank or 0) + self._launch.global_rank)
            if self.resume > 1 and pretrain_path is not None and "rank_states" not in other_info:
                logger.info("This checkpoint predates per-rank random states. Resuming with seeded streams.")
            if self.resume > 1 and "rank_states" in other_info:
                rank_states = other_info["rank_states"]
                if len(rank_states) == self._launch.world_size:
                    _restore_rng(rank_states[self._launch.global_rank], self.device)
                else:
                    logger.warning("Checkpoint world size changed; starting new per-rank random streams")
            if train_sampler is not None and self.resume > 1 and "sampler_seed" in other_info:
                train_sampler.seed = other_info["sampler_seed"]

            epoch = start_epoch
            epoch_in_iter = meta_state_dict.get("epoch_in_iter", 0)
            if start_epoch > 0 and self._is_rank0:
                if epoch_in_iter > 0:
                    logger.info(f"Resuming from epoch {start_epoch + 1}, epoch {epoch_in_iter + 1} in active learning iteration")
                else:
                    logger.info(f"Resuming from epoch {start_epoch + 1}...")
            if wait > 0 and wait >= self.patience:
                max_epochs = start_epoch
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
                accumulated_loss = 0.0
                
                for i, batch in enumerate(train_dataloader):
                    net_input, net_target = self.to_device(batch)
                    
                    with torch.set_grad_enabled(True):
                        output = model(net_input)
                        loss = training_loss(loss_terms, output, net_target)
                    accumulated_loss += loss.detach().item()

                    batch_bar.set_postfix(
                        Epoch="Epoch {}/{}".format(epoch+1, max_epochs),
                        loss="{:.04f}".format(accumulated_loss / (i + 1)),
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
                        # Both scalars describe the just-completed global optimizer step.
                        tb_writer.add_scalar("train_loss", float(loss.detach().item()), global_step)
                        tb_writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), global_step)

                    if self.use_ema:
                        ema.update()
                    
                    scheduler.step()
                    batch_bar.update()

                batch_bar.close()
                total_trn_loss_epoch = self._reduce_mean_scalar(
                    accumulated_loss / len(train_dataloader), count=len(train_dataloader)
                )
                message = f'Epoch [{epoch + 1}/{max_epochs}]' + (f' ({epoch_in_iter + 1}/{max_epoch_per_iter})' if max_epoch_per_iter > 0 else '') + f', train_loss: {total_trn_loss_epoch:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'

                if self.use_ema:
                    cm = ema.average_parameters()
                else:
                    cm = contextlib.nullcontext()

                if valid_dataset is not None:
                    if self._use_ddp:
                        # Validation bypasses DDP forward so empty shards never enter a collective.
                        # Match rank zero's running buffers after the final training batch.
                        for buffer in _unwrap_ddp(model).buffers():
                            dist.broadcast(buffer, src=0)
                    with cm:
                        predict_result = self._predict(
                            model=model, 
                            dataset=valid_dataset, 
                            loss_terms=loss_terms, 
                            dump_dir=dump_dir, 
                            transform=transform, 
                            epoch=epoch, 
                            load_model=False, gather_outputs=False,
                        )
                    val_loss = predict_result["val_loss"]
                    metric_score = predict_result["metric_score"]
                    total_val_loss = float(np.mean(val_loss)) if len(val_loss) else 0.0
                    _score = metric_score["_judge_score"]
                    _metric = str(self.metrics)
                    training_state = self._epoch_state(global_step, train_sampler)
                    save_handle = partial(self.save_state_dict, model=model, optimizer=optimizer, scheduler=scheduler, dump_dir=dump_dir, ema=ema, suffix="best", model_rank=model_rank, wait=0, **training_state)
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
                    training_state = self._epoch_state(global_step, train_sampler)
                    if tb_writer is not None:
                        tb_writer.add_scalar("train_loss_epoch", total_trn_loss_epoch, epoch + 1)

                epoch_in_iter += 1
                end_time = time.time()
                message += f', {(end_time - start_time):.1f}s'
                if self._is_rank0:
                    logger.info(message)
                self.save_state_dict(model, optimizer, scheduler, dump_dir, ema, "last", model_rank, epoch=epoch, best_score=best_score, best_epoch=best_epoch, wait=wait, **training_state)
                if self.dump_interval > 0 and (epoch + 1) % self.dump_interval == 0:
                    self.save_state_dict(model, optimizer, scheduler, dump_dir, ema, f"epoch={epoch}", model_rank, epoch=epoch, best_score=best_score, best_epoch=best_epoch, wait=wait, **training_state)
                run_rank0_exclusive(
                    lambda: meta_state_dict.update({"epoch_in_iter": epoch_in_iter}),
                    self._launch, sync_dir=dump_dir, name="training_progress",
                )
                if is_early_stop:
                    break

            run_rank0_exclusive(
                lambda: meta_state_dict.update({"model_rank": (model_rank + 1) if model_rank is not None else 0,
                                                "epoch_in_iter": 0}),
                self._launch, sync_dir=dump_dir, name="training_progress",
            )
            if test_dataset is not None:
                if self.use_ema:
                    cm = ema.average_parameters()
                else:
                    cm = contextlib.nullcontext()
                with cm:
                    predict_result = self._predict(
                        model=model, gather_outputs=True,
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

    def _epoch_state(self, global_step, sampler):
        local_state = _capture_rng(self.device)
        states = [local_state]
        if self._use_ddp:
            states = [None] * self._launch.world_size if self._is_rank0 else None
            dist.gather_object(local_state, states, dst=0)
        result = {"global_step": global_step, "rank_states": states}
        if sampler is not None:
            result["sampler_seed"] = sampler.seed
        return result

    def _early_stop_choice(self, wait, min_loss, metric_score, save_handle, patience, epoch):
        return self.metrics._early_stop_choice(wait, min_loss, metric_score, save_handle, patience, epoch)
    
    def predict(self, model: Module, dataset: Dataset, loss_terms: Iterable[Callable], dump_dir: str, transform: Transform, epoch: int=1, load_model: bool=False, model_rank: Optional[str]=None, test_mode: bool=False) -> Dict[Literal["y_pred", "y_truth", "val_loss", "metric_score"], Any]:
        self._set_seed(self.seed)
        if self.lightning:
            self.device = native_device_for_launch(self._launch, self.lightning_cuda and torch.cuda.is_available())
        with self._distributed_session():
            return self._predict(model, dataset, loss_terms, dump_dir, transform, epoch,
                                 load_model, model_rank, test_mode, gather_outputs=True)

    def _predict(self, model, dataset, loss_terms, dump_dir, transform, epoch=1,
                 load_model=False, model_rank=None, test_mode=False, *, gather_outputs):
        from ..models.loss import EvaluationLoss

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
        shard_indices: List[int] = []
        if use_ddp:
            shard_indices = _strided_shard_indices(
                dist.get_rank(), dist.get_world_size(), len(dataset)
            )
            eval_dataset = Subset(dataset, shard_indices)
        else:
            eval_dataset = dataset
        pin_memory = use_ddp and self.device.type == "cuda"
        dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            collate_fn=self._collate_batch(generator_training=False),
            **self._dataloader_kwargs(
                self.num_workers,
                pin_memory=pin_memory,
                distributed=use_ddp,
            ),
        )
        raw_model = raw_model.eval()
        raw_model.test_mode = test_mode
        batch_bar = self._progress_bar(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='val', ncols=5)
        eval_loss = EvaluationLoss(loss_terms) if not load_model else None
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
                with torch.enable_grad():
                    output = raw_model(net_input)
            # Get model outputs
            if self.monitor is not None:
                self.monitor.collect(output)
            loss = 0
            with torch.no_grad():
                if not load_model:
                    loss = eval_loss.update(output, net_target)
                    val_loss.append(float(loss))
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

        if transform is not None and y_preds:
            transform.inverse_transform(y_preds)
            transform.inverse_transform(y_truths)
        metric_score = self.metrics.distributed_metric(y_truths, y_preds)
        if eval_loss is not None:
            val_loss = [eval_loss.compute(self.device)]
        if gather_outputs and use_ddp:
            y_preds, y_truths = self._gather_prediction_dicts(y_preds, y_truths, shard_indices)
        if load_model and "_judge_score" in metric_score:
            metric_score.pop("_judge_score")
        batch_bar.close()
        raw_model.test_mode = False
        return {"y_pred": y_preds, "y_truth": y_truths, "val_loss": val_loss, "metric_score": metric_score}
