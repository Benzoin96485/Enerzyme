"""Launch-environment detection and rank primitives for native torch DDP.

Distributed training must be started by an external launcher (``srun`` or
``torchrun``) that creates one process per GPU. Enerzyme wraps
``DistributedDataParallel``; it does not spawn processes.
These helpers work from environment variables **before**
``torch.distributed`` is initialized.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, Mapping, MutableMapping, Optional, Union

LaunchMode = Literal["single", "slurm_srun", "torchrun", "slurm_unlaunched"]


def prepare_nccl_for_single_visible_gpu(
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Disable NCCL P2P/SHM when this process has exactly one visible GPU.

    ``srun --gpus-per-task=1`` remaps every rank to ``cuda:0``. NCCL P2P and
    SHM then use invalid physical device ordinals (CUDA error 101). Must run
    **before** NCCL loads.
    """
    env_map: Mapping[str, str] = os.environ if environ is None else environ
    visible = env_map.get("CUDA_VISIBLE_DEVICES")
    if visible is None or str(visible).strip() == "":
        return
    ids = [part.strip() for part in str(visible).split(",") if part.strip()]
    if len(ids) != 1:
        return
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")

_TORCHRUN_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


@dataclass(frozen=True)
class LaunchEnv:
    """Snapshot of how this process was launched."""

    mode: LaunchMode
    global_rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    num_nodes: int = 1
    allocated_gpus: int = 1

    def is_global_zero(self) -> bool:
        return self.global_rank == 0


def _env_int(environ: Mapping[str, str], key: str, default: int) -> int:
    value = environ.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _has_torchrun_env(environ: Mapping[str, str]) -> bool:
    return all(key in environ for key in _TORCHRUN_KEYS)


def _in_slurm_job(environ: Mapping[str, str]) -> bool:
    return "SLURM_JOB_ID" in environ


def _is_srun_step(environ: Mapping[str, str]) -> bool:
    """True when this process is a task inside an ``srun`` job step.

    A bare ``sbatch``/``salloc`` shell typically has ``SLURM_JOB_ID`` (and often
    ``SLURM_NTASKS``) but no ``SLURM_STEP_ID``. ``srun`` creates a step and sets
    ``SLURM_STEP_ID`` / ``SLURM_PROCID`` on each task.
    """
    return "SLURM_STEP_ID" in environ and "SLURM_PROCID" in environ


def _allocated_gpu_count(environ: Mapping[str, str]) -> int:
    """GPUs assigned to this allocation / process (best-effort from env).

    Used to fail-fast when an interactive ``salloc -G N`` (N>1) has
    ``SLURM_NTASKS`` unset or 1, which would otherwise look like a single-task
    job and silently train on one GPU.
    """
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        value = environ.get(key)
        if value is None or str(value).strip() == "":
            continue
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            return max(1, int(digits))
    job_gpus = environ.get("SLURM_JOB_GPUS")
    if job_gpus and str(job_gpus).strip():
        ids = [
            part.strip()
            for part in str(job_gpus).replace(";", ",").split(",")
            if part.strip()
        ]
        if ids:
            return len(ids)
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and str(visible).strip() != "":
        ids = [part.strip() for part in str(visible).split(",") if part.strip()]
        if ids:
            return len(ids)
    return 1


def detect_launch_env(
    environ: Optional[Mapping[str, str]] = None,
) -> LaunchEnv:
    """Infer launch mode and ranks from the process environment.

    Priority: torchrun vars → SLURM srun step → SLURM job without srun → single.
    """
    env = os.environ if environ is None else environ

    allocated_gpus = _allocated_gpu_count(env)

    if _has_torchrun_env(env):
        world_size = _env_int(env, "WORLD_SIZE", 1)
        local_world_size = _env_int(env, "LOCAL_WORLD_SIZE", 0)
        if local_world_size <= 0:
            # Fall back to per-node task count under SLURM, else world_size.
            local_world_size = _env_int(
                env, "SLURM_NTASKS_PER_NODE", world_size
            )
        num_nodes = _env_int(env, "SLURM_NNODES", 0)
        if num_nodes <= 0:
            num_nodes = max(1, world_size // max(1, local_world_size))
        return LaunchEnv(
            mode="torchrun",
            global_rank=_env_int(env, "RANK", 0),
            local_rank=_env_int(env, "LOCAL_RANK", 0),
            world_size=max(1, world_size),
            local_world_size=max(1, local_world_size),
            num_nodes=max(1, num_nodes),
            allocated_gpus=allocated_gpus,
        )

    if _in_slurm_job(env):
        num_nodes = max(1, _env_int(env, "SLURM_NNODES", 1))
        world_size = max(1, _env_int(env, "SLURM_NTASKS", 1))
        local_world_size = _env_int(env, "SLURM_NTASKS_PER_NODE", 0)
        if local_world_size <= 0:
            local_world_size = max(1, world_size // num_nodes)

        if _is_srun_step(env):
            return LaunchEnv(
                mode="slurm_srun",
                global_rank=_env_int(env, "SLURM_PROCID", 0),
                local_rank=_env_int(env, "SLURM_LOCALID", 0),
                world_size=world_size,
                local_world_size=max(1, local_world_size),
                num_nodes=num_nodes,
                allocated_gpus=allocated_gpus,
            )

        return LaunchEnv(
            mode="slurm_unlaunched",
            global_rank=0,
            local_rank=0,
            world_size=world_size,
            local_world_size=max(1, local_world_size),
            num_nodes=num_nodes,
            allocated_gpus=allocated_gpus,
        )

    return LaunchEnv(mode="single", allocated_gpus=allocated_gpus)


def is_global_zero(env: Optional[LaunchEnv] = None) -> bool:
    return (env or detect_launch_env()).is_global_zero()


def local_rank(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).local_rank


def world_size(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).world_size


def global_rank(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).global_rank


def barrier(
    env: Optional[LaunchEnv] = None,
    *,
    sync_dir: Optional[str] = None,
    name: str = "default",
    timeout_seconds: float = 1800.0,
    poll_interval: float = 0.1,
) -> None:
    """Synchronize ranks.

    Uses ``torch.distributed.barrier`` when the process group is initialized.
    Before init (or when dist is unavailable), falls back to a shared-filesystem
    flag under ``sync_dir`` when ``world_size > 1``. Single-process is a no-op.

    ``name`` distinguishes successive file barriers that share the same
    ``sync_dir`` (e.g. ``datahub`` then ``splitter`` then ``config``).
    """
    launch = env or detect_launch_env()
    if launch.world_size <= 1 or launch.mode in ("single", "slurm_unlaunched"):
        return

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            return
    except ImportError:
        pass

    if sync_dir is None:
        raise RuntimeError(
            "barrier() before torch.distributed init requires sync_dir on a "
            "shared filesystem (e.g. the training output directory)."
        )

    sync_path = Path(sync_dir)
    sync_path.mkdir(parents=True, exist_ok=True)
    # One flag file per named collective so callers can reuse sync_dir.
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    prefix = f".enerzyme_barrier_{launch.world_size}_{safe_name}"
    flag = sync_path / f"{prefix}.done"
    arrived = sync_path / f"{prefix}.r{launch.global_rank}"
    arrived.touch()

    if launch.is_global_zero():
        # Rank 0 waits until every rank has checked in, then writes the done flag.
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            present = list(sync_path.glob(f"{prefix}.r*"))
            if len(present) >= launch.world_size:
                flag.touch()
                for path in present:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                return
            time.sleep(poll_interval)
        raise TimeoutError(
            f"barrier timed out after {timeout_seconds}s waiting for "
            f"{launch.world_size} ranks under {sync_dir} (name={name!r})"
        )

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if flag.exists():
            return
        time.sleep(poll_interval)
    raise TimeoutError(
        f"barrier timed out after {timeout_seconds}s waiting for rank0 flag "
        f"under {sync_dir} (name={name!r})"
    )


def validate_distributed_launch(env: Optional[LaunchEnv] = None) -> None:
    """Fail fast when a multi-task / multi-GPU SLURM job lacks srun/torchrun.

    Single-task, single-GPU allocations (``world_size == 1`` and one allocated
    GPU) may run the native loop without ``srun``. Multi-task
    ``slurm_unlaunched``, or ``salloc -G N`` with N>1 and no launcher, would
    hang or silently train on one GPU while other allocated GPUs sit idle.
    """
    launch = env or detect_launch_env()
    if launch.mode != "slurm_unlaunched":
        return
    if launch.world_size <= 1 and launch.allocated_gpus <= 1:
        return

    raise RuntimeError(
        "Distributed training detected a SLURM allocation without an external "
        "launcher (srun/torchrun). Enerzyme requires one process per GPU; it "
        "will not spawn workers. Starting `enerzyme train` directly inside "
        "sbatch/salloc with multiple tasks hangs or uses only one GPU.\n\n"
        "Launch with srun (fill in site resource options yourself):\n"
        "  srun --ntasks-per-node=<GPUS_PER_NODE> --gpus-per-task=1 \\\n"
        "      enerzyme train -c train.yaml -o <OUTPUT_DIR>\n\n"
        "Or with torchrun on a single node:\n"
        "  torchrun --nproc_per_node=<GPUS_PER_NODE> \\\n"
        "      $(which enerzyme) train -c train.yaml -o <OUTPUT_DIR>\n"
    )


_DEFAULT_MASTER_PORT = "29500"


def _first_slurm_hostname(nodelist: str) -> str:
    """Return the first host from a SLURM ``SLURM_NODELIST`` string.

    Matches Lightning's ``SLURMEnvironment.resolve_root_node_address`` so
    compact ranges such as ``nid[0001-0004]`` become ``nid0001``.
    """
    nodes = re.sub(r"\[(.*?)[,-].*\]", r"\1", nodelist)
    nodes = re.sub(r"\[(.*?)\]", r"\1", nodes)
    return nodes.split(" ")[0].split(",")[0]


def resolve_master_addr(environ: Optional[Mapping[str, str]] = None) -> str:
    """Rendezvous address for externally launched DDP.

    Preference: existing ``MASTER_ADDR`` → ``127.0.0.1`` on single-node
    SLURM → first host in ``SLURM_NODELIST`` / ``SLURM_JOB_NODELIST`` →
    ``SLURM_LAUNCH_NODE_IPADDR`` (last resort; often the submit/login host
    and unreachable from compute under interactive ``salloc``) → localhost.

    Preferring ``SLURM_LAUNCH_NODE_IPADDR`` first breaks Perlmutter-style
    interactive jobs: ranks launched by ``srun`` on the compute node try to
    TCPStore-connect to the login IP and hang until timeout.
    """
    env = os.environ if environ is None else environ
    existing = env.get("MASTER_ADDR")
    if existing:
        return existing
    # Single-node: loopback is always reachable between srun tasks.
    if _env_int(env, "SLURM_NNODES", 0) == 1:
        return "127.0.0.1"
    nodelist = env.get("SLURM_NODELIST") or env.get("SLURM_JOB_NODELIST")
    if nodelist:
        return _first_slurm_hostname(nodelist)
    launch_ip = env.get("SLURM_LAUNCH_NODE_IPADDR")
    if launch_ip:
        return launch_ip
    return "127.0.0.1"


def resolve_master_port(environ: Optional[Mapping[str, str]] = None) -> str:
    """Shared ``MASTER_PORT`` (all ranks must agree).

    Preference: existing ``MASTER_PORT`` → SLURM job-id offset (same formula
    as Lightning's SLURM plugin) → torchrun default 29500.
    """
    env = os.environ if environ is None else environ
    existing = env.get("MASTER_PORT")
    if existing:
        return existing
    job_id = env.get("SLURM_JOB_ID")
    if job_id:
        digits = "".join(ch for ch in job_id if ch.isdigit())
        if digits:
            return str(int(digits[-4:]) + 15000)
    return _DEFAULT_MASTER_PORT


def resolve_world_size(
    num_nodes: Union[int, str, None],
    env: LaunchEnv,
) -> int:
    """DDP world size for external launch.

    Prefer ``LaunchEnv.world_size`` (``WORLD_SIZE`` / ``SLURM_NTASKS``).
    Optional ``num_nodes`` only fills a missing world size as
    ``num_nodes * local_world_size``.
    """
    if env.world_size > 1:
        return max(1, env.world_size)
    if num_nodes is not None and num_nodes != "auto":
        return max(1, int(num_nodes) * max(1, env.local_world_size))
    return max(1, env.world_size)


def export_torchrun_env(
    env: Optional[LaunchEnv] = None,
    *,
    num_nodes: Union[int, str, None] = "auto",
    environ: Optional[MutableMapping[str, str]] = None,
) -> LaunchEnv:
    """Translate SLURM/launch ranks into torchrun-style env vars.

    Fills missing keys only (``setdefault``): ``RANK``, ``WORLD_SIZE``,
    ``LOCAL_WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``. Callers then
    bind one visible GPU and set ``LOCAL_RANK=0`` / ``NODE_RANK=<global>``.
    """
    launch = env or detect_launch_env(environ)
    env_map: MutableMapping[str, str] = os.environ if environ is None else environ
    world = resolve_world_size(num_nodes, launch)
    env_map.setdefault("RANK", str(launch.global_rank))
    env_map.setdefault("WORLD_SIZE", str(world))
    env_map.setdefault("LOCAL_WORLD_SIZE", str(max(1, launch.local_world_size)))
    env_map.setdefault("MASTER_ADDR", resolve_master_addr(env_map))
    env_map.setdefault("MASTER_PORT", resolve_master_port(env_map))
    return launch


def infer_num_workers(
    requested: int = -1,
    *,
    env: Optional[LaunchEnv] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Resolve DataLoader ``num_workers``.

    Prefer ``SLURM_CPUS_PER_TASK`` (minus one for the main process). Otherwise
    use ``os.cpu_count() // local_world_size``. Never uses ``SLURM_NTASKS``.

    ``requested > 0`` is returned unchanged. ``requested == 0`` is an explicit
    in-process loader (no workers). ``requested < 0`` means auto.
    """
    if requested == 0:
        return 0
    if requested > 0:
        return int(requested)

    launch = env or detect_launch_env(environ)
    env_map: Mapping[str, str] = os.environ if environ is None else environ

    cpus_per_task = env_map.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None and cpus_per_task != "":
        return max(0, int(cpus_per_task) - 1)

    cpu_count = os.cpu_count() or 1
    return max(0, cpu_count // max(1, launch.local_world_size))


def bind_single_visible_gpu(launch: LaunchEnv, *, cuda: bool = True) -> None:
    """Leave this process with exactly one visible GPU.

    ``srun --gpus-per-task=1`` already sets a single ``CUDA_VISIBLE_DEVICES``.
    ``torchrun`` usually leaves all node GPUs visible; select ``local_rank``.
    Must run **before** CUDA / NCCL initialize.
    """
    if not cuda:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and str(visible).strip() != "":
        ids = [part.strip() for part in str(visible).split(",") if part.strip()]
        if not ids:
            return
        if len(ids) == 1:
            prepare_nccl_for_single_visible_gpu()
            return
        chosen = ids[min(max(0, launch.local_rank), len(ids) - 1)]
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
        prepare_nccl_for_single_visible_gpu()
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max(0, launch.local_rank))
    prepare_nccl_for_single_visible_gpu()


def init_process_group(
    launch: Optional[LaunchEnv] = None,
    *,
    timeout_minutes: float = 30,
    backend: Optional[str] = None,
) -> bool:
    """Initialize ``torch.distributed`` for externally launched DDP.

    Returns ``True`` when a process group is (or was already) initialized.
    Single-process and ``slurm_unlaunched`` are no-ops (``False``).
    """
    launch = launch or detect_launch_env()
    if launch.world_size <= 1 or launch.mode in ("single", "slurm_unlaunched"):
        return False

    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return True

    export_torchrun_env(launch)
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=float(timeout_minutes)),
    )
    return True


def destroy_process_group() -> None:
    """Tear down the process group when it was initialized."""
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_distributed_runtime() -> bool:
    """True when ``torch.distributed`` is initialized with world size > 1."""
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
