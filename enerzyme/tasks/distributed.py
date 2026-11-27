"""Launch-environment detection and rank primitives for native torch DDP.

Distributed training must be started by an external launcher (``srun`` or
``torchrun``) that creates one process per GPU. Enerzyme wraps
``DistributedDataParallel``; it does not spawn processes.
These helpers work from environment variables **before**
``torch.distributed`` is initialized.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Literal, Mapping, MutableMapping, Optional, Union

LaunchMode = Literal["single", "slurm_srun", "torchrun", "slurm_unlaunched"]


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


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    """Leading integer from a Slurm env value (``4``, ``4(x2)``, ``gpu:4``)."""
    if value is None:
        return None
    digits: list[str] = []
    for ch in str(value).strip():
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return None
    parsed = int("".join(digits))
    return parsed if parsed > 0 else None


def _slurm_local_world_size(environ: Mapping[str, str]) -> int:
    for key in (
        "SLURM_STEP_NUM_TASKS_PER_NODE",
        "SLURM_STEP_TASKS_PER_NODE",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_TASKS_PER_NODE",
    ):
        parsed = _parse_positive_int(environ.get(key))
        if parsed:
            return parsed
    return 0


def _slurm_world_size(
    environ: Mapping[str, str],
    num_nodes: int,
    local_world_size: int,
    *,
    srun: bool,
) -> tuple[int, int]:
    """Task counts for a SLURM allocation or srun step.

    ``SLURM_NTASKS`` is often unset when the script only specifies
    ``--ntasks-per-node``. Prefer step-level counts over job-level
    ``SLURM_NTASKS`` so a smaller ``srun`` inside a larger allocation
    does not wait for missing ranks, then ``ntasks_per_node * nnodes``.
    For ``srun``, also lower-bound by ``SLURM_PROCID + 1`` so a
    multi-task step cannot look like world_size=1.
    """
    world = 0
    for key in ("SLURM_STEP_NUM_TASKS", "SLURM_NTASKS", "SLURM_NPROCS"):
        parsed = _parse_positive_int(environ.get(key))
        if parsed:
            world = parsed
            break
    if world <= 0 and local_world_size > 0:
        world = local_world_size * max(1, num_nodes)
    world = max(1, world)
    local = local_world_size
    if srun:
        procid = _env_int(environ, "SLURM_PROCID", 0)
        localid = _env_int(environ, "SLURM_LOCALID", 0)
        world = max(world, procid + 1)
        if local <= 0:
            local = max(
                1,
                localid + 1,
                (world + max(1, num_nodes) - 1) // max(1, num_nodes),
            )
    if local <= 0:
        local = max(1, world // max(1, num_nodes))
    return world, max(1, min(local, world))


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
    return environ.get("SLURM_STEP_ID") not in (None, "", "batch", "extern") and "SLURM_PROCID" in environ


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
        count = str(value).rsplit(":", 1)[-1]
        if count.isdigit():
            return max(1, int(count))
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

    if "WORLD_SIZE" in env and not _has_torchrun_env(env) and not _is_srun_step(env):
        if int(env["WORLD_SIZE"]) > 1 and not is_lightning_subprocess(env):
            raise ValueError("Distributed launch requires RANK, LOCAL_RANK and WORLD_SIZE")

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
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=max(1, num_nodes),
            allocated_gpus=allocated_gpus,
        )

    if _in_slurm_job(env):
        num_nodes = max(1, _env_int(env, "SLURM_STEP_NUM_NODES", _env_int(env, "SLURM_NNODES", 1)))
        local_world_size = _slurm_local_world_size(env)
        srun = _is_srun_step(env)
        world_size, local_world_size = _slurm_world_size(
            env, num_nodes, local_world_size, srun=srun
        )

        if srun:
            return LaunchEnv(
                mode="slurm_srun",
                global_rank=_env_int(env, "SLURM_PROCID", 0),
                local_rank=_env_int(env, "SLURM_LOCALID", 0),
                world_size=world_size,
                local_world_size=local_world_size,
                num_nodes=num_nodes,
                allocated_gpus=allocated_gpus,
            )

        return LaunchEnv(
            mode="slurm_unlaunched",
            global_rank=0,
            local_rank=0,
            world_size=world_size,
            local_world_size=local_world_size,
            num_nodes=num_nodes,
            allocated_gpus=allocated_gpus,
        )

    return LaunchEnv(mode="single", allocated_gpus=allocated_gpus)


def is_global_zero(env: Optional[LaunchEnv] = None) -> bool:
    if env is not None:
        return env.is_global_zero()
    dist = sys.modules.get("torch.distributed")
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return not is_lightning_subprocess() and detect_launch_env().is_global_zero()


def is_lightning_subprocess(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Identify a child re-executing the CLI after Lightning launched locally."""
    env = os.environ if environ is None else environ
    return (int(env.get("LOCAL_RANK", 0)) > 0 and "RANK" not in env
            and "MASTER_ADDR" in env and "MASTER_PORT" in env
            and not _is_srun_step(env))


def local_rank(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).local_rank


def world_size(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).world_size


def global_rank(env: Optional[LaunchEnv] = None) -> int:
    return (env or detect_launch_env()).global_rank


def _read(path):
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _publish(path, text):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex)
    tmp.write_text(text)
    tmp.replace(path)


def _wait(check, timeout, interval, name):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        value = check()
        if value:
            return value
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for distributed {name}")
        time.sleep(interval)


_round_numbers = {}


def _file_rank0(fn, launch, directory, name, handshake, work, interval):
    """Agree on a fresh generation before publishing rank zero's result.

    Each participant supplies a nonce. A restarted job cannot accept an old
    completion record, even if its scheduler or rendezvous identifier is reused.
    Distinct invocations have distinct paths so a fast rank can enter the next
    call while another rank is still leaving the previous one.
    """
    job = (os.getenv("TORCHELASTIC_RUN_ID"), os.getenv("TORCHELASTIC_RESTART_COUNT"),
           os.getenv("SLURM_JOB_ID"), os.getenv("SLURM_STEP_ID"))
    key = (str(directory.resolve()), name, job, launch.world_size, launch.global_rank)
    sequence = _round_numbers.get(key, 0)
    _round_numbers[key] = sequence + 1
    digest = hashlib.sha256(repr((name, job, launch.world_size, sequence)).encode()).hexdigest()[:24]
    shared = directory / (".enerzyme-sync-" + digest)
    shared.mkdir(parents=True, exist_ok=True)
    generation = shared / "generation"
    started = shared / "started"
    nonce = uuid.uuid4().hex
    peers = [shared / f"rank-{i}" for i in range(1, launch.world_size)]
    if launch.is_global_zero():
        _publish(generation, nonce)

        def arrivals():
            values = [_read(peer) for peer in peers]
            if all(value and value.startswith(nonce + ":") for value in values):
                return [value.split(":")[1] for value in values]

        tokens = _wait(arrivals, handshake, interval, name)
        _publish(started, json.dumps(tokens))
        error = None
        try:
            fn()
        except BaseException as exc:
            error = exc
        message = json.dumps(None if error is None else f"{type(error).__name__}: {error}")
        for token in tokens:
            _publish(shared / token, message)
        # Wait for readers before removing this generation's files.
        _wait(lambda: all(not (shared / token).exists() for token in tokens),
              handshake, interval, name)
        for peer in peers:
            peer.unlink()
        started.unlink()
        generation.unlink()
        # Files from an interrupted earlier attempt may still exist.
        try:
            shared.rmdir()
        except OSError:
            pass
        if error is not None:
            raise error
    else:
        peer = peers[launch.global_rank - 1]
        result = shared / nonce
        seen = None

        def joined():
            nonlocal seen
            payload = _read(result)
            if payload is not None:
                return payload
            token = _read(generation)
            if token and token != seen:
                _publish(peer, token + ":" + nonce)
                seen = token
            accepted = _read(started)
            return accepted and nonce in json.loads(accepted)

        _wait(joined, handshake, interval, name)
        payload = _wait(lambda: _read(result), work, interval, name)
        result.unlink()
        error = json.loads(payload)
        if error is not None:
            raise RuntimeError(f"rank 0 failed during {name!r}: {error}")


def run_rank0_exclusive(fn, env=None, *, sync_dir=None, name="default",
                        handshake_timeout_seconds=1800.0, work_timeout_seconds=None,
                        poll_interval=0.1):
    """Run shared work once and propagate its success or exception to all ranks."""
    dist = sys.modules.get("torch.distributed")
    active = dist is not None and dist.is_available() and dist.is_initialized()
    if is_lightning_subprocess() and not active and env is None:
        return
    if active:
        error = None
        if dist.get_rank() == 0:
            try:
                fn()
            except BaseException as exc:
                error = exc
        message = [None if error is None else f"{type(error).__name__}: {error}"]
        dist.broadcast_object_list(message, src=0)
        if error is not None:
            raise error
        if message[0] is not None:
            raise RuntimeError(f"rank 0 failed during {name!r}: {message[0]}")
        return
    launch = env or detect_launch_env()
    if launch.world_size <= 1 or launch.mode == "slurm_unlaunched":
        fn()
        return
    if sync_dir is None:
        raise ValueError("Distributed data preparation needs a shared sync_dir")
    _file_rank0(fn, launch, Path(sync_dir), name, handshake_timeout_seconds,
                work_timeout_seconds, poll_interval)


def barrier(env=None, *, sync_dir=None, name="default", timeout_seconds=1800.0, poll_interval=0.1):
    run_rank0_exclusive(lambda: None, env, sync_dir=sync_dir, name=name,
                        handshake_timeout_seconds=timeout_seconds, work_timeout_seconds=timeout_seconds,
                        poll_interval=poll_interval)


def validate_distributed_launch(env: Optional[LaunchEnv] = None) -> None:
    """Fail fast when a multi-task / multi-GPU SLURM job lacks srun/torchrun.

    Single-task, single-GPU allocations (``world_size == 1`` and one allocated
    GPU) may run the native loop without ``srun``. Multi-task
    ``slurm_unlaunched``, or ``salloc -G N`` with N>1 and no launcher, would
    hang or silently train on one GPU while other allocated GPUs sit idle.
    """
    launch = env or detect_launch_env()
    if not (0 <= launch.global_rank < launch.world_size and
            0 <= launch.local_rank < launch.local_world_size <= launch.world_size):
        raise ValueError(f"Inconsistent ranks and world size: {launch}")
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
    nodes = re.sub(r"\[([^]\-,]+)[^]]*\]", r"\1", nodelist)
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
    if _env_int(env, "SLURM_STEP_NUM_NODES", _env_int(env, "SLURM_NNODES", 0)) == 1:
        return "127.0.0.1"
    nodelist = env.get("SLURM_STEP_NODELIST") or env.get("SLURM_NODELIST") or env.get("SLURM_JOB_NODELIST")
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

    Prefer ``LaunchEnv.world_size`` (``WORLD_SIZE`` / SLURM task counts).
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

    Preserve explicit rendezvous settings and the launcher’s GPU visibility.
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


# Auto workers overlap HDF5/collate with GPU compute. Past a handful, extra
# processes mostly cost RAM and forkserver startup; HPC GPU nodes often expose
# 32–128 CPUs that are not meant to each host a DataLoader worker.
_AUTO_NUM_WORKERS_CAP = 8


def infer_num_workers(
    requested: int = -1,
    *,
    env: Optional[LaunchEnv] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Resolve DataLoader ``num_workers``.

    Auto (``requested < 0``) leaves one CPU for the training process when
    possible, then clamps to ``[1, _AUTO_NUM_WORKERS_CAP]``. Prefer
    ``SLURM_CPUS_PER_TASK`` as the per-rank budget; otherwise share
    ``os.cpu_count()`` across local ranks. Never uses ``SLURM_NTASKS``.

    ``requested > 0`` is returned unchanged. ``requested == 0`` is an explicit
    in-process loader (no worker subprocesses). Auto never returns 0.
    """
    if requested == 0:
        return 0
    if requested > 0:
        return int(requested)

    launch = env or detect_launch_env(environ)
    env_map: Mapping[str, str] = os.environ if environ is None else environ

    cpus_per_task = env_map.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task is not None and cpus_per_task != "":
        budget = max(0, int(cpus_per_task) - 1)
    else:
        cpu_count = os.cpu_count() or 1
        per_rank = max(1, cpu_count // max(1, launch.local_world_size))
        budget = max(0, per_rank - 1)
    return max(1, min(_AUTO_NUM_WORKERS_CAP, budget))


def native_device_for_launch(launch: LaunchEnv, cuda: bool):
    """Select the tensor device without rewriting visibility or NCCL's topology."""
    import torch
    if not cuda:
        return torch.device("cpu")
    count = torch.cuda.device_count()
    if not count:
        if launch.world_size > 1:
            raise RuntimeError("CUDA training requested but this rank has no visible GPU")
        return torch.device("cpu")
    index = 0 if count == 1 else launch.local_rank
    if index >= count or (launch.mode == "torchrun" and launch.local_world_size > count):
        raise RuntimeError(f"{launch.local_world_size} local workers with {count} visible GPU(s); "
                           "launch one process per GPU")
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


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
        dist.destroy_process_group()


def is_distributed_runtime() -> bool:
    """True when ``torch.distributed`` is initialized with world size > 1."""
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
