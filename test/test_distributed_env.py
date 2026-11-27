"""Unit tests for enerzyme.tasks.distributed launch detection."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from enerzyme.tasks.distributed import (
    LaunchEnv,
    barrier,
    native_device_for_launch,
    detect_launch_env,
    export_torchrun_env,
    infer_num_workers,
    init_process_group,
    is_global_zero,
    local_rank,
    resolve_master_addr,
    resolve_master_port,
    resolve_world_size,
    run_rank0_exclusive,
    validate_distributed_launch,
    world_size,
)


def test_detect_single_process_default():
    env = detect_launch_env({})
    assert env == LaunchEnv(mode="single")
    assert env.is_global_zero()
    assert world_size(env) == 1
    assert local_rank(env) == 0


def test_detect_torchrun():
    environ = {
        "RANK": "2",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "2",
    }
    env = detect_launch_env(environ)
    assert env.mode == "torchrun"
    assert env.global_rank == 2
    assert env.local_rank == 1
    assert env.world_size == 4
    assert env.local_world_size == 2
    assert env.num_nodes == 2
    assert not is_global_zero(env)


def test_detect_slurm_srun():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "3",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS": "8",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_NNODES": "2",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_srun"
    assert env.global_rank == 3
    assert env.local_rank == 1
    assert env.world_size == 8
    assert env.local_world_size == 4
    assert env.num_nodes == 2


def test_detect_slurm_srun_without_slurm_ntasks():
    """sbatch --ntasks-per-node without global -n often leaves SLURM_NTASKS unset."""
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "5",
        "SLURM_LOCALID": "1",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_NNODES": "2",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_srun"
    assert env.world_size == 8
    assert env.local_world_size == 4
    assert env.global_rank == 5
    assert env.local_rank == 1


def test_detect_slurm_srun_step_num_tasks():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "2",
        "SLURM_LOCALID": "2",
        "SLURM_STEP_NUM_TASKS": "4",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_srun"
    assert env.world_size == 4
    assert env.local_world_size == 4


def test_detect_slurm_srun_prefers_step_over_job_ntasks():
    """A smaller srun inside a larger allocation must use the step size."""
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": "4",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_STEP_NUM_TASKS_PER_NODE": "1",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_srun"
    assert env.world_size == 1
    assert env.local_world_size == 1


def test_detect_slurm_srun_procid_lower_bounds_world_size():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "3",
        "SLURM_LOCALID": "3",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_srun"
    assert env.world_size == 4
    assert env.local_world_size == 4


def test_detect_slurm_unlaunched_from_ntasks_per_node_only():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_unlaunched"
    assert env.world_size == 4
    assert env.local_world_size == 4


def test_detect_slurm_unlaunched_without_step():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "4",
        "SLURM_NTASKS_PER_NODE": "4",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_unlaunched"
    assert env.global_rank == 0
    assert env.world_size == 4
    assert env.allocated_gpus == 1


def test_detect_slurm_unlaunched_gpu_count_from_slurm_gpus():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_GPUS": "4",
        "SLURM_NNODES": "1",
    }
    env = detect_launch_env(environ)
    assert env.mode == "slurm_unlaunched"
    assert env.world_size == 1
    assert env.allocated_gpus == 4


def test_torchrun_takes_priority_over_slurm():
    environ = {
        "SLURM_JOB_ID": "12345",
        "SLURM_STEP_ID": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": "2",
        "SLURM_NNODES": "1",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
    }
    env = detect_launch_env(environ)
    assert env.mode == "torchrun"
    assert env.global_rank == 1


def test_validate_distributed_launch_rejects_unlaunched_multi_task():
    env = LaunchEnv(mode="slurm_unlaunched", world_size=4, local_world_size=4)
    with pytest.raises(RuntimeError) as exc_info:
        validate_distributed_launch(env)
    msg = str(exc_info.value)
    assert "without an external launcher" in msg
    assert "srun" in msg
    assert "torchrun" in msg
    assert "<GPUS_PER_NODE>" in msg
    assert "--account" not in msg
    assert "--partition" not in msg


def test_validate_distributed_launch_allows_single_task_slurm():
    validate_distributed_launch(
        LaunchEnv(mode="slurm_unlaunched", world_size=1, local_world_size=1)
    )


def test_validate_distributed_launch_rejects_unlaunched_multi_gpu():
    env = LaunchEnv(mode="slurm_unlaunched", world_size=1, allocated_gpus=4)
    with pytest.raises(RuntimeError) as exc_info:
        validate_distributed_launch(env)
    assert "without an external launcher" in str(exc_info.value)


def test_validate_distributed_launch_allows_supported_modes():
    for mode in ("single", "slurm_srun", "torchrun"):
        validate_distributed_launch(LaunchEnv(mode=mode))


def test_resolve_master_addr_from_nodelist():
    assert resolve_master_addr({"SLURM_NODELIST": "nid[0001-0004]", "SLURM_NNODES": "2"}) == "nid0001"
    assert resolve_master_addr({"SLURM_NODELIST": "host0,host1", "SLURM_NNODES": "2"}) == "host0"
    # Prefer compute NODELIST over login LAUNCH_NODE_IPADDR on multi-node.
    assert (
        resolve_master_addr(
            {
                "SLURM_LAUNCH_NODE_IPADDR": "10.1.2.3",
                "SLURM_NODELIST": "nid[0001-0004]",
                "SLURM_NNODES": "2",
            }
        )
        == "nid0001"
    )
    # Single-node interactive: never use login IP.
    assert (
        resolve_master_addr(
            {
                "SLURM_LAUNCH_NODE_IPADDR": "10.1.2.3",
                "SLURM_NODELIST": "nid0001",
                "SLURM_NNODES": "1",
            }
        )
        == "127.0.0.1"
    )
    assert resolve_master_addr({"MASTER_ADDR": "explicit.example"}) == "explicit.example"
    assert resolve_master_addr({}) == "127.0.0.1"


def test_resolve_master_port_from_job_id():
    assert resolve_master_port({"MASTER_PORT": "1234"}) == "1234"
    assert resolve_master_port({"SLURM_JOB_ID": "12345"}) == str(2345 + 15000)
    assert resolve_master_port({}) == "29500"


def test_resolve_world_size_prefers_launch_env():
    env = LaunchEnv(mode="slurm_srun", world_size=8, local_world_size=4, num_nodes=2)
    assert resolve_world_size("auto", env) == 8
    assert resolve_world_size(3, env) == 8  # YAML num_nodes ignored when WORLD_SIZE known


def test_resolve_world_size_yaml_fills_missing():
    env = LaunchEnv(mode="torchrun", world_size=1, local_world_size=2, num_nodes=1)
    assert resolve_world_size(3, env) == 6
    assert resolve_world_size("auto", env) == 1


def test_export_torchrun_env_from_slurm(monkeypatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_NODELIST", "gpu[01-04]")

    launch = LaunchEnv(
        mode="slurm_srun",
        global_rank=3,
        local_rank=3,
        world_size=4,
        local_world_size=4,
        num_nodes=1,
    )
    export_torchrun_env(launch)
    assert os.environ["RANK"] == "3"
    assert os.environ["WORLD_SIZE"] == "4"
    assert os.environ["LOCAL_WORLD_SIZE"] == "4"
    assert os.environ["MASTER_ADDR"] == "gpu01"
    assert os.environ["MASTER_PORT"] == str(42 + 15000)


def test_export_torchrun_env_does_not_overwrite_existing(monkeypatch):
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("MASTER_ADDR", "kept.example")
    monkeypatch.setenv("MASTER_PORT", "1234")
    launch = LaunchEnv(mode="slurm_srun", global_rank=0, world_size=4, local_world_size=4)
    export_torchrun_env(launch)
    assert os.environ["RANK"] == "9"
    assert os.environ["WORLD_SIZE"] == "16"
    assert os.environ["MASTER_ADDR"] == "kept.example"
    assert os.environ["MASTER_PORT"] == "1234"


def test_infer_num_workers_explicit():
    assert infer_num_workers(8, environ={}) == 8
    # 0 is explicit (no workers), not auto.
    assert infer_num_workers(0, environ={"SLURM_CPUS_PER_TASK": "16"}) == 0


def test_infer_num_workers_prefers_cpus_per_task():
    environ = {"SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "4"}
    env = LaunchEnv(mode="slurm_srun", world_size=4, local_world_size=4)
    # Must not use SLURM_NTASKS (4) as the worker count.
    assert infer_num_workers(-1, env=env, environ=environ) == 7


def test_infer_num_workers_caps_cpus_per_task():
    environ = {"SLURM_CPUS_PER_TASK": "32"}
    env = LaunchEnv(mode="slurm_srun", world_size=4, local_world_size=4)
    assert infer_num_workers(-1, env=env, environ=environ) == 8


def test_infer_num_workers_one_cpu_per_task_still_uses_one_worker():
    environ = {"SLURM_CPUS_PER_TASK": "1"}
    env = LaunchEnv(mode="slurm_srun", world_size=1, local_world_size=1)
    assert infer_num_workers(-1, env=env, environ=environ) == 1


def test_infer_num_workers_cpu_count_fallback(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    env = LaunchEnv(mode="torchrun", world_size=4, local_world_size=4)
    # 16 CPUs / 4 ranks → 4 per rank, leave one for the trainer.
    assert infer_num_workers(-1, env=env, environ={}) == 3


def test_infer_num_workers_cpu_count_single_process_is_capped(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    env = LaunchEnv(mode="single", world_size=1, local_world_size=1)
    assert infer_num_workers(-1, env=env, environ={}) == 8


def test_infer_num_workers_cpu_count_does_not_use_all_cores(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    env = LaunchEnv(mode="single", world_size=1, local_world_size=1)
    assert infer_num_workers(-1, env=env, environ={}) == 8


def test_file_barrier_two_ranks(tmp_path):
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]

    def _run(launch: LaunchEnv) -> None:
        barrier(launch, sync_dir=str(tmp_path), name="test", timeout_seconds=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    leftover = list(tmp_path.iterdir())
    assert leftover == []


def test_named_file_barriers_can_reuse_sync_dir(tmp_path):
    """Successive barriers must not short-circuit on a previous .done flag."""
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]

    def _run(launch: LaunchEnv) -> None:
        barrier(launch, sync_dir=str(tmp_path), name="first", timeout_seconds=10.0)
        barrier(launch, sync_dir=str(tmp_path), name="second", timeout_seconds=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    leftover = list(tmp_path.iterdir())
    assert leftover == []


def test_rank0_exclusive_propagates_failure_to_peers(tmp_path):
    """Peers must raise instead of proceeding to preload leftover artifacts."""
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]
    errors: list = [None, None]

    def _run(launch: LaunchEnv) -> None:
        def _work() -> None:
            raise ValueError("hdf5 exploded")

        try:
            run_rank0_exclusive(
                _work,
                env=launch,
                sync_dir=str(tmp_path),
                name="boom",
                handshake_timeout_seconds=5.0,
                work_timeout_seconds=5.0,
            )
        except Exception as exc:
            errors[launch.global_rank] = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    assert isinstance(errors[0], ValueError)
    assert "hdf5 exploded" in str(errors[0])
    assert isinstance(errors[1], RuntimeError)
    assert "hdf5 exploded" in str(errors[1])


def test_rank0_exclusive_work_exceeds_handshake_timeout(tmp_path):
    """Non-zero ranks must not use the handshake timeout during rank-0 work.

    DataHub HDF5 builds often take longer than the 30-minute handshake. The
    old pattern (rank 0 works, then barrier) let peers TimeoutError first.
    """
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]
    handshake = 1.0
    work_seconds = 1.5

    def _run(launch: LaunchEnv) -> None:
        def _work() -> None:
            time.sleep(work_seconds)

        run_rank0_exclusive(
            _work,
            env=launch,
            sync_dir=str(tmp_path),
            name="slow_cache",
            handshake_timeout_seconds=handshake,
            work_timeout_seconds=5.0,
            poll_interval=0.05,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))


def test_rank0_exclusive_single_process_runs_fn(tmp_path):
    ran = []
    run_rank0_exclusive(
        lambda: ran.append(1),
        env=LaunchEnv(mode="single"),
        sync_dir=str(tmp_path),
        name="single",
    )
    assert ran == [1]


def test_init_process_group_skips_single_process():
    assert init_process_group(LaunchEnv(mode="single")) is False
    assert init_process_group(LaunchEnv(mode="slurm_unlaunched", world_size=4)) is False


def test_logger_uses_shared_file_name_on_all_ranks(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    from enerzyme.utils.base_logger import Logger

    log = Logger("enerzyme-rank-io-test")
    log.log_path = str(tmp_path)
    log.log_file_name = "Enerzyme_shared.log"

    logger = log.get_logger()
    logger.info("peer info should not create a rank file")
    created = list(tmp_path.glob("*.log"))
    assert created == []


@pytest.mark.parametrize(("mode", "visible", "expected"), [("torchrun", 2, 1), ("slurm_srun", 1, 0)])
def test_cuda_binding_respects_visibility_and_transport(monkeypatch, mode, visible, expected):
    import torch
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b" if visible == 1 else "GPU-a,GPU-b")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    launch = LaunchEnv(mode, global_rank=1, local_rank=1, world_size=2, local_world_size=2)
    assert native_device_for_launch(launch, True) == torch.device("cuda", expected)
    assert selected == [expected]
    assert "NCCL_P2P_DISABLE" not in os.environ and "NCCL_SHM_DISABLE" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")


def test_invalid_torchrun_world_and_multi_range_host():
    with pytest.raises(ValueError):
        validate_distributed_launch(detect_launch_env({"RANK":"2","LOCAL_RANK":"0","WORLD_SIZE":"2"}))
    assert resolve_master_addr({"SLURM_NNODES":"2", "SLURM_STEP_NODELIST":"rack[2-3]gpu[01,03]"}) == "rack2gpu01"


def test_restart_after_abandoned_file_round(tmp_path, monkeypatch):
    import enerzyme.tasks.distributed as distributed
    a, b = [LaunchEnv("torchrun", global_rank=i, world_size=2, local_world_size=2) for i in (0, 1)]
    with pytest.raises(TimeoutError):
        run_rank0_exclusive(lambda: None, a, sync_dir=tmp_path, handshake_timeout_seconds=0.05, poll_interval=0.01)
    monkeypatch.setattr(distributed, "_round_numbers", {})
    ran = []
    def call(env):
        run_rank0_exclusive(lambda: ran.append(1), env, sync_dir=tmp_path, handshake_timeout_seconds=3, work_timeout_seconds=3, poll_interval=0.01)
    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(call, b)
        time.sleep(0.05)
        call(a)
        future.result()
    assert ran == [1]


def test_missing_root_times_out_before_long_work(tmp_path):
    peer = LaunchEnv("torchrun", global_rank=1, world_size=2, local_world_size=2)
    with pytest.raises(TimeoutError):
        run_rank0_exclusive(lambda: None, peer, sync_dir=tmp_path,
                            handshake_timeout_seconds=0.05, poll_interval=0.01)


def test_slurm_batch_shell_and_smaller_step():
    allocation = {"SLURM_JOB_ID": "42", "SLURM_NTASKS_PER_NODE": "4", "SLURM_NNODES": "2",
                  "SLURM_PROCID": "0", "SLURM_STEP_ID": "batch"}
    assert detect_launch_env(allocation).mode == "slurm_unlaunched"
    step = {**allocation, "SLURM_STEP_ID": "1", "SLURM_STEP_NUM_TASKS": "1", "SLURM_STEP_NUM_NODES": "1"}
    launch = detect_launch_env(step)
    assert launch.local_world_size == launch.world_size == launch.num_nodes == 1
    assert resolve_master_addr(step) == "127.0.0.1"
