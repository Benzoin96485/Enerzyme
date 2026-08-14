"""Unit tests for enerzyme.tasks.distributed launch detection."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from enerzyme.tasks.distributed import (
    LaunchEnv,
    barrier,
    bind_single_visible_gpu,
    detect_launch_env,
    export_torchrun_env,
    infer_num_workers,
    init_process_group,
    is_global_zero,
    local_rank,
    prepare_nccl_for_single_visible_gpu,
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


def test_prepare_nccl_disables_p2p_for_single_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    prepare_nccl_for_single_visible_gpu()
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NCCL_SHM_DISABLE"] == "1"


def test_prepare_nccl_leaves_multi_gpu_visible_alone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    prepare_nccl_for_single_visible_gpu()
    assert "NCCL_P2P_DISABLE" not in os.environ
    assert "NCCL_SHM_DISABLE" not in os.environ


def test_infer_num_workers_explicit():
    assert infer_num_workers(8, environ={}) == 8
    # 0 is explicit (no workers), not auto.
    assert infer_num_workers(0, environ={"SLURM_CPUS_PER_TASK": "16"}) == 0


def test_infer_num_workers_prefers_cpus_per_task():
    environ = {"SLURM_CPUS_PER_TASK": "8", "SLURM_NTASKS": "4"}
    env = LaunchEnv(mode="slurm_srun", world_size=4, local_world_size=4)
    # Must not use SLURM_NTASKS (4) as the worker count.
    assert infer_num_workers(-1, env=env, environ=environ) == 7


def test_infer_num_workers_cpu_count_fallback(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    env = LaunchEnv(mode="torchrun", world_size=4, local_world_size=4)
    assert infer_num_workers(-1, env=env, environ={}) == 4


def test_file_barrier_two_ranks(tmp_path):
    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]

    def _run(launch: LaunchEnv) -> None:
        barrier(launch, sync_dir=str(tmp_path), name="test", timeout_seconds=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))


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


def test_file_barrier_ignores_stale_done_flag(tmp_path):
    """A leftover .done from a previous job must not let rank 1 skip check-in."""
    prefix = ".enerzyme_barrier_2_stale"
    (tmp_path / f"{prefix}.done").write_text("old-token")
    (tmp_path / f"{prefix}.r0").write_text("old-token")
    (tmp_path / f"{prefix}.r1").write_text("old-token")
    (tmp_path / f"{prefix}.gen").write_text("old-token")

    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]

    def _run(launch: LaunchEnv) -> None:
        barrier(launch, sync_dir=str(tmp_path), name="stale", timeout_seconds=10.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    done = (tmp_path / f"{prefix}.done").read_text()
    assert done != "old-token"


def test_file_barrier_ignores_stale_gen_without_done(tmp_path):
    """Leftover .gen/.r0 with no .done must not look like a live round.

    Rank 1 is started first so it would previously check in with the stale
    token after seeing .done already gone.
    """
    prefix = ".enerzyme_barrier_2_stale_gen"
    (tmp_path / f"{prefix}.gen").write_text("old-token")
    (tmp_path / f"{prefix}.r0").write_text("old-token")
    (tmp_path / f"{prefix}.r1").write_text("old-token")

    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]

    def _run(launch: LaunchEnv) -> None:
        if launch.global_rank == 0:
            time.sleep(0.3)
        barrier(launch, sync_dir=str(tmp_path), name="stale_gen", timeout_seconds=5.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    done = (tmp_path / f"{prefix}.done").read_text()
    assert done != "old-token"


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


def test_rank0_exclusive_ignores_stale_done_flag(tmp_path):
    prefix = ".enerzyme_barrier_2_slow_stale"
    (tmp_path / f"{prefix}.done").write_text("old-token")
    (tmp_path / f"{prefix}.r0").write_text("old-token")
    (tmp_path / f"{prefix}.r1").write_text("old-token")
    (tmp_path / f"{prefix}.gen").write_text("old-token")

    envs = [
        LaunchEnv(mode="torchrun", global_rank=0, world_size=2, local_world_size=2),
        LaunchEnv(mode="torchrun", global_rank=1, world_size=2, local_world_size=2),
    ]
    marker = tmp_path / "built"

    def _run(launch: LaunchEnv) -> None:
        def _work() -> None:
            marker.write_text("ok")

        run_rank0_exclusive(
            _work,
            env=launch,
            sync_dir=str(tmp_path),
            name="slow_stale",
            handshake_timeout_seconds=10.0,
            work_timeout_seconds=10.0,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(_run, envs))

    assert marker.read_text() == "ok"
    done = (tmp_path / f"{prefix}.done").read_text()
    assert done != "old-token"


def test_rank0_exclusive_single_process_runs_fn(tmp_path):
    ran = []
    run_rank0_exclusive(
        lambda: ran.append(1),
        env=LaunchEnv(mode="single"),
        sync_dir=str(tmp_path),
        name="single",
    )
    assert ran == [1]


def test_bind_single_visible_gpu_torchrun_selects_local_rank(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    launch = LaunchEnv(mode="torchrun", global_rank=3, local_rank=3, world_size=4, local_world_size=4)
    bind_single_visible_gpu(launch, cuda=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NCCL_SHM_DISABLE"] == "1"


def test_bind_single_visible_gpu_keeps_already_single_id(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    launch = LaunchEnv(mode="slurm_srun", global_rank=1, local_rank=1, world_size=4, local_world_size=4)
    bind_single_visible_gpu(launch, cuda=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NCCL_SHM_DISABLE"] == "1"


def test_bind_single_visible_gpu_selects_from_visible_list(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    monkeypatch.delenv("NCCL_SHM_DISABLE", raising=False)
    launch = LaunchEnv(mode="torchrun", global_rank=2, local_rank=2, world_size=4, local_world_size=4)
    bind_single_visible_gpu(launch, cuda=True)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"


def test_bind_single_visible_gpu_rejects_local_rank_past_visible_list(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    launch = LaunchEnv(mode="torchrun", global_rank=3, local_rank=3, world_size=4, local_world_size=4)
    with pytest.raises(RuntimeError, match="local_rank=3 is out of range"):
        bind_single_visible_gpu(launch, cuda=True)


def test_init_process_group_skips_single_process():
    assert init_process_group(LaunchEnv(mode="single")) is False
    assert init_process_group(LaunchEnv(mode="slurm_unlaunched", world_size=4)) is False


def test_logger_uses_per_rank_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    from enerzyme.utils.base_logger import Logger

    log = Logger("enerzyme-rank-io-test")
    log.log_path = str(tmp_path)
    assert log._file_log_name().endswith("_rank1.log")
