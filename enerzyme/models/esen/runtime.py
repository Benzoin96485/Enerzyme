"""Pinned fairchem / UMA runtime for the DC-UMA C-PCM PoC.

The PoC must import fairchem from the pinned source tree. If that source cannot
be imported, or its HEAD commit does not match ``PINNED_FAIRCHEM_COMMIT``,
raise — never fall back to a different installed ``fairchem-core``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PINNED_FAIRCHEM_COMMIT = "de5db01588da57665bde96966091d268a9b6b8f7"
PINNED_FAIRCHEM_ROOT = Path("/global/cfs/cdirs/m4787/src/fairchem")
PINNED_FAIRCHEM_SRC = PINNED_FAIRCHEM_ROOT / "src"
FAIRCHEM_CACHE_DIR = Path("/global/cfs/cdirs/m4787/models/fairchem")
HF_HUB_OFFLINE_DEFAULT = "1"
UMA_MODEL_NAME = "uma-s-1p2"
OMOL_TASK_NAME = "omol"
DEFAULT_UMA_CHECKPOINT = (
    FAIRCHEM_CACHE_DIR
    / "models--facebook--UMA"
    / "snapshots"
    / "be2896459a03fcde05e20d2fcefd11f450601fce"
    / "checkpoints"
    / "uma-s-1p2.pt"
)
EXPERIMENTS_ROOT = Path("/global/cfs/cdirs/m4787/experiments/dc_uma_cpcm")
CHECKPOINT_PLACEHOLDER = "UMA_CHECKPOINT"


class PinnedFairchemError(RuntimeError):
    """Pinned fairchem source is missing, wrong commit, or shadowed by another install."""


def apply_runtime_env(overwrite: bool = False) -> dict[str, str]:
    """Set cache / offline / checkpoint env vars used by job scripts.

    Does not change ``PYTHONPATH``; call :func:`ensure_pinned_fairchem` for that.
    """
    applied: dict[str, str] = {}
    defaults = {
        "FAIRCHEM_CACHE_DIR": str(FAIRCHEM_CACHE_DIR),
        "HF_HUB_OFFLINE": HF_HUB_OFFLINE_DEFAULT,
        "UMA_CHECKPOINT": str(DEFAULT_UMA_CHECKPOINT),
        "WANDB_MODE": "disabled",
    }
    for key, value in defaults.items():
        if overwrite or not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
        else:
            applied[key] = os.environ[key]
    return applied


def prepend_pinned_src_to_sys_path() -> None:
    src = str(PINNED_FAIRCHEM_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def read_fairchem_git_commit(repo_root: Path | None = None) -> str:
    root = Path(repo_root) if repo_root is not None else PINNED_FAIRCHEM_ROOT
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PinnedFairchemError(
            f"Could not read git HEAD for pinned fairchem at {root}: {exc}"
        ) from exc


def _fairchem_module_is_pinned(fairchem: ModuleType) -> Path:
    core = getattr(fairchem, "core", None)
    module_file_raw = getattr(core, "__file__", None) or getattr(fairchem, "__file__", None)
    if not module_file_raw:
        raise PinnedFairchemError(
            "fairchem.core has no __file__; cannot verify the pinned source. "
            f"Set PYTHONPATH={PINNED_FAIRCHEM_SRC}."
        )
    module_file = Path(module_file_raw).resolve()
    pinned_src = PINNED_FAIRCHEM_SRC.resolve()
    try:
        module_file.relative_to(pinned_src)
    except ValueError as exc:
        raise PinnedFairchemError(
            "fairchem was imported from "
            f"{module_file}, not the pinned source {pinned_src}. "
            "Set PYTHONPATH="
            f"{pinned_src} and do not fall back to an installed fairchem-core."
        ) from exc
    return module_file


def ensure_pinned_fairchem() -> ModuleType:
    """Import fairchem from the pinned source and verify the git commit.

    Raises:
        PinnedFairchemError: source missing, wrong commit, or already imported
            from a different install. Never silently uses another fairchem.
    """
    if not PINNED_FAIRCHEM_SRC.is_dir():
        raise PinnedFairchemError(
            f"Pinned fairchem source directory does not exist: {PINNED_FAIRCHEM_SRC}"
        )
    prepend_pinned_src_to_sys_path()
    import fairchem.core  # noqa: WPS433  # namespace package; pin is fairchem.core

    fairchem = sys.modules["fairchem"]
    _fairchem_module_is_pinned(fairchem)
    commit = read_fairchem_git_commit()
    if not commit.startswith(PINNED_FAIRCHEM_COMMIT[:12]):
        raise PinnedFairchemError(
            "Pinned fairchem git HEAD is "
            f"{commit}, expected {PINNED_FAIRCHEM_COMMIT}. "
            "Do not fall back to another fairchem-core install."
        )
    return fairchem


def resolve_uma_checkpoint(checkpoint_path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the uma-s-1p2 checkpoint from an argument, env, or default cache path."""
    if checkpoint_path is None or str(checkpoint_path).strip() in {
        "",
        CHECKPOINT_PLACEHOLDER,
    }:
        raw = os.environ.get("UMA_CHECKPOINT", str(DEFAULT_UMA_CHECKPOINT))
    else:
        raw = str(checkpoint_path)
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"UMA checkpoint is not a file: {path}. "
            "Set UMA_CHECKPOINT to the local uma-s-1p2.pt under FAIRCHEM_CACHE_DIR."
        )
    return path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 of ``path``.

    Hugging Face cache blobs are stored under their SHA256 hex digest. When the
    resolved filename is a 64-character hex string, use it directly.
    """
    resolved = path.resolve()
    name = resolved.name.lower()
    if len(name) == 64 and all(char in "0123456789abcdef" for char in name):
        return name
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_runtime_versions(
    checkpoint_path: Path | None = None,
    include_orca: bool = True,
) -> dict[str, Any]:
    """Record Python / PyTorch / ASE / fairchem / ORCA versions and hashes."""
    apply_runtime_env()
    fairchem = ensure_pinned_fairchem()
    import ase
    import torch

    ckpt = resolve_uma_checkpoint(checkpoint_path)
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "ase": ase.__version__,
        "fairchem_file": str(_fairchem_module_is_pinned(fairchem)),
        "fairchem_package_version": getattr(fairchem.core, "__version__", None),
        "fairchem_git_commit": read_fairchem_git_commit(),
        "fairchem_pinned_commit": PINNED_FAIRCHEM_COMMIT,
        "fairchem_cache_dir": os.environ.get("FAIRCHEM_CACHE_DIR"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "uma_model_name": UMA_MODEL_NAME,
        "uma_checkpoint": str(ckpt),
        "uma_checkpoint_sha256": sha256_file(ckpt),
    }
    if torch.cuda.is_available():
        versions["cuda_device"] = torch.cuda.get_device_name(0)
    if include_orca:
        versions["orca"] = _orca_version()
    return versions


def _orca_version() -> str | None:
    import shutil

    orca = shutil.which("orca")
    if orca is None:
        return None
    try:
        proc = subprocess.run(
            [orca, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return orca
    text = (proc.stdout or proc.stderr or "").strip()
    return text.splitlines()[0] if text else orca
