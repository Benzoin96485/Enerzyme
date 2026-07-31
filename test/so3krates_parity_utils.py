"""Helpers for So3krates numerical parity against vendored So3krates-torch fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from numpy.testing import assert_allclose
from torch import Tensor
from torch.nn import Module

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "so3krates_upstream"
if str(FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURE_ROOT))

PARITY_HPARAMS = {
    "degrees": [1, 2, 3],
    "num_features": 12,
    "num_heads": 3,
    "num_rbf": 8,
    "avg_num_neighbors": 4.0,
}


def assert_close(a: Tensor, b: Tensor, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
    )


def copy_state_dict(dst: Module, src: Module) -> None:
    dst.load_state_dict(src.state_dict())
