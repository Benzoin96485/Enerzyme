"""Helpers for SchNet numerical parity against torch_geometric.nn.models.schnet."""

from __future__ import annotations

from typing import Dict

import pytest
import torch
from numpy.testing import assert_allclose
from torch import Tensor
from torch.nn import Module

pytest.importorskip("torch_geometric")

PARITY_HPARAMS = {
    "hidden_channels": 32,
    "num_filters": 32,
    "num_gaussians": 16,
    "num_interactions": 2,
    "cutoff": 5.0,
}


def assert_close(
    a: Tensor,
    b: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    err_msg: str = "",
) -> None:
    assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )


def copy_state_dict(dst: Module, src: Module) -> None:
    dst.load_state_dict(src.state_dict())


def make_parity_graph(
    num_atoms: int = 6,
    cutoff: float = PARITY_HPARAMS["cutoff"],
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Tensor]:
    """Build a shared radius graph for Enerzyme / PyG SchNet parity."""
    from torch_geometric.nn.models.schnet import RadiusInteractionGraph

    torch.manual_seed(seed)
    z = torch.randint(1, 10, (num_atoms,), dtype=torch.long)
    pos = torch.randn(num_atoms, 3, dtype=dtype)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    graph = RadiusInteractionGraph(cutoff=cutoff, max_num_neighbors=32)
    edge_index, edge_weight = graph(pos, batch)
    return {
        "z": z,
        "pos": pos,
        "batch": batch,
        "edge_index": edge_index,
        "edge_weight": edge_weight.to(dtype),
    }
