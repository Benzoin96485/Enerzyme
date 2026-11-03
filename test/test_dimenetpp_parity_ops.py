"""Numerical parity of the DimeNet++ interaction vs PyG ``InteractionPPBlock``.

The released block filters and down-projects every edge, then gathers
triplets for the spherical product. These tests copy PyG weights into
:class:`~enerzyme.models.dimenet.interaction.DimeNetPPInteractionBlock`.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.extend(["..", "."])

from dimenet_parity_utils import (  # noqa: E402
    SWISH_PARAMS,
    assert_close,
    copy_linear_to_dense,
    copy_residual,
    pyg_swish,
)


def _copy_interaction_pp(dst, src) -> None:
    copy_linear_to_dense(dst.dense_rbf1, src.lin_rbf1)
    copy_linear_to_dense(dst.dense_rbf2, src.lin_rbf2)
    copy_linear_to_dense(dst.dense_sbf1, src.lin_sbf1)
    copy_linear_to_dense(dst.dense_sbf2, src.lin_sbf2)
    copy_linear_to_dense(dst.dense_kj, src.lin_kj)
    copy_linear_to_dense(dst.dense_ji, src.lin_ji)
    copy_linear_to_dense(dst.down, src.lin_down)
    copy_linear_to_dense(dst.up, src.lin_up)
    for left, right in zip(dst.layers_before_skip, src.layers_before_skip):
        copy_residual(left, right)
    copy_linear_to_dense(dst.final_before_skip, src.lin)
    for left, right in zip(dst.layers_after_skip, src.layers_after_skip):
        copy_residual(left, right)


def _pair(num_triplets: int, seed: int = 0):
    from torch_geometric.nn.models.dimenet import InteractionPPBlock

    from enerzyme.models.dimenet.interaction import DimeNetPPInteractionBlock

    hidden, n_rad, n_sph = 8, 4, 3
    int_emb, basis_emb = 5, 3
    dtype = torch.float64
    pyg = InteractionPPBlock(
        hidden_channels=hidden,
        int_emb_size=int_emb,
        basis_emb_size=basis_emb,
        num_spherical=n_sph,
        num_radial=n_rad,
        num_before_skip=1,
        num_after_skip=1,
        act=pyg_swish(),
    ).to(dtype)
    ours = DimeNetPPInteractionBlock(
        dim_embedding=hidden,
        num_rbf=n_rad,
        num_sbf=n_sph * n_rad,
        int_emb_size=int_emb,
        basis_emb_size=basis_emb,
        num_before_skip=1,
        num_after_skip=1,
        activation_fn="swish",
        activation_params=dict(SWISH_PARAMS),
    ).to(dtype)
    _copy_interaction_pp(ours, pyg)
    torch.manual_seed(seed)
    num_edges = 11
    x = torch.randn(num_edges, hidden, dtype=dtype)
    rbf = torch.randn(num_edges, n_rad, dtype=dtype)
    sbf = torch.randn(num_triplets, n_sph * n_rad, dtype=dtype)
    if num_triplets == 0:
        idx_kj = torch.empty(0, dtype=torch.long)
        idx_ji = torch.empty(0, dtype=torch.long)
    else:
        idx_kj = torch.randint(0, num_edges, (num_triplets,))
        idx_ji = torch.randint(0, num_edges, (num_triplets,))
    return ours, pyg, x, rbf, sbf, idx_kj, idx_ji


def test_interaction_pp_matches_pyg():
    pytest.importorskip("torch_geometric")
    ours, pyg, x, rbf, sbf, idx_kj, idx_ji = _pair(num_triplets=23)
    assert_close(
        ours(x, rbf, sbf, idx_kj, idx_ji),
        pyg(x, rbf, sbf, idx_kj, idx_ji),
        atol=1e-12,
        rtol=1e-12,
        err_msg="interaction_pp",
    )


def test_interaction_pp_empty_triplets_match_pyg():
    pytest.importorskip("torch_geometric")
    ours, pyg, x, rbf, sbf, idx_kj, idx_ji = _pair(num_triplets=0, seed=1)
    assert_close(
        ours(x, rbf, sbf, idx_kj, idx_ji),
        pyg(x, rbf, sbf, idx_kj, idx_ji),
        atol=1e-12,
        rtol=1e-12,
        err_msg="interaction_pp_empty",
    )
