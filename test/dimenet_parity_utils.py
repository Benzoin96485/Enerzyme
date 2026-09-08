"""Helpers for original DimeNet numerical parity vs torch_geometric.nn.models.dimenet.

PyG 2.6.1 ``DimeNet.forward`` needs ``torch-cluster`` (radius_graph) and
``torch-sparse`` (``triplets``). Those packages are optional here, so tests
rebuild the official forward from PyG submodules plus a dense radius graph.
``dimenet_utils.sph_harm_prefactor`` still uses removed ``numpy.math``; the
shim below is test-only and is not a production dependency.

Angle convention: PyG original DimeNet uses the TF pretrained bug (angle at
atom ``i``). Enerzyme production Core uses the paper / DimeNet++ angle at
``j``. Energy parity injects PyG angles into both sides.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch import Tensor
from torch.nn import Module

pytest.importorskip("torch_geometric")

# PyG 2.6 dimenet_utils: numpy>=2 dropped ``np.math``.
np.math = math

from enerzyme.models.dimenet.triplets import directed_triplets  # noqa: E402

PARITY_HPARAMS = {
    "hidden_channels": 8,
    "out_channels": 1,
    "num_blocks": 2,
    "num_bilinear": 4,
    "num_spherical": 3,
    "num_radial": 4,
    "cutoff": 5.0,
    "envelope_exponent": 5,
    "num_before_skip": 1,
    "num_after_skip": 1,
    "num_output_layers": 2,
}

SWISH_PARAMS = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
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


def pyg_swish():
    from torch_geometric.nn.resolver import activation_resolver

    return activation_resolver("swish")


def copy_linear_to_dense(dst: Module, src: Module) -> None:
    dst.weight.data.copy_(src.weight)
    if dst.bias is not None:
        if src.bias is None:
            raise ValueError("source Linear has no bias but destination DenseLayer does")
        dst.bias.data.copy_(src.bias)


def copy_residual(dst: Module, src: Module) -> None:
    copy_linear_to_dense(dst.dense1, src.lin1)
    copy_linear_to_dense(dst.dense2, src.lin2)


def copy_interaction(dst: Module, src: Module) -> None:
    copy_linear_to_dense(dst.dense_rbf, src.lin_rbf)
    copy_linear_to_dense(dst.dense_sbf, src.lin_sbf)
    copy_linear_to_dense(dst.dense_kj, src.lin_kj)
    copy_linear_to_dense(dst.dense_ji, src.lin_ji)
    dst.W.data.copy_(src.W)
    for left, right in zip(dst.layers_before_skip, src.layers_before_skip):
        copy_residual(left, right)
    copy_linear_to_dense(dst.final_before_skip, src.lin)
    for left, right in zip(dst.layers_after_skip, src.layers_after_skip):
        copy_residual(left, right)


def copy_embedding_block(dst: Module, src: Module) -> None:
    copy_linear_to_dense(dst.dense_rbf, src.lin_rbf)
    copy_linear_to_dense(dst.dense, src.lin)


def copy_output_head(dst_head: Module, src_output: Module) -> None:
    for layer, lin in zip(dst_head[:-1], src_output.lins):
        copy_linear_to_dense(layer, lin)
    copy_linear_to_dense(dst_head[-1], src_output.lin)


def copy_pyg_dimenet_into_enerzyme(
    core: Module,
    rbf_layer: Module,
    atom_embedding: Module,
    readout: Module,
    pyg: Module,
) -> None:
    rbf_layer.bessel_weights.data.copy_(pyg.rbf.freq)
    atom_embedding.embedding.weight.data.copy_(pyg.emb.emb.weight)
    copy_embedding_block(core.emb_block, pyg.emb)
    for dst, src in zip(core.interactions, pyg.interaction_blocks):
        copy_interaction(dst, src)
    for dst, src in zip(core.atom_readouts, pyg.output_blocks):
        copy_linear_to_dense(dst.dense_rbf, src.lin_rbf)
    for dst, src in zip(readout.heads, pyg.output_blocks):
        copy_output_head(dst, src)


def make_parity_graph(
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Tensor]:
    """Small directed radius graph (no torch-cluster).

    ``idx_i`` / ``idx_j`` follow Enerzyme (receiver, sender). PyG DimeNet uses
    ``edge_index = [j, i]`` (row=sender, col=receiver).
    """
    torch.manual_seed(seed)
    z = torch.tensor([8, 1, 1, 6], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.96, 0.05, 0.02],
            [-0.24, 0.93, -0.04],
            [1.40, 1.10, 0.80],
        ],
        dtype=dtype,
    )
    cutoff = PARITY_HPARAMS["cutoff"]
    dist = torch.cdist(pos, pos)
    jj, ii = torch.where((dist > 0) & (dist <= cutoff))
    return {
        "z": z,
        "pos": pos,
        "idx_i": ii,
        "idx_j": jj,
        "edge_index": torch.stack([jj, ii], dim=0),
    }


def pyg_dimenet_angles(
    pos: Tensor,
    triplet_i: Tensor,
    triplet_j: Tensor,
    triplet_k: Tensor,
) -> Tensor:
    """Official PyG original-DimeNet angles (at atom ``i``, TF pretrained bug)."""
    if triplet_i.numel() == 0:
        return pos.new_zeros(0)
    pos_ji = pos[triplet_j] - pos[triplet_i]
    pos_ki = pos[triplet_k] - pos[triplet_i]
    cosine_like = (pos_ji * pos_ki).sum(dim=-1)
    sine_like = torch.linalg.cross(pos_ji, pos_ki, dim=-1).norm(dim=-1)
    return torch.atan2(sine_like, cosine_like)


def triplet_indices(
    idx_i: Tensor,
    idx_j: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return directed_triplets(idx_i, idx_j, num_nodes)


def pyg_dimenet_energy(
    pyg: Module,
    z: Tensor,
    pos: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
) -> Tensor:
    """Replicate ``DimeNet.forward`` with a supplied graph (no torch-sparse)."""
    num_nodes = z.size(0)
    idx_kj, idx_ji, trip_i, trip_j, trip_k = triplet_indices(idx_i, idx_j, num_nodes)
    dist = (pos[idx_i] - pos[idx_j]).pow(2).sum(dim=-1).sqrt()
    angle = pyg_dimenet_angles(pos, trip_i, trip_j, trip_k)
    rbf = pyg.rbf(dist)
    sbf = pyg.sbf(dist, angle, idx_kj)
    x = pyg.emb(z, rbf, idx_i, idx_j)
    contrib = pyg.output_blocks[0](x, rbf, idx_i, num_nodes=num_nodes)
    for interaction, output in zip(pyg.interaction_blocks, pyg.output_blocks[1:]):
        x = interaction(x, rbf, sbf, idx_kj, idx_ji)
        contrib = contrib + output(x, rbf, idx_i, num_nodes=num_nodes)
    return contrib.sum(dim=0)


def make_tiny_pyg_dimenet(dtype: torch.dtype = torch.float64) -> Module:
    from torch_geometric.nn.models.dimenet import DimeNet

    hp = PARITY_HPARAMS
    model = DimeNet(
        hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"],
        num_blocks=hp["num_blocks"],
        num_bilinear=hp["num_bilinear"],
        num_spherical=hp["num_spherical"],
        num_radial=hp["num_radial"],
        cutoff=hp["cutoff"],
        envelope_exponent=hp["envelope_exponent"],
        num_before_skip=hp["num_before_skip"],
        num_after_skip=hp["num_after_skip"],
        num_output_layers=hp["num_output_layers"],
        output_initializer="glorot_orthogonal",
    )
    return model.to(dtype).eval()


def make_tiny_enerzyme_stack(dtype: torch.dtype = torch.float64):
    from enerzyme.models.dimenet.core import DimeNetCore
    from enerzyme.models.layers.atom_embedding import RandomAtomEmbedding
    from enerzyme.models.layers.rbf import BesselRBFLayer
    from enerzyme.models.layers.readout import HierachicalReadout

    hp = PARITY_HPARAMS
    core = DimeNetCore(
        dim_embedding=hp["hidden_channels"],
        num_rbf=hp["num_radial"],
        cutoff_sr=hp["cutoff"],
        num_blocks=hp["num_blocks"],
        num_bilinear=hp["num_bilinear"],
        num_spherical=hp["num_spherical"],
        num_before_skip=hp["num_before_skip"],
        num_after_skip=hp["num_after_skip"],
        envelope_exponent=hp["envelope_exponent"],
        activation_fn="swish",
        activation_params=dict(SWISH_PARAMS),
    ).to(dtype)
    rbf = BesselRBFLayer(
        num_rbf=hp["num_radial"],
        cutoff_sr=hp["cutoff"],
        flavor="dimenet",
        trainable=True,
        envelope_exponent=hp["envelope_exponent"],
    ).to(dtype)
    atom_embedding = RandomAtomEmbedding(max_Za=94, dim_embedding=hp["hidden_channels"]).to(
        dtype
    )
    readout = HierachicalReadout(
        output_fields=["Ea"],
        built_layers=[core],
        head_type="mlp",
        num_hidden_layers=hp["num_output_layers"],
        activation_fn="swish",
        activation_params=dict(SWISH_PARAMS),
        use_bias_out=False,
    ).to(dtype)
    core.eval()
    rbf.eval()
    atom_embedding.eval()
    readout.eval()
    return core, rbf, atom_embedding, readout


def enerzyme_energy(
    core: Module,
    rbf_layer: Module,
    atom_embedding: Module,
    readout: Module,
    z: Tensor,
    pos: Tensor,
    idx_i: Tensor,
    idx_j: Tensor,
    *,
    angle_at: str = "i",
) -> Tensor:
    """DimeNet energy from Core internals + shared HierarchicalReadout.

    ``angle_at='i'`` matches PyG original DimeNet; ``'j'`` is the paper
    convention used by :meth:`DimeNetCore.get_output`.
    """
    num_nodes = z.size(0)
    vij = pos[idx_j] - pos[idx_i]
    dist = vij.norm(dim=-1)
    idx_kj, idx_ji, trip_i, trip_j, trip_k = triplet_indices(idx_i, idx_j, num_nodes)
    if angle_at == "i":
        angle = pyg_dimenet_angles(pos, trip_i, trip_j, trip_k)
    elif angle_at == "j":
        from enerzyme.models.dimenet.triplets import triplet_angles

        angle = triplet_angles(vij, idx_kj, idx_ji)
    else:
        raise ValueError(f"angle_at must be 'i' or 'j', got {angle_at!r}")
    rbf = rbf_layer.get_rbf(dist)
    sbf = core.sbf(dist, angle, idx_kj)
    messages = core.emb_block(atom_embedding.get_embedding(z), rbf, idx_i, idx_j)
    features = [core.atom_readouts[0](messages, rbf, idx_i, num_nodes)]
    for block, atom_readout in zip(core.interactions, core.atom_readouts[1:]):
        messages = block(messages, rbf, sbf, idx_kj, idx_ji)
        features.append(atom_readout(messages, rbf, idx_i, num_nodes))
    atom_feature = torch.stack(features, dim=-1)
    return readout.get_output(atom_feature)["Ea"].sum()
