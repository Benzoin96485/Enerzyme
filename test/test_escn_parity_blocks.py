"""Numerical parity: Enerzyme Message/Layer blocks vs vendored fairchem eSCN blocks."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from escn_parity_utils import (  # noqa: E402
    assert_close,
    build_complete_graph,
    build_layer_pair,
    build_message_pair,
    deterministic_edge_rot_mat,
    load_parity_mol,
)


def _prep_inputs(h):
    from enerzyme.models.so3 import CoefficientMapping, SO3_Embedding, SO3_Rotation

    mol = load_parity_mol()
    pos = mol["pos"]
    Za = mol["Za"].clamp(max=h["max_Za"])
    edge_index = build_complete_graph(pos.shape[0])
    vij = pos[edge_index[0]] - pos[edge_index[1]]
    # Enerzyme convention: vij = Rj - Ri with idx_i target, idx_j source in some stacks;
    # for parity use the same edge_index and vij consistently on both sides.
    rot_mat = deterministic_edge_rot_mat(vij)
    lmax, mmax = h["lmax"], h["mmax"]
    device = pos.device
    dtype = pos.dtype
    channels = h["sphere_channels"]
    num_atoms = pos.shape[0]

    SO3_edge_rot = [SO3_Rotation(rot_mat, lmax)]
    mapping = CoefficientMapping([lmax], [mmax], device)

    # After edge rotate, embedding size shrinks to mapping.res_size — MessageBlock
    # rotates internally. Start from full lmax embedding like the paper forward.
    x0 = torch.randn(num_atoms, (lmax + 1) ** 2, channels, dtype=dtype)
    distance_features = torch.randn(edge_index.shape[1], h["num_rbf"], dtype=dtype)

    def make_emb(Emb):
        x = Emb(0, [lmax], channels, device, dtype)
        x.set_embedding(x0.clone())
        return x

    return {
        "Za": Za,
        "edge_index": edge_index,
        "SO3_edge_rot": SO3_edge_rot,
        "mapping": mapping,
        "distance_features": distance_features,
        "x0": x0,
        "make_emb": make_emb,
        "device": device,
        "dtype": dtype,
        "lmax": lmax,
        "sphere_channels": channels,
    }


def test_message_block_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from so3 import SO3_Embedding as OffEmb
    from so3 import SO3_Rotation as OffRot
    from so3 import CoefficientMapping as OffMap

    torch.manual_seed(0)
    ez, off, h, _, _ = build_message_pair()
    inp = _prep_inputs(h)

    # Official SO3_Rotation / mapping (same rot mat)
    vij = None
    mol = load_parity_mol()
    pos = mol["pos"]
    edge_index = inp["edge_index"]
    vij = pos[edge_index[0]] - pos[edge_index[1]]
    rot_mat = deterministic_edge_rot_mat(vij)
    off_rot = [OffRot(rot_mat, h["lmax"])]
    off_map = OffMap([h["lmax"]], [h["mmax"]], inp["device"])

    ez_x = inp["make_emb"](EZEmb)
    off_x = OffEmb(0, [h["lmax"]], h["sphere_channels"], inp["device"], inp["dtype"])
    off_x.set_embedding(inp["x0"].clone())

    ez_out = ez(
        ez_x,
        inp["Za"],
        inp["distance_features"],
        edge_index,
        inp["SO3_edge_rot"],
        inp["mapping"],
    )
    off_out = off(
        off_x,
        inp["Za"],
        inp["distance_features"],
        edge_index,
        off_rot,
        off_map,
    )
    assert_close(ez_out.embedding, off_out.embedding, atol=1e-5, rtol=1e-5)


def test_layer_block_matches_upstream():
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from so3 import SO3_Embedding as OffEmb
    from so3 import SO3_Rotation as OffRot
    from so3 import CoefficientMapping as OffMap

    torch.manual_seed(1)
    ez, off, h, _, _ = build_layer_pair()
    inp = _prep_inputs(h)
    mol = load_parity_mol()
    pos = mol["pos"]
    edge_index = inp["edge_index"]
    vij = pos[edge_index[0]] - pos[edge_index[1]]
    rot_mat = deterministic_edge_rot_mat(vij)
    off_rot = [OffRot(rot_mat, h["lmax"])]
    off_map = OffMap([h["lmax"]], [h["mmax"]], inp["device"])

    ez_x = inp["make_emb"](EZEmb)
    off_x = OffEmb(0, [h["lmax"]], h["sphere_channels"], inp["device"], inp["dtype"])
    off_x.set_embedding(inp["x0"].clone())

    ez_out = ez(
        ez_x,
        inp["Za"],
        inp["distance_features"],
        edge_index,
        inp["SO3_edge_rot"],
        inp["mapping"],
    )
    off_out = off(
        off_x,
        inp["Za"],
        inp["distance_features"],
        edge_index,
        off_rot,
        off_map,
    )
    assert_close(ez_out.embedding, off_out.embedding, atol=1e-5, rtol=1e-5)
