"""Numerical parity: EquiformerV2 FFN / TransBlock vs upstream fixture."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from equiformer_v2_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    build_complete_graph,
    build_so3_grids_v2,
    copy_state_dict,
    deterministic_edge_rot_mat,
)


def test_feedforward_network_matches_upstream():
    from enerzyme.models.equiformer_v2.transformer_block import (
        FeedForwardNetwork as EZFFN,
    )
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from eqv2_so3 import SO3_Embedding as OffEmb
    from transformer_block import FeedForwardNetwork as OffFFN

    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax = h["lmax"]
    ez_grid, off_grid = build_so3_grids_v2(lmax)
    kwargs = dict(
        sphere_channels=h["sphere_channels"],
        hidden_channels=h["ffn_hidden_channels"],
        output_channels=h["sphere_channels"],
        lmax_list=[lmax],
        mmax_list=[h["mmax"]],
        activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
    )
    ez = EZFFN(SO3_grid=ez_grid, **kwargs).to(dtype)
    off = OffFFN(SO3_grid=off_grid, **kwargs).to(dtype)
    copy_state_dict(ez, off)

    n = 5
    emb = torch.randn(n, (lmax + 1) ** 2, h["sphere_channels"], dtype=dtype)
    ez_x = EZEmb(0, [lmax], h["sphere_channels"], device, dtype)
    off_x = OffEmb(0, [lmax], h["sphere_channels"], device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    ez_x.set_lmax_mmax([lmax], [lmax])
    off_x.set_lmax_mmax([lmax], [lmax])
    assert_close(ez(ez_x).embedding, off(off_x).embedding)


def test_transblock_v2_matches_upstream():
    from enerzyme.models.equiformer_v2.transformer_block import TransBlockV2 as EZBlock
    from enerzyme.models.so3 import CoefficientMapping as EZMap
    from enerzyme.models.so3 import SO3_Embedding as EZEmb
    from enerzyme.models.so3 import SO3_Rotation as EZRot
    from eqv2_so3 import CoefficientMappingModule as OffMap
    from eqv2_so3 import SO3_Embedding as OffEmb
    from eqv2_so3 import SO3_Rotation as OffRot
    from transformer_block import TransBlockV2 as OffBlock

    torch.manual_seed(1)
    device = torch.device("cpu")
    # Upstream SO3_Rotation.set_wigner builds Wigner in float32; keep dtype matched.
    dtype = torch.float32
    h = PARITY_HPARAMS
    lmax, mmax = h["lmax"], h["mmax"]
    n = 4
    edge_index = build_complete_graph(n)
    num_edges = edge_index.shape[1]
    vij = torch.randn(num_edges, 3, dtype=dtype)
    vij = vij / torch.linalg.norm(vij, dim=1, keepdim=True).clamp(min=1e-8)
    rot_mat = deterministic_edge_rot_mat(vij)

    ez_rot_list = [None]
    off_rot = OffRot(lmax)
    off_rot.set_wigner(rot_mat)
    # Upstream stores ModuleList-like list; use same object references.
    off_rot_list = [off_rot]
    ez_rot_list[0] = EZRot(rot_mat, lmax)

    ez_map = EZMap([lmax], [mmax], device)
    off_map = OffMap([lmax], [mmax])
    ez_grid, off_grid = build_so3_grids_v2(lmax)
    edge_channels_list = [h["num_rbf"], h["edge_channels"], h["edge_channels"]]

    common = dict(
        sphere_channels=h["sphere_channels"],
        attn_hidden_channels=h["attn_hidden_channels"],
        num_heads=h["num_heads"],
        attn_alpha_channels=h["attn_alpha_channels"],
        attn_value_channels=h["attn_value_channels"],
        ffn_hidden_channels=h["ffn_hidden_channels"],
        output_channels=h["sphere_channels"],
        lmax_list=[lmax],
        mmax_list=[mmax],
        max_num_elements=h["max_Za"] + 1,
        edge_channels_list=edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        norm_type="rms_norm_sh",
        alpha_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
    )
    ez = EZBlock(
        SO3_rotation=ez_rot_list,
        mappingReduced=ez_map,
        SO3_grid=ez_grid,
        **common,
    ).to(dtype)
    off = OffBlock(
        SO3_rotation=off_rot_list,
        mappingReduced=off_map,
        SO3_grid=off_grid,
        **common,
    ).to(dtype)
    copy_state_dict(ez, off)

    emb = torch.randn(n, (lmax + 1) ** 2, h["sphere_channels"], dtype=dtype)
    Za = torch.tensor([1, 6, 8, 1], dtype=torch.long)
    rbf = torch.randn(num_edges, h["num_rbf"], dtype=dtype)
    batch = torch.zeros(n, dtype=torch.long)

    ez_x = EZEmb(0, [lmax], h["sphere_channels"], device, dtype)
    off_x = OffEmb(0, [lmax], h["sphere_channels"], device, dtype)
    ez_x.set_embedding(emb.clone())
    off_x.set_embedding(emb.clone())
    ez_x.set_lmax_mmax([lmax], [lmax])
    off_x.set_lmax_mmax([lmax], [lmax])

    ez.eval()
    off.eval()
    with torch.no_grad():
        ez_out = ez(ez_x, Za, rbf, edge_index, batch)
        off_out = off(off_x, Za, rbf, edge_index, batch)
    assert_close(ez_out.embedding, off_out.embedding, atol=1e-4, rtol=1e-4)
