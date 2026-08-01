"""Numerical parity: EquiformerV3 FFN / TransBlock vs upstream fixture."""

from __future__ import annotations

import sys

import torch

sys.path.extend(["..", "."])

from equiformer_v3_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
)


def _copy_params(dst: torch.nn.Module, src: torch.nn.Module, skip_substr: str = "") -> None:
    dst_params = dict(dst.named_parameters())
    src_params = dict(src.named_parameters())
    with torch.no_grad():
        for name, p_dst in dst_params.items():
            if skip_substr and skip_substr in name:
                continue
            if name not in src_params:
                raise AssertionError(f"missing param in src: {name}")
            p_src = src_params[name]
            if p_dst.shape != p_src.shape:
                raise AssertionError(f"shape mismatch {name}")
            p_dst.copy_(p_src)


def test_ffn_matches_upstream():
    from enerzyme.models.equiformer_v3.interaction import FeedForwardNetwork as EZFFN
    from transformer_block import FeedForwardNetwork as OffFFN

    torch.manual_seed(0)
    dtype = torch.float64
    h = PARITY_HPARAMS
    lmax, mmax = h["lmax"], h["mmax"]
    c = h["sphere_channels"]
    hidden = h["ffn_hidden_channels"]
    kwargs = dict(
        num_in_channels=c,
        num_hidden_channels=hidden,
        num_out_channels=c,
        lmax=lmax,
        mmax=mmax,
        grid_resolution_list=[8, 8],
        activation="sep-merge_gates2_swiglu",
        use_grid_mlp=True,
        dropout=0.0,
    )
    ez = EZFFN(**kwargs).to(dtype)
    off = OffFFN(**kwargs).to(dtype)
    _copy_params(ez, off)
    x = torch.randn(5, (lmax + 1) ** 2, c, dtype=dtype)
    assert_close(ez(x.clone()), off(x.clone()), atol=1e-4, rtol=1e-4)


def test_transblock_matches_upstream():
    from enerzyme.models.equiformer_v3.interaction import TransBlockV3 as EZBlock
    from enerzyme.models.so3 import SO3RotationFused, init_edge_rot_mat
    from eqv3_so3 import SO3Rotation as OffRot
    from transformer_block import TransBlockV3 as OffBlock

    torch.manual_seed(1)
    # Upstream SO3Rotation buffers are float32; keep parity in float32.
    dtype = torch.float32
    h = PARITY_HPARAMS
    lmax, mmax = h["lmax"], h["mmax"]
    c = h["sphere_channels"]
    N, E = 4, 12
    edge_index = torch.stack(
        [
            torch.arange(E, dtype=torch.long) % N,
            (torch.arange(E, dtype=torch.long) + 1) % N,
        ],
        dim=0,
    )
    vij = torch.randn(E, 3, dtype=dtype)
    vij = vij / torch.linalg.norm(vij, dim=-1, keepdim=True).clamp_min(1e-6)

    ez_rot = SO3RotationFused(lmax, mmax)
    off_rot = OffRot(lmax, mmax)
    # Same edge frames for both stacks (upstream init uses RNG for ortho basis).
    edge_rot = init_edge_rot_mat(vij)
    ez_rot.set_wigner(edge_rot)
    off_rot.set_wigner(edge_rot.clone())

    edge_channels_list = [h["num_rbf"], h["edge_channels"], h["edge_channels"]]
    kwargs = dict(
        num_in_channels=c,
        attn_hidden_channels=h["attn_hidden_channels"],
        num_heads=h["num_heads"],
        attn_alpha_channels=h["attn_alpha_channels"],
        attn_value_channels=h["attn_value_channels"],
        ffn_hidden_channels=h["ffn_hidden_channels"],
        num_out_channels=c,
        lmax=lmax,
        mmax=mmax,
        attn_grid_resolution_list=[8, 8],
        ffn_grid_resolution_list=[8, 8],
        max_num_elements=h["max_Za"],
        edge_channels_list=edge_channels_list,
        use_atom_edge_embedding=True,
        attn_activation="sep-merge_gates2_swiglu",
        ffn_activation="sep-merge_gates2_swiglu",
        use_grid_mlp=True,
        norm_type="merge_layer_norm",
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
    )
    ez = EZBlock(so3_rotation=ez_rot, **kwargs).to(dtype)
    off = OffBlock(so3_rotation=off_rot, **kwargs).to(dtype)
    _copy_params(ez, off, skip_substr="so3_rotation")

    x = torch.randn(N, (lmax + 1) ** 2, c, dtype=dtype)
    Za = torch.arange(N, dtype=torch.long) % (h["max_Za"] - 1) + 1
    rbf = torch.randn(E, h["num_rbf"], dtype=dtype)
    env = torch.rand(E, 1, dtype=dtype)
    batch = torch.zeros(N, dtype=torch.long)
    src, tgt = Za[edge_index[0]], Za[edge_index[1]]
    ez_out = ez(x.clone(), src, tgt, rbf, edge_index, env, batch)
    off_out = off(x.clone(), src, tgt, rbf, edge_index, env, batch)
    assert_close(ez_out, off_out, atol=1e-4, rtol=1e-4)
