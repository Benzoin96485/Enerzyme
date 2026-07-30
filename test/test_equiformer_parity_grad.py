"""Gradient parity vs official Equiformer (no optimizer / training loop)."""

from __future__ import annotations

import torch
from e3nn import o3

from equiformer_parity_utils import (
    assert_close,
    build_enerzyme_parts,
    build_official_md17,
    build_radius_graph,
    copy_official_weights_to_enerzyme,
    enerzyme_atomic_energy,
    load_parity_molecule,
    make_parity_hparams,
    to_enerzyme_edges,
)


def _zero_grads(module: torch.nn.Module) -> None:
    for p in module.parameters():
        if p.grad is not None:
            p.grad = None


def test_loss_grads_match_official():
    """L = E.sum(); compare dL/dRa and a shared TransBlock parameter grad."""
    dtype = torch.float64
    hp = make_parity_hparams()
    torch.manual_seed(12)

    mol = load_parity_molecule(dtype=dtype)
    za = mol["Za"]
    pos = mol["pos"].clone().requires_grad_(True)
    r_max = mol["r_max"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="direct", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    # Ensure both sides track the same leaf params we will compare.
    for p in official.parameters():
        p.requires_grad_(True)
    for mod in (embed, rbf, core):
        for p in mod.parameters():
            p.requires_grad_(True)

    _zero_grads(official)
    e_off, _ = official(za, pos, batch)
    loss_off = e_off.sum()
    loss_off.backward()
    g_pos_off = pos.grad.detach().clone()
    # Parameter under GraphAttention projection in first block
    g_param_off = official.blocks[0].ga.proj.tp.weight.grad.detach().clone()

    # Fresh pos for Enerzyme path (same values)
    pos2 = mol["pos"].clone().requires_grad_(True)
    _zero_grads(embed)
    _zero_grads(rbf)
    _zero_grads(core)

    edge_src, edge_dst, edge_vec = build_radius_graph(pos2, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)
    edge_sh = o3.spherical_harmonics(
        l=core.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
    )
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])
    atom_emb = embed.get_atom_embedding(za)
    feats = core.encode_irreps(
        vij_sr=ez["vij_sr"],
        idx_i_sr=ez["idx_i_sr"],
        idx_j_sr=ez["idx_j_sr"],
        rbf=rbf_vals,
        atom_embedding=atom_emb,
        batch_seg=batch,
    )
    _, e_graph = enerzyme_atomic_energy(
        core,
        feats,
        batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_sh=edge_sh,
        edge_scalars=rbf_vals,
        avg_num_nodes=hp["avg_num_nodes"],
    )
    loss_ours = e_graph.sum()
    loss_ours.backward()

    g_pos_ours = pos2.grad.detach().clone()
    g_param_ours = core.blocks[0].ga.proj.tp.weight.grad.detach().clone()

    assert_close(loss_ours.detach(), loss_off.detach(), atol=1e-5, rtol=1e-5, err_msg="L")
    assert_close(g_pos_ours, g_pos_off, atol=1e-4, rtol=1e-4, err_msg="dL/dRa")
    assert_close(g_param_ours, g_param_off, atol=1e-4, rtol=1e-4, err_msg="dL/dW_proj")
