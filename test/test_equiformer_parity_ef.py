"""Direct-mode energy/force parity vs vendored Equiformer MD17."""

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
    official_atomic_energy,
    official_node_features_after_norm,
    to_enerzyme_edges,
)


def test_direct_energy_and_force_parity():
    dtype = torch.float64
    hp = make_parity_hparams()
    torch.manual_seed(11)

    mol = load_parity_molecule(dtype=dtype)
    za = mol["Za"]
    pos = mol["pos"].clone().requires_grad_(True)
    r_max = mol["r_max"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="direct", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)
    edge_sh = o3.spherical_harmonics(
        l=core.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
    )
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])
    atom_emb = embed.get_atom_embedding(za)

    # --- energies from shared latent + head ---
    with torch.no_grad():
        # Recompute graph without grad for feature path comparison first
        pos_det = pos.detach()
        edge_src_d, edge_dst_d, edge_vec_d = build_radius_graph(pos_det, batch, r_max)
        ez_d = to_enerzyme_edges(edge_src_d, edge_dst_d, edge_vec_d)
        edge_sh_d = o3.spherical_harmonics(
            l=core.irreps_edge_attr,
            x=edge_vec_d,
            normalize=True,
            normalization="component",
        )
        rbf_d = rbf.get_rbf(ez_d["Dij_sr"])
        feats_ref = official_node_features_after_norm(
            official, za, pos_det, batch, edge_src_d, edge_dst_d, edge_vec_d
        )
        feats_ours = core.encode_irreps(
            vij_sr=ez_d["vij_sr"],
            idx_i_sr=ez_d["idx_i_sr"],
            idx_j_sr=ez_d["idx_j_sr"],
            rbf=rbf_d,
            atom_embedding=atom_emb,
            batch_seg=batch,
        )
        assert_close(feats_ours, feats_ref, atol=1e-5, rtol=1e-5, err_msg="latent_ef")

        ea_ref, e_ref = official_atomic_energy(
            official,
            feats_ref,
            batch,
            edge_src=edge_src_d,
            edge_dst=edge_dst_d,
            edge_sh=edge_sh_d,
            edge_scalars=rbf_d,
        )
        ea_ours, e_ours = enerzyme_atomic_energy(
            core,
            feats_ours,
            batch,
            edge_src=edge_src_d,
            edge_dst=edge_dst_d,
            edge_sh=edge_sh_d,
            edge_scalars=rbf_d,
            avg_num_nodes=hp["avg_num_nodes"],
        )
        assert_close(ea_ours, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea")
        assert_close(e_ours, e_ref, atol=1e-5, rtol=1e-5, err_msg="E")

    # --- forces via autograd of graph energy ---
    # Rebuild graph from pos with grad; use official forward for reference forces.
    e_off, f_off = official(za, pos, batch)
    e_off = e_off.view(-1)

    # Enerzyme: encode + head + scaled scatter, then -dE/dpos
    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
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
    e_graph = e_graph.view(-1)
    f_ours = -torch.autograd.grad(
        e_graph.sum(),
        pos,
        create_graph=False,
        retain_graph=False,
    )[0]

    assert_close(e_graph.detach(), e_off.detach(), atol=1e-5, rtol=1e-5, err_msg="E_grad")
    assert_close(f_ours, f_off, atol=1e-4, rtol=1e-4, err_msg="Fa")
