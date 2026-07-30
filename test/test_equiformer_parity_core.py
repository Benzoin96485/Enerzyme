"""Core latent numerical parity vs vendored Equiformer MD17."""

from __future__ import annotations

import torch

from equiformer_parity_utils import (
    assert_close,
    build_enerzyme_parts,
    build_official_md17,
    build_radius_graph,
    copy_official_weights_to_enerzyme,
    load_parity_molecule,
    make_parity_hparams,
    official_node_features_after_norm,
    to_enerzyme_edges,
)


def test_core_latent_parity_after_norm():
    dtype = torch.float64
    hp = make_parity_hparams()
    torch.manual_seed(10)

    mol = load_parity_molecule(dtype=dtype)
    za, pos, r_max = mol["Za"], mol["pos"], mol["r_max"]
    # Fixture Za max is 8; ensure within max_Za
    assert int(za.max()) <= hp["max_Za"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="feature", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)

    ref = official_node_features_after_norm(
        official, za, pos, batch, edge_src, edge_dst, edge_vec
    )
    atom_emb = embed.get_atom_embedding(za)
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])
    # Sanity: RBF matches official on the same distances
    assert_close(
        rbf_vals,
        official.rbf(ez["Dij_sr"]),
        atol=1e-6,
        rtol=1e-5,
        err_msg="rbf_on_graph",
    )

    ours = core.encode_irreps(
        vij_sr=ez["vij_sr"],
        idx_i_sr=ez["idx_i_sr"],
        idx_j_sr=ez["idx_j_sr"],
        rbf=rbf_vals,
        atom_embedding=atom_emb,
        batch_seg=batch,
    )
    assert_close(ours, ref, atol=1e-5, rtol=1e-5, err_msg="core_latent")
