"""Core latent numerical parity vs vendored Equiformer MD17.

Covers both ``encode_irreps`` (parity hook) and feature-mode ``get_output``
(full irreps ``atom_feature`` + ``feature_irreps`` contract).
"""

from __future__ import annotations

import torch
from e3nn import o3

from equiformer_parity_utils import (
    assert_close,
    build_enerzyme_parts,
    build_official_md17,
    build_radius_graph,
    copy_official_weights_to_enerzyme,
    enerzyme_feature_atom_feature,
    load_parity_molecule,
    make_parity_hparams,
    official_node_features_after_norm,
    to_enerzyme_edges,
)
from enerzyme.models.e3nn_nn import extract_scalar_0e, scalar_0e_dim


def _run_latent_parity(hp: dict, *, dtype: torch.dtype = torch.float64) -> None:
    torch.manual_seed(10)
    mol = load_parity_molecule(dtype=dtype)
    za, pos, r_max = mol["Za"], mol["pos"], mol["r_max"]
    assert int(za.max()) <= hp["max_Za"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="feature", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    # Feature / scalar-readout contract
    assert core.feature_irreps == str(o3.Irreps(hp["irreps_feature"]))
    assert core.dim_feature_out == scalar_0e_dim(hp["irreps_feature"])

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)

    ref = official_node_features_after_norm(
        official, za, pos, batch, edge_src, edge_dst, edge_vec
    )
    atom_emb = embed.get_atom_embedding(za)
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])
    assert_close(
        rbf_vals,
        official.rbf(ez["Dij_sr"]),
        atol=1e-6,
        rtol=1e-5,
        err_msg="rbf_on_graph",
    )

    ours_encode = core.encode_irreps(
        vij_sr=ez["vij_sr"],
        idx_i_sr=ez["idx_i_sr"],
        idx_j_sr=ez["idx_j_sr"],
        rbf=rbf_vals,
        atom_embedding=atom_emb,
        batch_seg=batch,
    )
    ours_get = enerzyme_feature_atom_feature(core, embed, rbf, za, batch, ez)

    assert_close(ours_encode, ref, atol=1e-5, rtol=1e-5, err_msg="encode_irreps")
    assert_close(ours_get, ref, atol=1e-5, rtol=1e-5, err_msg="get_output_atom_feature")
    assert_close(ours_get, ours_encode, atol=0.0, rtol=0.0, err_msg="get_output_vs_encode")

    # 0e extract width matches dim_feature_out (identity when feature is pure 0e)
    scalars = extract_scalar_0e(ours_get, core.feature_irreps)
    assert scalars.shape[-1] == core.dim_feature_out


def test_core_latent_parity_after_norm():
    _run_latent_parity(make_parity_hparams())


def test_core_latent_parity_mixed_irreps_feature():
    """Full-irreps ``atom_feature`` when ``irreps_feature`` includes l>0.

    Official MLP head (SiLU Activation) only accepts pure-0e features, so use
    ``use_attn_head=True`` — latent path is identical; only the unused head differs.
    """
    _run_latent_parity(
        make_parity_hparams(
            irreps_feature="32x0e+16x1e+8x2e",
            use_attn_head=True,
        )
    )
