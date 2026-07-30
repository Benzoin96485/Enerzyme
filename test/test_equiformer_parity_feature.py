"""Feature-mode / modular readout parity vs vendored Equiformer MD17.

Production ``SimpleReadout`` (Dense MLP after 0e extract) is *not* numerically
aligned with the official LinearRS energy head. These tests cover:

1. Feature-mode ``atom_feature`` fed into the official MLP / attn head.
2. ``EquiformerGraphAttentionReadout`` weight-matched to official ``use_attn_head``.
3. 0e extract + official LinearRS MLP (scalar path shared with SimpleReadout).
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
    official_atomic_energy,
    official_node_features_after_norm,
    scaled_scatter_energy,
    to_enerzyme_edges,
)
from enerzyme.models.irreps_tools import extract_scalar_0e
from enerzyme.models.layers.readout import EquiformerGraphAttentionReadout


def test_feature_atom_feature_feeds_official_mlp_head():
    """get_output full irreps → official LinearRS MLP → ScaledScatter."""
    dtype = torch.float64
    hp = make_parity_hparams(use_attn_head=False)
    torch.manual_seed(21)

    mol = load_parity_molecule(dtype=dtype)
    za, pos, r_max = mol["Za"], mol["pos"], mol["r_max"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="feature", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)
    edge_sh = o3.spherical_harmonics(
        l=core.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
    )
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])

    feats_ref = official_node_features_after_norm(
        official, za, pos, batch, edge_src, edge_dst, edge_vec
    )
    feats_ours = enerzyme_feature_atom_feature(core, embed, rbf, za, batch, ez)
    assert_close(feats_ours, feats_ref, atol=1e-5, rtol=1e-5, err_msg="feat_mlp")

    ea_ref, e_ref = official_atomic_energy(
        official,
        feats_ref,
        batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_sh=edge_sh,
        edge_scalars=rbf_vals,
    )
    ea_ours = official.head(feats_ours).view(-1)
    e_ours = scaled_scatter_energy(ea_ours, batch, hp["avg_num_nodes"])
    assert_close(ea_ours, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_feature_mlp")
    assert_close(e_ours, e_ref, atol=1e-5, rtol=1e-5, err_msg="E_feature_mlp")

    # Same scalars SimpleReadout would see (pure 0e feature → identity extract)
    scalars = extract_scalar_0e(feats_ours, core.feature_irreps)
    assert_close(scalars, feats_ours, atol=0.0, rtol=0.0, err_msg="0e_identity_on_64x0e")
    ea_from_0e = official.head(scalars).view(-1)
    assert_close(ea_from_0e, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_after_0e")


def test_equiformer_graph_attention_readout_parity():
    """``EquiformerGraphAttentionReadout`` matches official attn energy head."""
    dtype = torch.float64
    hp = make_parity_hparams(use_attn_head=True)
    torch.manual_seed(22)

    mol = load_parity_molecule(dtype=dtype)
    za, pos, r_max = mol["Za"], mol["pos"], mol["r_max"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="feature", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    readout = EquiformerGraphAttentionReadout(
        output_fields={"Ea"},
        built_layers=[core],
        irreps_head=hp["irreps_head"],
        num_heads=hp["num_heads"],
        fc_neurons=list(hp["fc_neurons"]),
        irreps_sh=hp["irreps_sh"],
        irreps_node_attr=hp["irreps_node_attr"],
        nonlinear_message=hp["nonlinear_message"],
        num_rbf=hp["num_rbf"],
        feature_irreps=core.feature_irreps,
    ).to(dtype=dtype).eval()
    readout.head.load_state_dict(official.head.state_dict())

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)
    edge_sh = o3.spherical_harmonics(
        l=core.irreps_edge_attr, x=edge_vec, normalize=True, normalization="component"
    )
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])

    feats_ref = official_node_features_after_norm(
        official, za, pos, batch, edge_src, edge_dst, edge_vec
    )
    feats_ours = enerzyme_feature_atom_feature(core, embed, rbf, za, batch, ez)
    assert_close(feats_ours, feats_ref, atol=1e-5, rtol=1e-5, err_msg="feat_attn")

    ea_ref, e_ref = official_atomic_energy(
        official,
        feats_ref,
        batch,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_sh=edge_sh,
        edge_scalars=rbf_vals,
    )
    out = readout.get_output(
        atom_feature=feats_ours,
        idx_i_sr=ez["idx_i_sr"],
        idx_j_sr=ez["idx_j_sr"],
        vij_sr=ez["vij_sr"],
        rbf=rbf_vals,
        batch_seg=batch,
    )
    ea_ours = out["Ea"].view(-1)
    e_ours = scaled_scatter_energy(ea_ours, batch, hp["avg_num_nodes"])
    assert_close(ea_ours, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_attn_readout")
    assert_close(e_ours, e_ref, atol=1e-5, rtol=1e-5, err_msg="E_attn_readout")
