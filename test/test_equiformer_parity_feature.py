"""Feature-mode / modular readout parity vs vendored Equiformer MD17.

Production ``SimpleReadout`` (Dense MLP after 0e extract) is *not* numerically
aligned with the official LinearRS energy head. These tests cover:

1. Feature-mode ``atom_feature`` fed into the official MLP / attn head.
2. Enerzyme LinearRS head weight-matched to official ``head`` (pure 0e).
3. Mixed irreps → ``extract_scalar_0e`` → official-style LinearRS energy head.
4. ``EquiformerGraphAttentionReadout`` weight-matched to official ``use_attn_head``.
"""

from __future__ import annotations

import torch
from e3nn import o3

from equiformer_parity_utils import (
    assert_close,
    build_enerzyme_parts,
    build_linear_rs_energy_head,
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
from enerzyme.models.irreps_tools import extract_scalar_0e, scalar_0e_dim
from enerzyme.models.equiformer.interaction import EquiformerGraphAttentionReadout


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

    # Enerzyme-ported LinearRS head with official weights must match official.head
    mul0 = int(o3.Irreps(hp["irreps_feature"]).dim)  # pure 0e
    ez_head = build_linear_rs_energy_head(mul0, flavor="enerzyme", dtype=dtype)
    ez_head.load_state_dict(official.head.state_dict())
    ea_ez = ez_head(feats_ours).view(-1)
    assert_close(ea_ez, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_enerzyme_LinearRS")

    # SimpleReadout(equiformer_linear_rs) with official head weights
    from enerzyme.models.layers.readout import SimpleReadout

    class _Core:
        dim_feature_out = mul0
        feature_irreps = core.feature_irreps

    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Core()],
        head_type="equiformer_linear_rs",
    ).to(dtype=dtype).eval()
    ro.head.load_state_dict(official.head.state_dict())
    ea_ro = ro.get_output(feats_ours)["Ea"].view(-1)
    assert_close(ea_ro, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_SimpleReadout_LinearRS")


def test_mixed_irreps_0e_extract_linear_rs_energy_head_parity():
    """Mixed ``irreps_feature`` → extract 0e → official-style LinearRS energy head.

    Official MD17 MLP head cannot be constructed on mixed irreps (SiLU Activation
    requires one act per irrep). After 0e extract the path matches the official
    scalar energy MLP; we对照 vendored vs Enerzyme LinearRS with shared weights.
    """
    dtype = torch.float64
    irreps_feature = "32x0e+16x1e+8x2e"
    hp = make_parity_hparams(
        irreps_feature=irreps_feature,
        use_attn_head=True,  # so official MD17 constructs without scalar MLP head
    )
    torch.manual_seed(23)
    mul0 = scalar_0e_dim(irreps_feature)

    mol = load_parity_molecule(dtype=dtype)
    za, pos, r_max = mol["Za"], mol["pos"], mol["r_max"]
    batch = torch.zeros(za.shape[0], dtype=torch.long)

    official = build_official_md17(hp, dtype=dtype)
    embed, rbf, core = build_enerzyme_parts(hp, output_mode="feature", dtype=dtype)
    copy_official_weights_to_enerzyme(official, embed, rbf, core)

    edge_src, edge_dst, edge_vec = build_radius_graph(pos, batch, r_max)
    ez = to_enerzyme_edges(edge_src, edge_dst, edge_vec)

    feats_ref = official_node_features_after_norm(
        official, za, pos, batch, edge_src, edge_dst, edge_vec
    )
    feats_ours = enerzyme_feature_atom_feature(core, embed, rbf, za, batch, ez)
    assert_close(feats_ours, feats_ref, atol=1e-5, rtol=1e-5, err_msg="mixed_latent")
    assert feats_ours.shape[-1] == o3.Irreps(irreps_feature).dim

    scalars_ref = extract_scalar_0e(feats_ref, irreps_feature)
    scalars_ours = extract_scalar_0e(feats_ours, core.feature_irreps)
    assert scalars_ours.shape == (za.shape[0], mul0)
    assert_close(scalars_ours, scalars_ref, atol=1e-5, rtol=1e-5, err_msg="mixed_0e")

    head_off = build_linear_rs_energy_head(mul0, flavor="official", dtype=dtype)
    head_ez = build_linear_rs_energy_head(mul0, flavor="enerzyme", dtype=dtype)
    head_ez.load_state_dict(head_off.state_dict())

    ea_ref = head_off(scalars_ref).view(-1)
    ea_ours = head_ez(scalars_ours).view(-1)
    e_ref = scaled_scatter_energy(ea_ref, batch, hp["avg_num_nodes"])
    e_ours = scaled_scatter_energy(ea_ours, batch, hp["avg_num_nodes"])
    assert_close(ea_ours, ea_ref, atol=1e-5, rtol=1e-5, err_msg="Ea_mixed_LinearRS")
    assert_close(e_ours, e_ref, atol=1e-5, rtol=1e-5, err_msg="E_mixed_LinearRS")

    # Same LinearRS head on Enerzyme scalars vs official scalars (cross check)
    assert_close(
        head_off(scalars_ours).view(-1),
        ea_ref,
        atol=1e-5,
        rtol=1e-5,
        err_msg="Ea_cross_scalars",
    )


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
