"""Operator-level numerical parity vs vendored Equiformer nets."""

from __future__ import annotations

import torch
from e3nn import o3

# Install torch_cluster stub before importing vendored nets.
import equiformer_parity_utils  # noqa: F401
from equiformer_parity_utils import assert_close, make_parity_hparams


def test_exp_normal_smearing_parity():
    from nets.graph_attention_transformer_md17 import ExpNormalSmearing as OfficialExp
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    hp = make_parity_hparams()
    dtype = torch.float64
    torch.manual_seed(0)
    d = torch.rand(32, dtype=dtype) * hp["r_max"]

    official = OfficialExp(
        cutoff_lower=0.0,
        cutoff_upper=hp["r_max"],
        num_rbf=hp["num_rbf"],
        trainable=False,
    ).to(dtype)
    ours = ExpNormalSmearing(
        num_rbf=hp["num_rbf"], cutoff_sr=hp["r_max"], cuton=0.0, trainable=False
    ).to(dtype)
    with torch.no_grad():
        ours.means.copy_(official.means)
        ours.betas.copy_(official.betas)

    assert_close(ours.get_rbf(d), official(d), atol=1e-6, rtol=1e-5, err_msg="rbf")


def test_linear_rs_parity():
    from nets.tensor_product_rescale import LinearRS as OfficialLinearRS
    from enerzyme.models.equiformer.tensor_product import LinearRS

    dtype = torch.float64
    torch.manual_seed(1)
    ir_in = o3.Irreps("16x0e+8x1e")
    ir_out = o3.Irreps("12x0e+4x1e")
    official = OfficialLinearRS(ir_in, ir_out, rescale=True).to(dtype)
    ours = LinearRS(ir_in, ir_out, rescale=True).to(dtype)
    ours.load_state_dict(official.state_dict())
    x = torch.randn(7, ir_in.dim, dtype=dtype)
    assert_close(ours(x), official(x), atol=1e-6, rtol=1e-5, err_msg="LinearRS")


def test_layer_norm_v2_parity():
    from nets.layer_norm import EquivariantLayerNormV2 as OfficialNorm
    from enerzyme.models.equiformer.norms import EquivariantLayerNormV2

    dtype = torch.float64
    torch.manual_seed(2)
    irreps = o3.Irreps("32x0e+16x1e+8x2e")
    official = OfficialNorm(irreps).to(dtype)
    ours = EquivariantLayerNormV2(irreps).to(dtype)
    ours.load_state_dict(official.state_dict())
    n = 9
    x = torch.randn(n, irreps.dim, dtype=dtype)
    batch = torch.zeros(n, dtype=torch.long)
    assert_close(
        ours(x, batch=batch),
        official(x, batch=batch),
        atol=1e-6,
        rtol=1e-5,
        err_msg="LayerNormV2",
    )


def test_edge_degree_embedding_parity():
    from nets.graph_attention_transformer import (
        EdgeDegreeEmbeddingNetwork as OfficialEDE,
    )
    from enerzyme.models.equiformer.embedding import EdgeDegreeEmbeddingNetwork

    hp = make_parity_hparams()
    dtype = torch.float64
    torch.manual_seed(3)
    ir_node = o3.Irreps(hp["irreps_node_embedding"])
    ir_edge = o3.Irreps(hp["irreps_sh"])
    fc = [hp["num_rbf"]] + list(hp["fc_neurons"])
    official = OfficialEDE(ir_node, ir_edge, fc, hp["avg_degree"]).to(dtype)
    ours = EdgeDegreeEmbeddingNetwork(ir_node, ir_edge, fc, hp["avg_degree"]).to(dtype)
    ours.load_state_dict(official.state_dict())

    n, e = 6, 20
    node_in = torch.randn(n, ir_node.dim, dtype=dtype)
    edge_attr = torch.randn(e, ir_edge.dim, dtype=dtype)
    edge_scalars = torch.randn(e, hp["num_rbf"], dtype=dtype)
    edge_src = torch.randint(0, n, (e,))
    edge_dst = torch.randint(0, n, (e,))
    batch = torch.zeros(n, dtype=torch.long)
    assert_close(
        ours(node_in, edge_attr, edge_scalars, edge_src, edge_dst, batch),
        official(node_in, edge_attr, edge_scalars, edge_src, edge_dst, batch),
        atol=1e-5,
        rtol=1e-4,
        err_msg="EdgeDegreeEmbedding",
    )


def test_trans_block_parity():
    from nets.graph_attention_transformer import TransBlock as OfficialTransBlock
    from enerzyme.models.equiformer.attention import TransBlock

    hp = make_parity_hparams()
    dtype = torch.float64
    torch.manual_seed(4)
    ir_node = o3.Irreps(hp["irreps_node_embedding"])
    ir_attr = o3.Irreps(hp["irreps_node_attr"])
    ir_edge = o3.Irreps(hp["irreps_sh"])
    ir_head = o3.Irreps(hp["irreps_head"])
    ir_mlp = o3.Irreps(hp["irreps_mlp_mid"])
    fc = [hp["num_rbf"]] + list(hp["fc_neurons"])
    kwargs = dict(
        irreps_node_input=ir_node,
        irreps_node_attr=ir_attr,
        irreps_edge_attr=ir_edge,
        irreps_node_output=ir_node,
        fc_neurons=fc,
        irreps_head=ir_head,
        num_heads=hp["num_heads"],
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=True,
        alpha_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        irreps_mlp_mid=ir_mlp,
        norm_layer="layer",
    )
    official = OfficialTransBlock(**kwargs).to(dtype)
    ours = TransBlock(**kwargs).to(dtype)
    ours.load_state_dict(official.state_dict())

    n, e = 5, 16
    node_input = torch.randn(n, ir_node.dim, dtype=dtype)
    node_attr = torch.ones(n, 1, dtype=dtype)
    edge_attr = torch.randn(e, ir_edge.dim, dtype=dtype)
    edge_scalars = torch.randn(e, hp["num_rbf"], dtype=dtype)
    edge_src = torch.randint(0, n, (e,))
    edge_dst = torch.randint(0, n, (e,))
    batch = torch.zeros(n, dtype=torch.long)
    assert_close(
        ours(
            node_input=node_input,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        ),
        official(
            node_input=node_input,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,
        ),
        atol=1e-5,
        rtol=1e-4,
        err_msg="TransBlock",
    )
