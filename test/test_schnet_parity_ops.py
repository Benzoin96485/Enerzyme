"""Operator-level numerical parity vs torch_geometric SchNet."""

from __future__ import annotations

import math
import sys

import torch

sys.path.extend(["..", "."])

from schnet_parity_utils import (  # noqa: E402
    PARITY_HPARAMS,
    assert_close,
    make_parity_graph,
)


def test_gaussian_rbf_schnet_flavor_matches_pyg():
    from torch_geometric.nn.models.schnet import GaussianSmearing as PygSmearing

    from enerzyme.models.layers.rbf import GaussianRBFLayer

    hp = PARITY_HPARAMS
    dtype = torch.float64
    torch.manual_seed(0)
    d = torch.rand(48, dtype=dtype) * hp["cutoff"]

    pyg = PygSmearing(0.0, hp["cutoff"], hp["num_gaussians"]).to(dtype)
    ours = GaussianRBFLayer(
        num_rbf=hp["num_gaussians"],
        cutoff_sr=hp["cutoff"],
        flavor="SchNet",
        apply_cutoff_fn=False,
    ).to(dtype)

    assert_close(ours.center, pyg.offset, atol=1e-12, rtol=1e-12, err_msg="centers")
    assert_close(
        ours.get_rbf(d),
        pyg(d),
        atol=1e-6,
        rtol=1e-5,
        err_msg="rbf",
    )


def test_cosine_transition_matches_pyg_inline():
    from enerzyme.models.mp.basis.envelope import cosine_transition

    cutoff = PARITY_HPARAMS["cutoff"]
    dtype = torch.float64
    d = torch.linspace(0.0, cutoff, 41, dtype=dtype)
    pyg = 0.5 * (torch.cos(d * math.pi / cutoff) + 1.0)
    ours = cosine_transition(d, cutoff=cutoff)
    assert_close(ours, pyg, atol=1e-7, rtol=1e-7, err_msg="cosine")


def _load_pyg_block(block, pyg_block):
    """Load a PyG ``InteractionBlock`` into a preset block through the ``main`` key rules."""
    from enerzyme.models.compat.main_branch import convert_state_dict

    state, _ = convert_state_dict("schnet", {f"interactions.0.{k}": v for k, v in pyg_block.state_dict().items()})
    block.load_state_dict({k[len("blocks.0."):]: v for k, v in state.items()})


def _schnet_context(edge_index, edge_weight, edge_attr, num_nodes):
    from enerzyme.models.mp.graph import GraphContext, GraphRequires

    inputs = {"idx_i_sr": edge_index[0], "idx_j_sr": edge_index[1], "Dij_sr": edge_weight, "rbf": edge_attr}
    requires = GraphRequires(vectors=False, edge_convention="swap", optional_cutoff_values=False)
    return GraphContext.from_inputs(inputs, requires, num_nodes=num_nodes)


def test_interaction_block_matches_pyg():
    from torch_geometric.nn.models.schnet import GaussianSmearing as PygSmearing
    from torch_geometric.nn.models.schnet import InteractionBlock as PygIB

    from enerzyme.models.mp.block import BlockState
    from enerzyme.models.presets.scalar import schnet_block

    hp = PARITY_HPARAMS
    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype)
    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]
    num_nodes = graph["pos"].size(0)

    torch.manual_seed(1)
    pyg = PygIB(
        hp["hidden_channels"],
        hp["num_gaussians"],
        hp["num_filters"],
        hp["cutoff"],
    ).to(dtype)
    ez = schnet_block(hp["hidden_channels"], hp["num_gaussians"], hp["num_filters"], hp["cutoff"]).to(dtype)
    _load_pyg_block(ez, pyg)

    edge_attr = PygSmearing(0.0, hp["cutoff"], hp["num_gaussians"]).to(dtype)(
        edge_weight
    )
    x = torch.randn(num_nodes, hp["hidden_channels"], dtype=dtype)
    ctx = _schnet_context(edge_index, edge_weight, edge_attr, num_nodes)

    assert_close(
        ez(BlockState(scalar=x), ctx).scalar,
        x + pyg(x, edge_index, edge_weight, edge_attr),
        atol=1e-5,
        rtol=1e-5,
        err_msg="InteractionBlock",
    )


def test_residual_interaction_stack_matches_pyg():
    from torch_geometric.nn.models.schnet import GaussianSmearing as PygSmearing
    from torch_geometric.nn.models.schnet import InteractionBlock as PygIB
    from torch.nn import ModuleList

    from enerzyme.models.layers.rbf import GaussianRBFLayer
    from enerzyme.models.presets.scalar import schnet

    hp = PARITY_HPARAMS
    dtype = torch.float64
    graph = make_parity_graph(dtype=dtype, seed=2)
    edge_index = graph["edge_index"]
    edge_weight = graph["edge_weight"]
    num_nodes = graph["pos"].size(0)

    torch.manual_seed(3)
    pyg_blocks = ModuleList(
        [
            PygIB(
                hp["hidden_channels"],
                hp["num_gaussians"],
                hp["num_filters"],
                hp["cutoff"],
            )
            for _ in range(hp["num_interactions"])
        ]
    ).to(dtype)

    ez_core = schnet(
        hidden_channels=hp["hidden_channels"],
        dim_embedding=hp["num_filters"],
        num_interactions=hp["num_interactions"],
        num_rbf=hp["num_gaussians"],
        cutoff_sr=hp["cutoff"],
        output_mode="feature",
    ).to(dtype)
    for ez_block, pyg_block in zip(ez_core.blocks, pyg_blocks):
        _load_pyg_block(ez_block, pyg_block)

    rbf_layer = GaussianRBFLayer(
        num_rbf=hp["num_gaussians"],
        cutoff_sr=hp["cutoff"],
        flavor="SchNet",
        apply_cutoff_fn=False,
    ).to(dtype)
    edge_attr = rbf_layer.get_rbf(edge_weight)
    # Cross-check against PyG smearing on the same edges.
    pyg_attr = PygSmearing(0.0, hp["cutoff"], hp["num_gaussians"]).to(dtype)(
        edge_weight
    )
    assert_close(edge_attr, pyg_attr, atol=1e-6, rtol=1e-5, err_msg="edge_attr")

    h0 = torch.randn(num_nodes, hp["hidden_channels"], dtype=dtype)
    h_pyg = h0.clone()
    for block in pyg_blocks:
        h_pyg = h_pyg + block(h_pyg, edge_index, edge_weight, edge_attr)

    out = ez_core.get_output(
        idx_i_sr=edge_index[0],
        idx_j_sr=edge_index[1],
        Dij_sr=edge_weight,
        rbf=edge_attr,
        atom_embedding=h0.clone(),
    )
    assert_close(
        out["atom_feature"],
        h_pyg,
        atol=1e-5,
        rtol=1e-5,
        err_msg="residual interaction stack",
    )
