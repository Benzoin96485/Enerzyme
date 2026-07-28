import sys

import pytest
import torch

sys.path.extend(["..", "."])


def _rand_graph_indices(num_nodes: int, num_edges: int, device=None):
    idx_i = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    idx_j = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    return idx_i, idx_j


def test_physnet_output_mode_feature_atom_feature_shape():
    from enerzyme.models.physnet.core import PhysNetCore

    torch.manual_seed(0)
    N = 7
    num_edges = 20
    dim_embedding = 16
    num_rbf = 8
    num_blocks = 3

    core = PhysNetCore(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        num_blocks=num_blocks,
        output_mode="feature",
    )
    core = core.double()
    dtype = torch.float64
    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    atom_embedding = torch.randn(N, dim_embedding, dtype=dtype)
    rbf = torch.randn(num_edges, num_rbf, dtype=dtype)
    out = core.get_output(rbf=rbf, atom_embedding=atom_embedding, idx_i_sr=idx_i_sr, idx_j_sr=idx_j_sr)
    assert "atom_feature" in out
    atom_feature = out["atom_feature"]
    assert atom_feature.ndim == 3
    assert atom_feature.shape[0] == N
    assert atom_feature.shape[1] == dim_embedding
    assert atom_feature.shape[2] == num_blocks


def test_schnet_output_mode_feature_atom_feature_shape():
    from enerzyme.models.schnet.core import SchNetCore

    torch.manual_seed(0)
    N = 9
    num_edges = 30
    hidden_channels = 32
    dim_embedding = 32
    num_rbf = 12

    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    Dij_sr = torch.rand(num_edges)
    rbf = torch.randn(num_edges, num_rbf)
    atom_embedding = torch.randn(N, hidden_channels)

    core = SchNetCore(
        hidden_channels=hidden_channels,
        dim_embedding=dim_embedding,
        num_interactions=2,
        num_rbf=num_rbf,
        cutoff_sr=5.0,
        output_mode="feature",
    )
    out = core.get_output(idx_i_sr=idx_i_sr, idx_j_sr=idx_j_sr, Dij_sr=Dij_sr, rbf=rbf, atom_embedding=atom_embedding)
    assert "atom_feature" in out
    atom_feature = out["atom_feature"]
    assert atom_feature.ndim == 2
    assert atom_feature.shape[0] == N


def test_spookynet_output_mode_feature_atom_feature_shape():
    from enerzyme.models.spookynet.core import SpookyNetCore

    torch.manual_seed(0)
    N = 11
    num_edges = 40
    dim_embedding = 32
    num_rbf = 16

    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    vij_sr = torch.randn(num_edges, 3)
    Dij_sr = vij_sr.norm(dim=-1).clamp_min(1e-6)
    rbf = torch.randn(num_edges, num_rbf)
    atom_embedding = torch.randn(N, dim_embedding)
    batch_seg = torch.zeros(N, dtype=torch.long)

    core = SpookyNetCore(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        num_modules=2,
        num_residual_pre=1,
        num_residual_local_x=1,
        num_residual_local_s=1,
        num_residual_local_p=1,
        num_residual_local_d=1,
        num_residual_local=1,
        num_residual_nonlocal_q=1,
        num_residual_nonlocal_k=1,
        num_residual_nonlocal_v=1,
        num_residual_post=1,
        num_residual_output=1,
        activation_fn="swish",
        use_irreps=False,
        dropout_rate=0.0,
        output_mode="feature",
    )
    out = core.get_output(
        Dij_sr=Dij_sr,
        vij_sr=vij_sr,
        idx_i_sr=idx_i_sr,
        idx_j_sr=idx_j_sr,
        rbf=rbf,
        atom_embedding=atom_embedding,
        batch_seg=batch_seg,
    )
    assert "atom_feature" in out
    atom_feature = out["atom_feature"]
    assert atom_feature.ndim == 2
    assert atom_feature.shape[0] == N


def test_mace_output_mode_feature_atom_feature_shape():
    try:
        from enerzyme.models.mace.core import MACECore
    except Exception as e:
        pytest.skip(f"MACECore not importable: {e}")

    torch.manual_seed(0)
    N = 8
    num_edges = 20
    max_Za = 10
    dim_embedding = 8
    num_rbf = 4
    max_ell = 1

    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    vij_sr = torch.randn(num_edges, 3)
    rbf = torch.randn(num_edges, num_rbf)
    Za = torch.randint(1, max_Za + 1, (N,), dtype=torch.long)
    atom_embedding = torch.randn(N, dim_embedding)

    core = MACECore(
        max_Za=max_Za,
        max_ell=max_ell,
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        additional_hidden_irreps="8x1o",
        interaction_cls_first="RealAgnosticResidualInteractionBlock",
        interaction_cls="RealAgnosticResidualInteractionBlock",
        correlation=2,
        num_interactions=2,
        avg_num_neighbors=8.0,
        MLP_irreps="4x0e",
        radial_MLP=[8, 8],
        gate="silu",
        output_mode="feature",
    )
    out = core.get_output(
        Za=Za,
        vij_sr=vij_sr,
        idx_i_sr=idx_i_sr,
        idx_j_sr=idx_j_sr,
        rbf=rbf,
        atom_embedding=atom_embedding,
        charge_embedding=None,
        spin_embedding=None,
    )
    assert "atom_feature" in out
    atom_feature = out["atom_feature"]
    assert atom_feature.ndim == 2
    assert atom_feature.shape[0] == N


def test_hierachical_nse_readout_positive_fa_3d_atom_feature():
    from enerzyme.models.layers.readout import HierachicalNSEReadout

    torch.manual_seed(0)
    N = 6
    dim_embedding = 12
    num_blocks = 4
    atom_feature = torch.randn(N, dim_embedding, num_blocks)

    readout = HierachicalNSEReadout(
        num_blocks=num_blocks,
        dim_embedding=dim_embedding,
        head_type="dense",
        positive_activation_fn="softplus",
    )
    out = readout.get_output(atom_feature=atom_feature)
    assert set(out.keys()) == {"Qa_alpha_tilde", "Qa_beta_tilde", "fa_alpha", "fa_beta"}
    assert out["Qa_alpha_tilde"].shape == (N,)
    assert out["Qa_beta_tilde"].shape == (N,)
    assert out["fa_alpha"].shape == (N,)
    assert out["fa_beta"].shape == (N,)
    assert torch.all(out["fa_alpha"] >= 0)
    assert torch.all(out["fa_beta"] >= 0)

