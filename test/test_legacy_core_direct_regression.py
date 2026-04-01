"""
Regression: legacy cores in default *direct* mode (no output_mode in config / constructor default).

Ensures unchanged YAML-style defaults still expose Ea/Qa (+ nh_loss for PhysNet) and that
full default stacks build, forward, and backpropagate.
"""
import sys

import numpy as np
import pytest
import torch

sys.path.extend(["..", "."])

from enerzyme.tasks.batch import _decorate_batch_input


def _rand_graph_indices(num_nodes: int, num_edges: int, device=None):
    idx_i = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    idx_j = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    return idx_i, idx_j


def _smoke_features(n_atoms: int = 8, seed: int = 0, with_charge_spin: bool = False):
    rng = np.random.default_rng(seed)
    feat = {
        "Ra": rng.random((n_atoms, 3), dtype=np.float64) * 3.0,
        "Za": rng.integers(1, 9, size=(n_atoms,), dtype=np.int64),
        "N": n_atoms,
    }
    if with_charge_spin:
        feat["Q"] = 0.0
        feat["S"] = 0.0
    return feat


# --- Core-level: explicit direct / default initializer -------------------------------------------------


def test_physnet_core_direct_default_matches_explicit():
    from enerzyme.models.physnet.core import PhysNetCore

    dim_embedding = 16
    num_rbf = 8
    num_blocks = 3
    c_default = PhysNetCore(dim_embedding=dim_embedding, num_rbf=num_rbf, num_blocks=num_blocks).float()
    c_direct = PhysNetCore(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        num_blocks=num_blocks,
        output_mode="direct",
    ).float()
    assert c_default._output_fields == {"Ea", "Qa", "nh_loss"}
    assert c_default._output_fields == c_direct._output_fields

    torch.manual_seed(0)
    N = 7
    num_edges = 24
    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    atom_embedding = torch.randn(N, dim_embedding, dtype=torch.float32)
    rbf = torch.randn(num_edges, num_rbf, dtype=torch.float32)
    out = c_default.get_output(
        rbf=rbf, atom_embedding=atom_embedding, idx_i_sr=idx_i_sr, idx_j_sr=idx_j_sr
    )
    assert set(out.keys()) >= {"Ea", "Qa", "nh_loss"}
    for k in ("Ea", "Qa"):
        assert out[k].shape == (N,)
    assert out["nh_loss"].shape == ()
    loss = out["Ea"].sum() + out["Qa"].sum() + out["nh_loss"]
    loss.backward()


def test_schnet_core_direct_default():
    from enerzyme.models.schnet.core import SchNetCore

    hidden_channels = 32
    dim_embedding = 32
    num_rbf = 12
    core = SchNetCore(
        hidden_channels=hidden_channels,
        dim_embedding=dim_embedding,
        num_interactions=2,
        num_rbf=num_rbf,
        cutoff_sr=5.0,
    )
    assert core._output_fields == {"Ea", "Qa"}

    N = 9
    num_edges = 30
    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    Dij_sr = torch.rand(num_edges, dtype=torch.float32)
    rbf = torch.randn(num_edges, num_rbf, dtype=torch.float32)
    atom_embedding = torch.randn(N, hidden_channels, dtype=torch.float32)
    out = core.get_output(
        idx_i_sr=idx_i_sr,
        idx_j_sr=idx_j_sr,
        Dij_sr=Dij_sr,
        rbf=rbf,
        atom_embedding=atom_embedding,
    )
    assert set(out.keys()) == {"Ea", "Qa"}
    loss = out["Ea"].sum() + out["Qa"].sum()
    loss.backward()


def test_spookynet_core_direct_default():
    from enerzyme.models.spookynet.core import SpookyNetCore

    dim_embedding = 32
    num_rbf = 16
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
    )
    assert core._output_fields == {"Ea", "Qa"}

    N = 11
    num_edges = 40
    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    vij_sr = torch.randn(num_edges, 3, dtype=torch.float32)
    Dij_sr = vij_sr.norm(dim=-1).clamp_min(1e-6)
    rbf = torch.randn(num_edges, num_rbf, dtype=torch.float32)
    atom_embedding = torch.randn(N, dim_embedding, dtype=torch.float32)
    batch_seg = torch.zeros(N, dtype=torch.long)
    out = core.get_output(
        Dij_sr=Dij_sr,
        vij_sr=vij_sr,
        idx_i_sr=idx_i_sr,
        idx_j_sr=idx_j_sr,
        rbf=rbf,
        atom_embedding=atom_embedding,
        batch_seg=batch_seg,
    )
    assert set(out.keys()) == {"Ea", "Qa"}
    loss = out["Ea"].sum() + out["Qa"].sum()
    loss.backward()


def test_mace_core_direct_default():
    try:
        from enerzyme.models.mace.core import MACECore
    except Exception as e:
        pytest.skip(f"MACECore not importable: {e}")

    N = 8
    num_edges = 20
    max_Za = 10
    dim_embedding = 8
    num_rbf = 4
    max_ell = 1
    idx_i_sr, idx_j_sr = _rand_graph_indices(N, num_edges)
    vij_sr = torch.randn(num_edges, 3, dtype=torch.float32)
    rbf = torch.randn(num_edges, num_rbf, dtype=torch.float32)
    Za = torch.randint(1, max_Za + 1, (N,), dtype=torch.long)
    atom_embedding = torch.randn(N, dim_embedding, dtype=torch.float32)

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
    )
    assert core._output_fields == {"Ea", "Qa"}
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
    assert set(out.keys()) == {"Ea", "Qa"}
    loss = out["Ea"].sum() + out["Qa"].sum()
    loss.backward()


# --- Full stack: build_model(DEFAULT_LAYER_PARAMS) without output_mode in YAML -----------------------


def test_physnet_build_default_layers_forward_backward():
    from enerzyme.models.ff import build_model
    from enerzyme.models.physnet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

    model = build_model(
        "PhysNet",
        layer_params=DEFAULT_LAYER_PARAMS,
        build_params=DEFAULT_BUILD_PARAMS,
        verbose=0,
    ).double()
    model.train()
    feat = _smoke_features(n_atoms=6, seed=1, with_charge_spin=False)
    batch = [(feat, None)]
    net_input, _ = _decorate_batch_input(
        batch, dtype=torch.float64, device=torch.device("cpu"), otf_graph=True
    )
    out = model(net_input)
    assert "E" in out
    out["E"].sum().backward()


def test_schnet_build_default_layers_forward_backward():
    from enerzyme.models.ff import build_model
    from enerzyme.models.schnet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

    model = build_model(
        "SchNet",
        layer_params=DEFAULT_LAYER_PARAMS,
        build_params=DEFAULT_BUILD_PARAMS,
        verbose=0,
    ).double()
    model.train()
    feat = _smoke_features(n_atoms=6, seed=2, with_charge_spin=False)
    batch = [(feat, None)]
    net_input, _ = _decorate_batch_input(
        batch, dtype=torch.float64, device=torch.device("cpu"), otf_graph=True
    )
    out = model(net_input)
    assert "E" in out
    out["E"].sum().backward()


def test_spookynet_build_default_layers_forward_backward():
    from enerzyme.models.ff import build_model
    from enerzyme.models.spookynet import DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

    model = build_model(
        "SpookyNet",
        layer_params=DEFAULT_LAYER_PARAMS,
        build_params=DEFAULT_BUILD_PARAMS,
        verbose=0,
    ).double()
    model.train()
    feat = _smoke_features(n_atoms=6, seed=3, with_charge_spin=True)
    batch = [(feat, None)]
    net_input, _ = _decorate_batch_input(
        batch, dtype=torch.float64, device=torch.device("cpu"), otf_graph=True
    )
    out = model(net_input)
    assert "E" in out
    out["E"].sum().backward()
