import sys

import pytest
import torch

sys.path.extend(["..", "."])


def _complete_graph_edges(num_nodes: int):
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def test_escn_core_atom_feature_shape():
    from enerzyme.models.escn import eSCNCore

    torch.manual_seed(0)
    N = 6
    sphere_channels = 16
    dim_embedding = 16
    num_rbf = 8
    core = eSCNCore(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        hidden_channels=32,
        edge_channels=16,
        lmax=2,
        mmax=1,
        num_layers=2,
    )
    idx_i, idx_j = _complete_graph_edges(N)
    atom_embedding = torch.randn(N, dim_embedding)
    Za = torch.tensor([1, 6, 7, 8, 1, 6])
    rbf = torch.randn(idx_i.shape[0], num_rbf)
    vij = torch.randn(idx_i.shape[0], 3)
    out = core.get_output(
        atom_embedding=atom_embedding,
        Za=Za,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
    )
    assert "atom_feature" in out
    assert out["atom_feature"].shape == (N, sphere_channels)
    assert "atom_sphere_feature" in out
    assert out["atom_sphere_feature"].shape[0] == N
    assert out["atom_sphere_feature"].shape[-1] == sphere_channels
    assert core.feature_irreps == f"{sphere_channels}x0e"
    assert core.dim_feature_out == sphere_channels


def test_escn_simple_readout_resolves_feature_irreps():
    from enerzyme.models.escn import eSCNCore
    from enerzyme.models.layers import SimpleReadout

    core = eSCNCore(
        dim_embedding=16,
        num_rbf=8,
        sphere_channels=16,
        hidden_channels=32,
        edge_channels=16,
        lmax=2,
        mmax=1,
        num_layers=1,
    )
    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="dense",
    )
    assert ro.feature_irreps == "16x0e"
    assert ro.dim_feature_in == 16
    out = ro.get_output(torch.randn(4, 16))
    assert out["Ea"].shape == (4,)


def test_escn_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("escn", verbose=0)
    N = 5
    Ra = torch.randn(N, 3, requires_grad=True)
    Za = torch.tensor([1, 6, 8, 1, 6])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    out = model(
        {
            "Ra": Ra,
            "Za": Za,
            "batch_seg": batch_seg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert out["atom_feature"].shape == (N, model.dim_feature_out)
    assert torch.isfinite(out["E"]).all()
    assert out["Fa"].shape == (N, 3)
    assert torch.isfinite(out["Fa"]).all()


def test_escn_energy_rotation_invariance():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("escn", verbose=0)
    model.eval()
    N = 4
    # ForceLayer needs grad on Ra even when we only compare energies
    Ra = torch.randn(N, 3, requires_grad=True)
    Za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)

    # Random rotation matrix
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    Ra_rot = (Ra.detach() @ q.T).requires_grad_(True)

    e0 = model(
        {"Ra": Ra, "Za": Za, "batch_seg": batch_seg, "idx_i": idx_i, "idx_j": idx_j}
    )["E"]
    e1 = model(
        {
            "Ra": Ra_rot,
            "Za": Za,
            "batch_seg": batch_seg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )["E"]
    assert torch.allclose(e0.detach(), e1.detach(), atol=1e-4, rtol=1e-4)
