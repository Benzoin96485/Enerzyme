import sys

import torch
from numpy.testing import assert_allclose

sys.path.extend(["..", "."])


def _complete_graph_edges(num_nodes: int):
    idx_i, idx_j = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _random_so3(dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def test_escn_core_atom_feature_shape():
    from enerzyme.models.escn import eSCNCore

    torch.manual_seed(0)
    N = 6
    sphere_channels = 16
    dim_embedding = 16
    num_rbf = 8
    lmax = 2
    core = eSCNCore(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        sphere_channels=sphere_channels,
        hidden_channels=32,
        edge_channels=16,
        lmax=lmax,
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
    assert out["atom_sphere_feature"].shape == (N, (lmax + 1) ** 2, sphere_channels)
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


def test_init_edge_rot_mat_keeps_grad():
    from enerzyme.models.so3 import init_edge_rot_mat

    vij = torch.randn(5, 3, requires_grad=True)
    rot = init_edge_rot_mat(vij)
    assert rot.requires_grad
    (rot.sum()).backward()
    assert vij.grad is not None
    assert torch.isfinite(vij.grad).all()


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


def test_escn_force_finite_difference_conservation():
    """Fa from ForceLayer matches central finite differences of E."""
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("escn", verbose=0).double()
    model.eval()
    N = 4
    Ra = (torch.randn(N, 3, dtype=torch.float64) * 0.4).requires_grad_(True)
    Za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    batch = {
        "Ra": Ra,
        "Za": Za,
        "batch_seg": batch_seg,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    fa = out["Fa"].detach()

    eps = 1e-4
    fd = torch.zeros_like(fa)
    base = Ra.detach().clone()
    for i in range(N):
        for d in range(3):
            # ForceLayer needs Ra.requires_grad; take E only for the FD stencil.
            rp = base.clone()
            rm = base.clone()
            rp[i, d] += eps
            rm[i, d] -= eps
            rp = rp.requires_grad_(True)
            rm = rm.requires_grad_(True)
            ep = model(
                {
                    "Ra": rp,
                    "Za": Za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            em = model(
                {
                    "Ra": rm,
                    "Za": Za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            fd[i, d] = -(ep.detach() - em.detach()).sum() / (2 * eps)

    assert_allclose(fa.cpu().numpy(), fd.cpu().numpy(), atol=2e-3, rtol=2e-3)


def test_escn_energy_invariant_force_equivariant():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("escn", verbose=0).double()
    model.eval()
    N = 4
    Ra = torch.randn(N, 3, dtype=torch.float64) * 0.4
    Za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    q = _random_so3(torch.float64)

    Ra0 = Ra.clone().requires_grad_(True)
    out0 = model(
        {
            "Ra": Ra0,
            "Za": Za,
            "batch_seg": batch_seg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    e0 = out0["E"].detach()
    f0 = out0["Fa"].detach()

    Ra1 = (Ra @ q.T).clone().requires_grad_(True)
    out1 = model(
        {
            "Ra": Ra1,
            "Za": Za,
            "batch_seg": batch_seg,
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    e1 = out1["E"].detach()
    f1 = out1["Fa"].detach()

    assert_allclose(e0.cpu().numpy(), e1.cpu().numpy(), atol=1e-4, rtol=1e-4)
    f0_rot = f0 @ q.T
    assert_allclose(f0_rot.cpu().numpy(), f1.cpu().numpy(), atol=1e-3, rtol=1e-3)
