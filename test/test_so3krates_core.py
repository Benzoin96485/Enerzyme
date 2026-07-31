"""So3krates Core smoke / equivariance / build_model tests."""

from __future__ import annotations

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


def test_so3krates_core_atom_feature_shape():
    from enerzyme.models.so3krates import So3kratesCore

    torch.manual_seed(0)
    N = 6
    F = 12
    degrees = [1, 2, 3]
    m_tot = sum(2 * l + 1 for l in degrees)
    core = So3kratesCore(
        dim_embedding=F,
        num_rbf=8,
        degrees=degrees,
        num_features=F,
        num_heads=3,
        num_layers=2,
        avg_num_neighbors=4.0,
    )
    idx_i, idx_j = _complete_graph_edges(N)
    P = idx_i.shape[0]
    out = core.get_output(
        atom_embedding=torch.randn(N, F),
        rbf=torch.randn(P, 8),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(P, 3),
        Dij_sr=torch.rand(P) * 4 + 0.1,
    )
    assert out["atom_feature"].shape == (N, F)
    assert out["atom_sphere_feature"].shape == (N, m_tot)
    assert core.feature_irreps == f"{F}x0e"
    assert core.dim_feature_out == F


def test_so3krates_simple_readout_resolves_feature_irreps():
    from enerzyme.models.so3krates import So3kratesCore
    from enerzyme.models.layers import SimpleReadout

    F = 12
    core = So3kratesCore(
        dim_embedding=F,
        num_rbf=8,
        degrees=[1, 2, 3],
        num_features=F,
        num_heads=3,
        num_layers=1,
        avg_num_neighbors=4.0,
    )
    ro = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="dense",
    )
    assert ro.feature_irreps == "12x0e"
    assert ro.dim_feature_in == 12
    out = ro.get_output(torch.randn(4, F))
    assert out["Ea"].shape == (4,)


def test_so3krates_build_model_energy_force_finite():
    from enerzyme.models.ff import build_model
    from enerzyme.models.so3krates.core import DEFAULT_LAYER_PARAMS

    torch.manual_seed(0)
    # Smaller stack for speed
    layers = []
    for item in DEFAULT_LAYER_PARAMS:
        item = dict(item)
        if item["name"] == "Core":
            item["params"] = {
                "degrees": [1, 2, 3],
                "num_features": 12,
                "num_heads": 3,
                "num_layers": 2,
                "avg_num_neighbors": 4.0,
            }
        layers.append(item)
    model = build_model(
        "so3krates",
        layer_params=layers,
        build_params={
            "dim_embedding": 12,
            "num_rbf": 8,
            "max_Za": 94,
            "cutoff_sr": 5.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    )
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
    assert torch.isfinite(out["E"]).all()
    assert out["Fa"].shape == (N, 3)
    assert torch.isfinite(out["Fa"]).all()
    out["E"].sum().backward()
    assert Ra.grad is not None
    assert torch.isfinite(Ra.grad).all()


def test_so3krates_energy_so3_invariant_forces_equivariant():
    from enerzyme.models.ff import build_model
    from enerzyme.models.so3krates.core import DEFAULT_LAYER_PARAMS

    torch.manual_seed(1)
    layers = []
    for item in DEFAULT_LAYER_PARAMS:
        item = dict(item)
        if item["name"] == "Core":
            item["params"] = {
                "degrees": [1, 2, 3],
                "num_features": 12,
                "num_heads": 3,
                "num_layers": 2,
                "avg_num_neighbors": 4.0,
            }
        layers.append(item)
    model = build_model(
        "so3krates",
        layer_params=layers,
        build_params={
            "dim_embedding": 12,
            "num_rbf": 8,
            "max_Za": 94,
            "cutoff_sr": 5.0,
            "cutoff_fn": "cosine",
        },
        verbose=0,
    )
    model.eval()
    N = 5
    Ra = torch.randn(N, 3, dtype=torch.float64)
    Za = torch.tensor([1, 6, 8, 1, 6])
    batch_seg = torch.zeros(N, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(N)
    R = _random_so3(dtype=torch.float64)
    Ra_rot = Ra @ R.T

    def run(pos):
        pos = pos.clone().requires_grad_(True)
        out = model(
            {
                "Ra": pos,
                "Za": Za,
                "batch_seg": batch_seg,
                "idx_i": idx_i,
                "idx_j": idx_j,
            }
        )
        return out["E"].double(), out["Fa"].double()

    # Rebuild model in float64 for tighter equivariance check
    for p in model.parameters():
        p.data = p.data.double()

    E0, F0 = run(Ra)
    E1, F1 = run(Ra_rot)
    assert_allclose(E0.detach().numpy(), E1.detach().numpy(), atol=1e-5, rtol=1e-5)
    F0_rot = F0 @ R.T
    assert_allclose(F0_rot.detach().numpy(), F1.detach().numpy(), atol=1e-4, rtol=1e-4)
