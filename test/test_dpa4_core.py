"""DPA4 Core smoke, geometry-gradient, registration, and YAML tests."""

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "dpa4_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return (torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs]))


def _random_so3(dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _core(**kwargs):
    from enerzyme.models.dpa4 import DPA4Core

    params = dict(
        dim_embedding=8, channels=8, lmax=2, mmax=1, n_blocks=1,
        mixing_layers=1, n_radial=8, ffn_neurons=16,
    )
    params.update(kwargs)
    return DPA4Core(**params)


def test_dpa4_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("DPA4")
    assert core_cls.__name__ == "DPA4Core"
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"], layer_params=ff["layers"],
        build_params=ff["build_params"], verbose=0,
    )
    assert model.__class__.__name__ == "DPA4Core"


def test_dpa4_shapes_and_geometry_autograd():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _core()
    out = core.get_output(
        torch.randn(n, 8), torch.tensor([1, 6, 8, 1]), idx_i, idx_j,
        positions[idx_i] - positions[idx_j], torch.zeros(n, dtype=torch.long),
    )
    assert out["atom_feature"].shape == (n, 8)
    assert out["atom_sphere_feature"].shape == (n, 9, 8)
    assert core.feature_irreps == "8x0e"
    grad = torch.autograd.grad(out["atom_feature"].square().sum(), positions)[0]
    assert torch.isfinite(grad).all()


def test_dpa4_wigner_blocks_are_orthogonal_and_differentiable():
    from enerzyme.models.dpa4.wignerd import WignerDCalculator, quaternion_normalize

    q = quaternion_normalize(torch.randn(5, 4, dtype=torch.float64, requires_grad=True))
    D, Dt = WignerDCalculator(2).double()(q)
    eye = torch.eye(9, dtype=torch.float64).expand(5, -1, -1)
    torch.testing.assert_close(D @ Dt, eye, atol=1e-6, rtol=1e-6)
    grad = torch.autograd.grad(D.square().sum(), q)[0]
    assert torch.isfinite(grad).all()


def test_dpa4_layer_stack_energy_and_force():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"}, {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {
            "channels": 8, "lmax": 2, "mmax": 1, "n_blocks": 1,
            "mixing_layers": 1, "n_radial": 8, "ffn_neurons": 16,
        }},
        {"name": "SimpleReadout", "params": {
            "output_fields": ["Ea"], "head_type": "dense", "keep_feature": False,
        }},
        {"name": "EnergyReduce"}, {"name": "Force"},
    ]
    model = build_model(
        "dpa4", layer_params=layers,
        build_params={"dim_embedding": 8, "max_Za": 20, "cutoff_sr": 6.0,
                      "cutoff_fn": "polynomial"}, verbose=0,
    )
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    out = model({
        "Ra": positions, "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "idx_i": idx_i, "idx_j": idx_j,
    })
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()


def test_dpa4_so3_scalar_invariance():
    torch.manual_seed(0)
    dtype = torch.float64
    core = _core().to(dtype)
    core.eval()
    n = 4
    idx_i, idx_j = _edges(n)
    atom_embedding = torch.randn(n, 8, dtype=dtype)
    Za = torch.tensor([1, 6, 8, 1])
    vij = torch.randn(idx_i.shape[0], 3, dtype=dtype)
    R = _random_so3(dtype)
    vij_rot = vij @ R.T
    batch_seg = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        out0 = core.get_output(atom_embedding, Za, idx_i, idx_j, vij, batch_seg)
        out1 = core.get_output(atom_embedding, Za, idx_i, idx_j, vij_rot, batch_seg)
    assert_allclose(
        out0["atom_feature"].numpy(),
        out1["atom_feature"].numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


def test_dpa4_force_finite_difference_conservation():
    """Fa from ForceLayer must match central finite differences of E."""
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    layers = [
        {"name": "RangeSeparation"},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "channels": 8,
                "lmax": 2,
                "mmax": 1,
                "n_blocks": 1,
                "mixing_layers": 1,
                "n_radial": 8,
                "ffn_neurons": 16,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "keep_feature": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "dpa4",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "max_Za": 20,
            "cutoff_sr": 6.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    ).double()
    model.eval()
    n = 4
    idx_i, idx_j = _edges(n)
    Ra = (torch.randn(n, 3, dtype=torch.float64) * 0.4).requires_grad_(True)
    Za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(n, dtype=torch.long)
    batch = {
        "Ra": Ra,
        "Za": Za,
        "batch_seg": batch_seg,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    fa = out["Fa"].detach()
    assert torch.isfinite(fa).all()
    assert fa.abs().sum() > 0

    eps = 1e-4
    fd = torch.zeros_like(fa)
    base = Ra.detach().clone()
    for i in range(n):
        for d in range(3):
            rp = base.clone()
            rm = base.clone()
            rp[i, d] += eps
            rm[i, d] -= eps
            rp = rp.requires_grad_(True)
            rm = rm.requires_grad_(True)
            ep = model(
                {"Ra": rp, "Za": Za, "batch_seg": batch_seg, "idx_i": idx_i, "idx_j": idx_j}
            )["E"]
            em = model(
                {"Ra": rm, "Za": Za, "batch_seg": batch_seg, "idx_i": idx_i, "idx_j": idx_j}
            )["E"]
            fd[i, d] = -(ep - em) / (2 * eps)
    torch.testing.assert_close(fa, fd, atol=5e-3, rtol=5e-3)
