"""Native eSEN Core: shapes, smooth cutoff, and conservative forces."""

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


def _tiny_core(**overrides):
    from enerzyme.models.esen.native import eSENCore

    params = dict(
        dim_embedding=8,
        num_rbf=4,
        max_Za=8,
        sphere_channels=8,
        hidden_channels=8,
        edge_channels=8,
        lmax=2,
        mmax=1,
        num_layers=2,
        cutoff_sr=6.0,
    )
    params.update(overrides)
    return eSENCore(**params)


def _pair_inputs(core, vij, rbf_scale=1.0):
    n_edge = vij.shape[0]
    return dict(
        atom_embedding=torch.randn(2, core.dim_embedding, dtype=vij.dtype),
        Za=torch.tensor([1, 6]),
        rbf=torch.randn(n_edge, core.num_rbf, dtype=vij.dtype) * rbf_scale,
        idx_i_sr=torch.tensor([0]),
        idx_j_sr=torch.tensor([1]),
        vij_sr=vij,
    )


def test_esen_gaussian_and_bessel_flavors():
    from enerzyme.models.cutoff import polynomial_envelope
    from enerzyme.models.layers.rbf import BesselRBFLayer, GaussianRBFLayer

    cutoff = 6.0
    num = 8
    dist = torch.tensor([0.0, 1.5, 6.0, 7.0])
    gaussian = GaussianRBFLayer(num, cutoff, apply_cutoff_fn=False, flavor="esen")
    center = torch.linspace(0, cutoff, num)
    delta = (center[1] - center[0]).item()
    coeff = -0.5 / (2.0 * delta) ** 2
    ref = torch.exp(coeff * (dist.view(-1, 1) - center) ** 2)
    assert_allclose(gaussian._get_rbf(dist).detach(), ref, rtol=1e-6, atol=1e-6)

    bessel = BesselRBFLayer(
        num, cutoff, flavor="esen", trainable=False, envelope_exponent=5
    )
    assert bessel.apply_cutoff_fn is False
    scaled = dist.view(-1, 1) / cutoff
    env = polynomial_envelope(scaled, 1.0, 5)
    freq = torch.arange(1, num + 1, dtype=dist.dtype) * torch.pi
    arg = scaled * freq.view(1, -1)
    safe = scaled.clamp_min(1e-12)
    quotient = torch.where(scaled == 0, freq.view(1, -1), torch.sin(arg) / safe)
    ref_b = env * (2.0 / cutoff**3) ** 0.5 * quotient
    assert_allclose(bessel._get_rbf(dist).detach(), ref_b, rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(bessel._get_rbf(dist)[-1]) == 0


def test_esen_core_shapes_and_readout():
    from enerzyme.models.layers import SimpleReadout

    torch.manual_seed(0)
    core = _tiny_core()
    n = 5
    idx_i, idx_j = _complete_graph_edges(n)
    out = core.get_output(
        atom_embedding=torch.randn(n, core.dim_embedding),
        Za=torch.tensor([1, 6, 7, 8, 1]),
        rbf=torch.randn(idx_i.shape[0], core.num_rbf),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
    )
    assert out["atom_feature"].shape == (n, core.sphere_channels)
    assert out["atom_sphere_feature"].shape == (
        n,
        (core.lmax_list[0] + 1) ** 2,
        core.sphere_channels,
    )
    assert core.feature_irreps == f"{core.sphere_channels}x0e"
    readout = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[core],
        head_type="mlp",
        num_hidden_layers=2,
        use_bias_out=True,
        activation_fn="swish",
    )
    energy = readout.get_output(out["atom_feature"])
    assert energy["Ea"].shape == (n,)


def test_envelope_matches_no_neighbor_limit():
    """A message at the cutoff is identically zero, so the last neighbor can leave."""
    torch.manual_seed(0)
    core = _tiny_core().double()
    cutoff = core.cutoff_sr
    at_cutoff = _pair_inputs(
        core, torch.tensor([[cutoff, 0.0, 0.0]], dtype=torch.float64)
    )
    # Same weights and the same node inputs, with the edge removed.
    isolated = dict(at_cutoff)
    isolated["rbf"] = at_cutoff["rbf"].new_zeros((0, core.num_rbf))
    isolated["idx_i_sr"] = torch.zeros(0, dtype=torch.long)
    isolated["idx_j_sr"] = torch.zeros(0, dtype=torch.long)
    isolated["vij_sr"] = at_cutoff["vij_sr"].new_zeros((0, 3))
    with_edge = core.get_output(**at_cutoff)["atom_sphere_feature"]
    without = core.get_output(**isolated)["atom_sphere_feature"]
    assert_allclose(with_edge.detach(), without.detach(), rtol=1e-6, atol=1e-6)

    inside = dict(at_cutoff)
    inside["vij_sr"] = torch.tensor([[cutoff * 0.5, 0.0, 0.0]], dtype=torch.float64)
    nearer = core.get_output(**inside)["atom_sphere_feature"]
    assert not torch.allclose(nearer, without)


def test_esen_build_model_force_matches_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    model = build_model("esen", verbose=0).double()
    model.eval()
    n = 4
    ra = (torch.randn(n, 3, dtype=torch.float64) * 0.3).requires_grad_(True)
    za = torch.tensor([1, 6, 8, 1])
    batch_seg = torch.zeros(n, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(n)
    batch = {
        "Ra": ra,
        "Za": za,
        "batch_seg": batch_seg,
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    assert out["atom_feature"].shape == (n, model.dim_feature_out)
    assert torch.isfinite(out["E"]).all()
    assert out["Fa"].shape == (n, 3)
    fa = out["Fa"].detach()

    eps = 1e-4
    fd = torch.zeros_like(fa)
    base = ra.detach().clone()
    for i in range(n):
        for axis in range(3):
            plus = base.clone()
            minus = base.clone()
            plus[i, axis] += eps
            minus[i, axis] -= eps
            ep = model(
                {
                    "Ra": plus.requires_grad_(True),
                    "Za": za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            em = model(
                {
                    "Ra": minus.requires_grad_(True),
                    "Za": za,
                    "batch_seg": batch_seg,
                    "idx_i": idx_i,
                    "idx_j": idx_j,
                }
            )["E"]
            fd[i, axis] = -(ep.detach() - em.detach()).sum() / (2 * eps)
    assert_allclose(fa, fd, rtol=1e-4, atol=1e-4)


def test_esen_energy_invariant_forces_equivariant():
    from enerzyme.models.ff import build_model

    torch.manual_seed(1)
    model = build_model("esen", verbose=0).double()
    model.eval()
    n = 4
    ra = torch.randn(n, 3, dtype=torch.float64) * 0.4
    za = torch.tensor([1, 6, 7, 8])
    batch_seg = torch.zeros(n, dtype=torch.long)
    idx_i, idx_j = _complete_graph_edges(n)
    rotation = _random_so3()
    ra_rot = ra @ rotation.T
    kwargs = {"Za": za, "batch_seg": batch_seg, "idx_i": idx_i, "idx_j": idx_j}
    out = model({"Ra": ra.requires_grad_(True), **kwargs})
    out_rot = model({"Ra": ra_rot.requires_grad_(True), **kwargs})
    assert_allclose(out["E"].detach(), out_rot["E"].detach(), rtol=1e-5, atol=1e-5)
    fa_rot = out["Fa"].detach() @ rotation.T
    assert_allclose(fa_rot, out_rot["Fa"].detach(), rtol=1e-5, atol=1e-5)


def test_esen_grid_ablation_runs():
    torch.manual_seed(0)
    core = _tiny_core(lmax=1, mmax=1, num_layers=1, mlp_type="grid", act_type="s2")
    idx_i, idx_j = _complete_graph_edges(3)
    out = core.get_output(
        atom_embedding=torch.randn(3, core.dim_embedding),
        Za=torch.tensor([1, 6, 8]),
        rbf=torch.randn(idx_i.shape[0], core.num_rbf),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=torch.randn(idx_i.shape[0], 3),
    )
    assert out["atom_feature"].shape == (3, core.sphere_channels)
    assert torch.isfinite(out["atom_sphere_feature"]).all()


def test_native_esen_does_not_import_fairchem_source():
    from pathlib import Path

    root = Path("enerzyme/models/esen")
    text = "\n".join(path.read_text() for path in (root / "native").rglob("*.py"))
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "fairchem" not in stripped
    init = (root / "__init__.py").read_text()
    assert "UMAWrapperQS" in init
    assert "eSENCore" in init


def test_get_ff_core_registers_esen_and_keeps_uma():
    from enerzyme.models.ff import get_ff_core

    core, _, layers = get_ff_core("esen")
    assert core.__name__ == "eSENCore"
    assert any(layer["name"] == "Force" for layer in layers)
    src = __import__("pathlib").Path("enerzyme/models/ff.py").read_text()
    assert 'architecture.lower() == "uma_qs"' in src
    assert "UMAWrapperQS" in src
