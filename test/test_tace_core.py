"""TACE Core smoke tests (spherical + Cartesian)."""

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPH = ROOT / "enerzyme" / "config" / "tace_layers_example.yaml"
EXAMPLE_CART = ROOT / "enerzyme" / "config" / "tace_cartesian_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.long),
        torch.tensor([p[1] for p in pairs], dtype=torch.long),
    )


def _tiny_core(tensor_basis: str, **kwargs):
    from enerzyme.models.tace import TACECore

    params = dict(
        max_Za=10,
        dim_embedding=8,
        num_rbf=4,
        num_channel=8,
        num_layers=2,
        Lmax=1,
        lmax=2,
        correlation=2,
        avg_num_neighbors=4.0,
        tensor_basis=tensor_basis,
        edge_embedding="identity",
        edge_update="identity",
        radial_mlp=[16],
        use_first_resnet=False,
    )
    if tensor_basis == "spherical":
        params["nonlinear"] = "sigmoid_gate"
        params["parity"] = False
    params.update(kwargs)
    return TACECore(**params)


def test_tace_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("tace")
    assert core_cls.__name__ == "TACECore"
    for path in (EXAMPLE_SPH, EXAMPLE_CART):
        with path.open() as stream:
            ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
        # shrink for CI
        layers = []
        for layer in ff["layers"]:
            layer = dict(layer)
            if layer.get("name") == "Core":
                params = dict(layer.get("params") or {})
                params.update(
                    {
                        "num_channel": 8,
                        "num_layers": 1,
                        "Lmax": 1,
                        "lmax": 1,
                        "correlation": 2,
                        "radial_mlp": [16],
                        "edge_embedding": "identity",
                        "edge_update": "identity",
                    }
                )
                layer["params"] = params
            layers.append(layer)
        bp = dict(ff["build_params"])
        bp.update({"dim_embedding": 8, "num_rbf": 4, "max_Za": 20})
        model = build_model(
            ff["architecture"], layer_params=layers, build_params=bp, verbose=0
        )
        assert model.__class__.__name__ == "TACECore"
        assert model.dim_feature_out == 8


def test_tace_spherical_feature_shape_and_grad():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _tiny_core("spherical")
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=positions[idx_i] - positions[idx_j],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    feat = out["atom_feature"]
    assert feat.ndim == 2 and feat.shape[0] == n
    assert core.dim_feature_out == 8
    assert "0e" in core.feature_irreps
    grad = torch.autograd.grad(feat.square().sum(), positions)[0]
    assert torch.isfinite(grad).all()


def test_tace_cartesian_feature_shape_and_grad():
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    core = _tiny_core("cartesian")
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=positions[idx_i] - positions[idx_j],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    feat = out["atom_feature"]
    assert feat.shape == (n, 8)
    assert core.feature_irreps == "8x0e"
    grad = torch.autograd.grad(feat.square().sum(), positions)[0]
    assert torch.isfinite(grad).all()


def test_tace_layer_stack_energy_and_force_spherical():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "tensor_basis": "spherical",
                "num_layers": 1,
                "num_channel": 8,
                "Lmax": 1,
                "lmax": 1,
                "correlation": 2,
                "avg_num_neighbors": 4.0,
                "edge_embedding": "identity",
                "edge_update": "identity",
                "nonlinear": "sigmoid_gate",
                "radial_mlp": [16],
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
        "tace",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 6.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    out = model(
        {
            "Ra": positions,
            "Za": torch.tensor([1, 6, 8, 1]),
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()


def test_tace_layer_stack_energy_and_force_cartesian():
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {"name": "BesselRBF", "params": {"trainable": False}},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "tensor_basis": "cartesian",
                "num_layers": 1,
                "num_channel": 8,
                "Lmax": 1,
                "lmax": 1,
                "correlation": 2,
                "avg_num_neighbors": 4.0,
                "edge_embedding": "identity",
                "edge_update": "identity",
                "radial_mlp": [16],
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
        "tace",
        layer_params=layers,
        build_params={
            "dim_embedding": 8,
            "num_rbf": 4,
            "max_Za": 20,
            "cutoff_sr": 6.0,
            "cutoff_fn": "polynomial",
        },
        verbose=0,
    )
    n = 4
    idx_i, idx_j = _edges(n)
    positions = torch.randn(n, 3, requires_grad=True)
    out = model(
        {
            "Ra": positions,
            "Za": torch.tensor([1, 6, 8, 1]),
            "batch_seg": torch.zeros(n, dtype=torch.long),
            "idx_i": idx_i,
            "idx_j": idx_j,
        }
    )
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    assert torch.isfinite(out["E"]).all() and torch.isfinite(out["Fa"]).all()


def test_tace_spherical_parity_true_bb_residual():
    """Regression: parity=True makes TP(node,SH) != TP(msg,msg); BB skip must match product."""
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    core = _tiny_core(
        "spherical",
        num_layers=3,
        Lmax=2,
        lmax=3,
        parity=True,
        use_first_resnet=True,
        resnet_type="BB",
        resnet_linear_type="aware",
    )
    for inter, prod in zip(core.interactions, core.products):
        if hasattr(inter, "resnetBB"):
            assert inter.irreps_sc == prod.irreps_out
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    assert torch.isfinite(out["atom_feature"]).all()


def test_tace_cartesian_identity_resnet():
    """Regression: cartesian must honor resnet_linear_type='identity' (no learned skip)."""
    from enerzyme.models.tace.cartesian.core_blocks import DictSkipIdentity

    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    core = _tiny_core(
        "cartesian",
        num_layers=2,
        use_first_resnet=True,
        resnet_linear_type="identity",
    )
    resnets = [r for r in core.cartesian_stack.resnets if r is not None]
    assert resnets, "expected residual modules"
    assert all(isinstance(r, DictSkipIdentity) for r in resnets)
    out = core.get_output(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    assert out["atom_feature"].shape == (n, 8)
    assert torch.isfinite(out["atom_feature"]).all()


def _forward_core(core, n=4, cutoff=None):
    idx_i, idx_j = _edges(n)
    kwargs = dict(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    if cutoff is not None:
        kwargs["cutoff_values_sr"] = (
            torch.full((idx_i.numel(),), float(cutoff))
            if isinstance(cutoff, (int, float))
            else cutoff
        )
    return core.get_output(**kwargs)


def test_tace_shared_knobs_honored_on_both_backends():
    """Contract: shared Core knobs must not be silently ignored by either backend."""
    import pytest

    for basis in ("spherical", "cartesian"):
        with pytest.raises(ValueError, match="scatter_norm"):
            _tiny_core(basis, scatter_norm="bogus")

    # density scatter_norm wires modules and runs on both backends
    for basis, mode in (
        ("spherical", "density"),
        ("spherical", "no_cutoff_density"),
        ("cartesian", "density"),
        ("cartesian", "no_cutoff_density"),
    ):
        core = _tiny_core(basis, scatter_norm=mode, num_layers=1)
        if basis == "spherical":
            assert hasattr(core.interactions[0], "edge_density")
        else:
            assert len(core.cartesian_stack.edge_densities) == 1
        feat = _forward_core(core)["atom_feature"]
        assert torch.isfinite(feat).all()

    # non-BB resnet_type disables residuals on both backends
    for basis in ("spherical", "cartesian"):
        core = _tiny_core(
            basis,
            resnet_type="none",
            use_first_resnet=True,
            num_layers=2,
        )
        if basis == "spherical":
            assert all(not hasattr(inter, "resnetBB") for inter in core.interactions)
        else:
            assert all(r is None for r in core.cartesian_stack.resnets)
        assert torch.isfinite(_forward_core(core)["atom_feature"]).all()

    # BB + use_first_resnet enables residuals on both backends
    for basis in ("spherical", "cartesian"):
        core = _tiny_core(
            basis,
            resnet_type="BB",
            use_first_resnet=True,
            num_layers=2,
        )
        if basis == "spherical":
            assert hasattr(core.interactions[0], "resnetBB")
            assert hasattr(core.interactions[1], "resnetBB")
        else:
            assert core.cartesian_stack.resnets[0] is not None
            assert core.cartesian_stack.resnets[1] is not None
        assert torch.isfinite(_forward_core(core)["atom_feature"]).all()


def test_tace_cartesian_density_changes_features_vs_avg():
    """density path must actually alter messages relative to avg_num_neighbors."""
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    kwargs = dict(
        Za=torch.tensor([1, 6, 8, 1]),
        vij_sr=torch.randn(idx_i.numel(), 3),
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 4),
        atom_embedding=torch.randn(n, 8),
    )
    avg = _tiny_core("cartesian", scatter_norm="avg_num_neighbors", num_layers=1)
    dens = _tiny_core("cartesian", scatter_norm="density", num_layers=1)
    dens.load_state_dict(avg.state_dict(), strict=False)
    # Force density MLP away from zero so normalization differs from constant avg.
    with torch.no_grad():
        for p in dens.cartesian_stack.edge_densities[0].parameters():
            p.fill_(0.2)
        dens.cartesian_stack.density_betas[0].fill_(1.0)
    out_avg = avg.get_output(**kwargs)["atom_feature"]
    out_dens = dens.get_output(**kwargs)["atom_feature"]
    assert torch.isfinite(out_dens).all()
    assert not torch.allclose(out_avg, out_dens)


def test_tace_element2_post_mlp_cutoff_kills_edge_messages():
    """Regression: element2 embeds are not in RBF; post-MLP cutoff must zero messages."""
    torch.manual_seed(0)
    n = 4
    idx_i, idx_j = _edges(n)
    e = idx_i.numel()
    Za = torch.tensor([1, 6, 8, 1])
    atom_embedding = torch.randn(n, 8)
    rbf = torch.randn(e, 4)  # nonzero even if envelope was forgotten upstream

    for basis in ("spherical", "cartesian"):
        core = _tiny_core(
            basis,
            num_layers=1,
            edge_embedding="nonlinear" if basis == "spherical" else "identity",
            edge_update="element2",
            use_first_resnet=False,
        )
        base = dict(
            Za=Za,
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            rbf=rbf,
            atom_embedding=atom_embedding,
        )
        feat_a = core.get_output(
            vij_sr=torch.randn(e, 3),
            cutoff_values_sr=torch.zeros(e),
            **base,
        )["atom_feature"]
        feat_b = core.get_output(
            vij_sr=torch.randn(e, 3),
            cutoff_values_sr=torch.zeros(e),
            **base,
        )["atom_feature"]
        # Different edges/directions but cutoff=0 ⇒ identical node-only features
        assert torch.allclose(feat_a, feat_b, atol=1e-6), basis

        feat_on = core.get_output(
            vij_sr=torch.randn(e, 3),
            cutoff_values_sr=torch.ones(e),
            **base,
        )["atom_feature"]
        assert not torch.allclose(feat_a, feat_on, atol=1e-5), basis
        assert torch.isfinite(feat_on).all()
