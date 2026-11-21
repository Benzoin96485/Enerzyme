"""DPA4C Core: registration, symmetry, calibration, and conservative forces."""

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "dpa4c_layers_example.yaml"


def _edges(n):
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    # Receiver is idx_i and sender is idx_j, matching DistanceLayer.
    return (
        torch.tensor([pair[0] for pair in pairs]),
        torch.tensor([pair[1] for pair in pairs]),
    )


def _rotation(dtype=torch.float64):
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=dtype))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _core(**kwargs):
    from enerzyme.models.dpa4c import DPA4CCore

    params = dict(dim_embedding=8, channels=8, lmax=2, radial_modes=0, n_radial=4)
    params.update(kwargs)
    return DPA4CCore(**params).double()


def _forward(core, embedding, positions, idx_i, idx_j):
    vij = positions[idx_j] - positions[idx_i]
    return core.get_output(embedding, idx_i, idx_j, vij)["atom_feature"]


def test_dpa4c_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core

    core_cls, _, _ = get_ff_core("dpa4c")
    assert core_cls.__name__ == "DPA4CCore"
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "DPA4CCore"
    assert model.dim_feature_out == 70


def test_dpa4c_shapes_grades_and_rejected_settings():
    from enerzyme.models.dpa4c import PUBLISHED_GRADES
    from enerzyme.models.dpa4c.geometry import invariant_width

    core = _core()
    assert core.dim_feature_out == invariant_width(8, 2)
    assert core.feature_irreps == "70x0e"
    for grade, params in PUBLISHED_GRADES.items():
        assert invariant_width(params["channels"], params["lmax"]) > 0
        if grade == "plus":
            continue
        built = _core(
            dim_embedding=params["channels"],
            channels=params["channels"],
            lmax=params["lmax"],
            radial_modes=params["radial_modes"],
        )
        assert built.dim_feature_out == invariant_width(params["channels"], params["lmax"])

    for kwargs in (
        {"channels": 12},
        {"lmax": 5},
        {"radial_modes": -1},
        {"dim_embedding": 16, "channels": 8},
        {"basis_type": "chebyshev"},
    ):
        try:
            _core(**kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"{kwargs} should be rejected")


def test_dpa4c_rotation_inversion_translation_and_edge_order():
    torch.manual_seed(0)
    core = _core().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [0.2, 1.3, 0.4], [-0.4, 0.3, 1.2]],
        dtype=torch.float64,
    )
    embedding = torch.randn(4, 8, dtype=torch.float64)
    idx_i, idx_j = _edges(4)
    reference = _forward(core, embedding, positions, idx_i, idx_j)

    rotated = positions @ _rotation().T
    torch.testing.assert_close(
        _forward(core, embedding, rotated, idx_i, idx_j),
        reference,
        atol=1e-8,
        rtol=1e-8,
    )
    improper = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64))
    reflected = positions @ improper.T
    torch.testing.assert_close(
        _forward(core, embedding, reflected, idx_i, idx_j),
        reference,
        atol=1e-8,
        rtol=1e-8,
    )
    torch.testing.assert_close(
        _forward(core, embedding, positions + 3.5, idx_i, idx_j),
        reference,
        atol=1e-8,
        rtol=1e-8,
    )
    order = torch.randperm(idx_i.shape[0])
    torch.testing.assert_close(
        _forward(core, embedding, positions, idx_i[order], idx_j[order]),
        reference,
        atol=1e-8,
        rtol=1e-8,
    )


def test_dpa4c_atom_permutation_batch_isolation_and_lmax():
    torch.manual_seed(1)
    core = _core().eval()
    positions = torch.randn(4, 3, dtype=torch.float64)
    embedding = torch.randn(4, 8, dtype=torch.float64)
    idx_i, idx_j = _edges(4)
    reference = _forward(core, embedding, positions, idx_i, idx_j)
    order = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(4)
    permuted_i = inverse[idx_i]
    permuted_j = inverse[idx_j]
    permuted = _forward(core, embedding[order], positions[order], permuted_i, permuted_j)
    torch.testing.assert_close(permuted[inverse], reference, atol=1e-8, rtol=1e-8)

    left = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]], dtype=torch.float64)
    right = torch.tensor([[5.0, 0.0, 0.0], [5.2, 1.1, 0.3]], dtype=torch.float64)
    both = torch.cat((left, right), dim=0)
    embed = torch.randn(4, 8, dtype=torch.float64)
    edges_i = torch.tensor([0, 1, 2, 3])
    edges_j = torch.tensor([1, 0, 3, 2])
    original = _forward(core, embed, both, edges_i, edges_j)
    moved = both.clone()
    moved[2:] += 4.0
    shifted = _forward(core, embed, moved, edges_i, edges_j)
    torch.testing.assert_close(shifted[:2], original[:2], atol=1e-8, rtol=1e-8)

    wider = _core(dim_embedding=8, channels=8, lmax=4, n_radial=4).eval()
    sample = _forward(wider, embedding, positions, idx_i, idx_j)
    assert sample.shape == (4, wider.dim_feature_out)
    assert torch.isfinite(sample).all()


def test_dpa4c_pair_modulation_and_radial_modes():
    torch.manual_seed(2)
    core = _core().eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.2, 0.0]], dtype=torch.float64)
    embedding = torch.tensor(
        [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float64,
    )
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    with torch.no_grad():
        core.radial.out_proj.weight.zero_()
        core.pair_film.out_proj.weight.zero_()
    # With both heads at zero, ψ = X_i + X_j and the two centers see the same sum.
    features = _forward(core, embedding, positions, idx_i, idx_j)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[0, :8], features[1, :8], atol=1e-8, rtol=1e-8)

    modes = _core(radial_modes=2).eval()
    base_embedding = torch.randn(3, 8, dtype=torch.float64)
    base_positions = torch.randn(3, 3, dtype=torch.float64)
    edges_i, edges_j = _edges(3)
    with torch.no_grad():
        modes.pair_film.out_proj.weight[16:].fill_(0.4)
        before = _forward(modes, base_embedding, base_positions, edges_i, edges_j)
        modes.radial.mode_head.weight.mul_(3)
        after = _forward(modes, base_embedding, base_positions, edges_i, edges_j)
    assert not torch.allclose(before, after)


def test_dpa4c_isolated_atom_and_cutoff_neighbor():
    core = _core().eval()
    embedding = torch.randn(2, 8, dtype=torch.float64)
    empty_i = torch.zeros(0, dtype=torch.long)
    empty_v = torch.zeros(0, 3, dtype=torch.float64)
    alone = core.get_output(embedding, empty_i, empty_i, empty_v)["atom_feature"]
    assert alone.shape == (2, 70)
    assert torch.isfinite(alone).all()

    positions = torch.tensor([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=torch.float64)
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    on_cutoff = _forward(core, embedding, positions, idx_i, idx_j)
    torch.testing.assert_close(on_cutoff, alone, atol=1e-8, rtol=1e-8)


def test_dpa4c_calibration_standardizes_masses_and_keeps_the_embedding():
    torch.manual_seed(3)
    core = _core().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.4, 0.2], [3.5, 0.0, 0.2]],
        dtype=torch.float64,
    )
    embedding = torch.randn(4, 8, dtype=torch.float64)
    idx_i, idx_j = _edges(4)
    vij = positions[idx_j] - positions[idx_i]
    raw = core.raw_descriptor(embedding, idx_i, idx_j, vij)
    core.fit_output_calibration(raw)
    calibrated = core.get_output(embedding, idx_i, idx_j, vij)["atom_feature"]
    torch.testing.assert_close(calibrated[:, -8:], embedding, atol=1e-8, rtol=1e-8)
    masses = calibrated[:, -10:-8].detach()
    assert_allclose(masses.mean(0).numpy(), [0, 0], atol=1e-6)
    core.reset_output_calibration()
    restored = core.get_output(embedding, idx_i, idx_j, vij)["atom_feature"]
    torch.testing.assert_close(restored, raw, atol=1e-8, rtol=1e-8)


def test_dpa4c_layer_stack_energy_force_and_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(0)
    layers = [
        {"name": "RangeSeparation"},
        {"name": "RandomAtomEmbedding"},
        {"name": "Core", "params": {
            "channels": 8, "lmax": 2, "radial_modes": 0, "n_radial": 4,
        }},
        {"name": "SimpleReadout", "params": {
            "output_fields": ["Ea"],
            "head_type": "mlp",
            "hidden_dims": [16, 16],
            "identity_residual": True,
            "use_bias_out": True,
            "activation_fn": "swish",
            "activation_params": {"initial_beta": 1.0, "learnable": False},
            "keep_feature": False,
        }},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    model = build_model(
        "dpa4c",
        layer_params=layers,
        build_params={"dim_embedding": 8, "max_Za": 20, "cutoff_sr": 6.0},
        verbose=0,
    ).double().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.05, 0.0], [0.1, 1.15, 0.1], [0.25, 0.2, 1.05]],
        dtype=torch.float64,
        requires_grad=True,
    )
    idx_i, idx_j = _edges(4)
    batch = {
        "Ra": positions,
        "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(4, dtype=torch.long),
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    output = model(batch)
    assert output["E"].shape == (1,)
    assert output["Fa"].shape == (4, 3)
    assert torch.isfinite(output["E"]).all() and torch.isfinite(output["Fa"]).all()
    assert output["Fa"].abs().sum() > 0
    assert_allclose(output["Fa"].sum(0).detach().numpy(), [0, 0, 0], atol=1e-8)

    shifted = {**batch, "Ra": (positions.detach() + 1.7).requires_grad_(True)}
    moved = model(shifted)["E"].detach()
    assert_allclose(output["E"].detach().numpy(), moved.numpy(), atol=1e-8)

    forces = output["Fa"].detach()
    step = 1e-5
    finite = torch.zeros_like(forces)
    base = positions.detach()
    for atom in range(4):
        for axis in range(3):
            plus = base.clone()
            minus = base.clone()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            plus.requires_grad_(True)
            minus.requires_grad_(True)
            energy_plus = model({**batch, "Ra": plus})["E"]
            energy_minus = model({**batch, "Ra": minus})["E"]
            finite[atom, axis] = -(energy_plus - energy_minus) / (2 * step)
    torch.testing.assert_close(forces, finite, atol=2e-4, rtol=2e-4)
