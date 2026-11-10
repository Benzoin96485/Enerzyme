"""EquFlash Core: irrep schedule, registration, equivariance, energy/force stack."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "equflash_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotation(seed: int = 0) -> torch.Tensor:
    dtype = torch.get_default_dtype()
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=dtype))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _tiny_core(**overrides):
    from enerzyme.models.equflash import EquFlashCore

    params = dict(
        dim_embedding=4,
        num_rbf=3,
        lmax=1,
        num_layers=2,
        radial_hidden=[8],
        conv_denominator=1.0,
    )
    params.update(overrides)
    core = EquFlashCore(**params)
    core.eval()
    return core


def _run(core, pos, embedding, rbf, idx_i, idx_j):
    vij = pos[idx_j] - pos[idx_i]
    return core.get_output(
        atom_embedding=embedding,
        rbf=rbf,
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        vij_sr=vij,
    )["atom_feature"]


def test_equflash_irreps_match_published_natural_parity():
    from enerzyme.models.equflash.irreps import equflash_irreps

    schedule = equflash_irreps(channel=256, lmax=3, num_layers=10, parity=True)
    assert len(schedule) == 11
    assert str(schedule[0]) == "256x0e"
    assert str(schedule[-1]) == "256x0e"
    hidden = "256x0e+256x1o+256x2e+256x3o"
    assert all(str(ir) == hidden for ir in schedule[1:-1])

    manual = equflash_irreps(
        channel=4,
        lmax=1,
        num_layers=1,
        irreps_manual=["4x0e", "4x0e+4x1o"],
    )
    assert [str(ir) for ir in manual] == ["4x0e", "4x0e+4x1o"]


def test_xplor_window_and_equflash_bessel():
    from enerzyme.models.cutoff import xplor_transition
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff, cuton = 6.0, 5.5
    distances = torch.tensor([1.0, 5.5, 5.75, 6.0, 6.5])
    window = xplor_transition(distances, cutoff, cuton)
    assert torch.allclose(window[:2], torch.ones(2))
    assert window[2] > 0 and window[2] < 1
    assert torch.allclose(window[3:], torch.zeros(2))

    layer = BesselRBFLayer(
        num_rbf=4,
        cutoff_sr=cutoff,
        flavor="equflash",
        trainable=False,
        apply_cutoff_fn=False,
    )
    assert layer.apply_cutoff_fn is False
    sample = torch.tensor([0.0, 0.4, 1.5, 4.9])
    got = layer.get_rbf(sample)
    n = torch.arange(1, 5, dtype=sample.dtype)
    freqs = n * math.pi / cutoff
    prefactor = 2.0 / cutoff
    expected = []
    for dist in sample:
        if dist == 0:
            expected.append(prefactor * freqs)
        else:
            expected.append(prefactor * torch.sin(freqs * dist) / dist)
    assert torch.allclose(got, torch.stack(expected), atol=1e-6, rtol=1e-6)

    windowed = BesselRBFLayer(
        num_rbf=4,
        cutoff_sr=cutoff,
        flavor="equflash",
        trainable=False,
        apply_cutoff_fn=True,
        cutoff_fn="xplor",
        cuton=cuton,
    )
    raw = windowed._get_rbf(distances)
    assert torch.allclose(
        windowed.get_rbf(distances),
        raw * xplor_transition(distances, cutoff, cuton).unsqueeze(-1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_equflash_registration_and_yaml_metadata():
    from enerzyme.models.ff import get_ff_core

    for name in ("equflash", "EquFlash"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "EquFlashCore"
        assert build_params["cutoff_fn"] == "xplor"
        assert build_params["cuton"] == 5.5
        assert layers[1]["params"]["flavor"] == "equflash"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    assert ff["architecture"] == "equflash"
    assert ff["build_params"]["num_rbf"] == 8
    assert ff["layers"][3]["params"]["num_layers"] == 10


def test_equflash_rotation_translation_and_isolated_atom():
    from e3nn import o3

    # e3nn stores Clebsch–Gordan coefficients at init in the default dtype.
    # Casting a float32 module with .double() does not recompute them.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        n, f, n_rbf = 5, 4, 3
        core = _tiny_core(
            dim_embedding=f,
            num_rbf=n_rbf,
            irreps_manual=["4x0e", "4x0e+4x1o", "4x0e+4x1o"],
        )
        idx_i, idx_j = _edges(n)
        pos = torch.randn(n, 3)
        embedding = torch.randn(n, f)
        rbf = torch.randn(idx_i.numel(), n_rbf)

        out = _run(core, pos, embedding, rbf, idx_i, idx_j)
        assert out.shape == (n, o3.Irreps(core.feature_irreps).dim)
        assert "1o" in core.feature_irreps

        rotation = _rotation(1)
        # Row positions p' = p @ R^T rotate column vectors by R.
        rotated = _run(core, pos @ rotation.T, embedding, rbf, idx_i, idx_j)
        transform = o3.Irreps(core.feature_irreps).D_from_matrix(rotation)
        assert torch.allclose(rotated, out @ transform.T, atol=1e-8, rtol=1e-8)

        shifted = _run(core, pos + torch.tensor([3.0, -2.0, 0.5]), embedding, rbf, idx_i, idx_j)
        assert torch.allclose(out, shifted, atol=1e-8, rtol=1e-8)

        inverted = _run(core, -pos, embedding, rbf, idx_i, idx_j)
        parity = o3.Irreps(core.feature_irreps).D_from_matrix(-torch.eye(3))
        assert torch.allclose(inverted, out @ parity.T, atol=1e-8, rtol=1e-8)

        alone = core.get_output(
            atom_embedding=torch.randn(1, f),
            rbf=torch.zeros(0, n_rbf),
            idx_i_sr=torch.zeros(0, dtype=torch.long),
            idx_j_sr=torch.zeros(0, dtype=torch.long),
            vij_sr=torch.zeros(0, 3),
        )
        assert alone["atom_feature"].shape[0] == 1
        assert torch.isfinite(alone["atom_feature"]).all()

        gated = _tiny_core(
            dim_embedding=f,
            num_rbf=n_rbf,
            irreps_manual=["4x0e", "4x0e+4x1o"],
            feed_forward=True,
        )
        gated_out = _run(gated, pos, embedding, rbf, idx_i, idx_j)
        gated_rot = _run(gated, pos @ rotation.T, embedding, rbf, idx_i, idx_j)
        gated_d = o3.Irreps(gated.feature_irreps).D_from_matrix(rotation)
        assert torch.allclose(gated_rot, gated_out @ gated_d.T, atol=1e-8, rtol=1e-8)
    finally:
        torch.set_default_dtype(previous)


def test_equflash_messages_do_not_cross_molecules():
    torch.manual_seed(3)
    f, n_rbf = 4, 3
    core = _tiny_core(dim_embedding=f, num_rbf=n_rbf)

    def _pack(n):
        idx_i, idx_j = _edges(n)
        pos = torch.randn(n, 3)
        return pos, torch.randn(n, f), torch.randn(idx_i.numel(), n_rbf), idx_i, idx_j

    left = _pack(3)
    right = _pack(4)
    separate_l = _run(core, *left)
    separate_r = _run(core, *right)
    n_left = left[0].shape[0]
    joined_pos = torch.cat([left[0], right[0]])
    joined_emb = torch.cat([left[1], right[1]])
    joined_rbf = torch.cat([left[2], right[2]])
    joined_i = torch.cat([left[3], right[3] + n_left])
    joined_j = torch.cat([left[4], right[4] + n_left])
    both = _run(core, joined_pos, joined_emb, joined_rbf, joined_i, joined_j)
    assert torch.allclose(both[:n_left], separate_l, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], separate_r, atol=1e-6, rtol=1e-6)


def _tiny_layers():
    return [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "equflash", "trainable": False, "apply_cutoff_fn": True},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {"lmax": 1, "num_layers": 2, "radial_hidden": [8]},
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "two_layer",
                "activation_fn": "swish",
                "activation_params": {
                    "dim_feature": 1,
                    "initial_alpha": 1.0,
                    "initial_beta": 1.0,
                    "learnable": False,
                },
                "keep_feature": False,
            },
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


def _forward_stack(model, pos, za, batch_seg):
    n = pos.shape[0]
    idx_i, idx_j = _edges(n)
    return model(
        {
            "Ra": pos,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
        }
    )


def test_equflash_stack_energy_force_and_finite_difference():
    from enerzyme.models.ff import build_model
    from enerzyme.models.layers.readout import SimpleReadout

    torch.manual_seed(4)
    model = build_model(
        "equflash",
        layer_params=_tiny_layers(),
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 20,
            "cutoff_sr": 5.0,
            "cutoff_fn": "xplor",
            "cuton": 4.0,
        },
        verbose=0,
    ).double()
    model.eval()
    assert model.dim_feature_out == 4
    assert model.feature_irreps == "4x0e"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.dim_feature_in == 4

    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.3, 0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([6, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    out = _forward_stack(model, pos, za, batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (3, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    rotation = _rotation(2).double()
    rotated = _forward_stack(model, (pos.detach() @ rotation).requires_grad_(True), za, batch)
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(3):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            plus.requires_grad_(True)
            minus.requires_grad_(True)
            e_plus = _forward_stack(model, plus, za, batch)["E"][0]
            e_minus = _forward_stack(model, minus, za, batch)["E"][0]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-5, rtol=1e-4)
