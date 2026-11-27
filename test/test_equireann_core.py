"""EquiREANN Core: registration, the orbital message, forces, and response heads."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "equireann_layers_example.yaml"


def _rotation(seed: int = 0, improper: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    if improper:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _pairs(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i), torch.tensor(idx_j)


def _tiny(**kwargs):
    from enerzyme.models.presets.cartesian_tensor import equireann

    params = dict(
        dim_embedding=4,
        max_Za=10,
        cutoff_sr=4.0,
        lmax=1,
        nwave=2,
        num_iterations=1,
        nblock=1,
        hidden_dims=(4,),
        activation="tanh_like",
        layer_norm=False,
        elemental=False,
        coeff_init=0.1,
    )
    params.update(kwargs)
    return equireann(**params).double()


def _run(core, pos, embedding, species):
    idx_i, idx_j = _pairs(pos.shape[0])
    vij = pos[idx_j] - pos[idx_i]
    distance = torch.linalg.vector_norm(vij, dim=-1)
    keep = distance < core.cutoff_sr
    return core.get_output(
        atom_embedding=embedding,
        Za=species,
        idx_i_sr=idx_i[keep],
        idx_j_sr=idx_j[keep],
        vij_sr=vij[keep],
        Dij_sr=distance[keep],
    )["atom_feature"]


def test_registration_presets_and_example_yaml():
    from enerzyme.models.equireann import (
        PUBLISHED_CUMULENE_CORE_PARAMS,
        PUBLISHED_CUMULENE_CUTOFF,
    )
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout
    from enerzyme.models.loss import LOSS_REGISTER
    from enerzyme.models.reann.density import default_n_orbit

    for name in ("equireann", "EquiREANN", "equi-reann", "equi_reann"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "equireann"
        assert build_params["cutoff_sr"] == 6.0
        assert layers[2]["params"]["nwave"] == 20
        assert layers[2]["params"]["num_iterations"] == 2
        assert layers[2]["params"]["activation"] == "tanh_like"
        assert layers[2]["params"]["layer_norm"] is False

    assert PUBLISHED_CUMULENE_CORE_PARAMS["n_orbit"] == 8
    assert PUBLISHED_CUMULENE_CORE_PARAMS["num_iterations"] == 2
    assert PUBLISHED_CUMULENE_CORE_PARAMS["hidden_dims"] == [16, 16]
    assert PUBLISHED_CUMULENE_CUTOFF == 6.0
    assert default_n_orbit(20, 2) == 630

    get_ff_core("physnet")
    assert "nh_penalty" in LOSS_REGISTER
    get_ff_core("equireann")
    assert "nh_penalty" not in LOSS_REGISTER

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "ComposableCore"
    assert model.n_orbit == 630
    assert model.embed.hyper.shape == (3, 3, 20, 630)
    assert model.dim_feature_out == 630
    assert model.feature_irreps == "630x0e"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.dim_feature_in == 630

    try:
        _tiny(lmax=7)
    except ValueError as exc:
        assert "lmax" in str(exc)
    else:
        raise AssertionError("lmax above 6 must be rejected")
    try:
        build_model(
            "equireann",
            layer_params=[
                {"name": "RangeSeparation", "params": {"cutoff_sr": 3.0}},
                {"name": "RandomAtomEmbedding"},
                {"name": "Core"},
            ],
            build_params={"dim_embedding": 4, "max_Za": 10, "cutoff_sr": 6.0},
            verbose=0,
        )
    except ValueError as exc:
        assert "cutoff_sr" in str(exc)
    else:
        raise AssertionError("a RangeSeparation cutoff that differs from the Core must be rejected")
    try:
        _tiny(elemental=True)
    except ValueError as exc:
        assert "atomic_numbers" in str(exc)
    else:
        raise AssertionError("elemental networks need a species list")


def test_the_orbital_message_moves_the_density_before_the_coefficient_network():
    torch.manual_seed(0)
    updated = _tiny(num_iterations=1)
    plain = _tiny(num_iterations=0)
    with torch.no_grad():
        plain.embed.rs.copy_(updated.embed.rs)
        plain.embed.inta.copy_(updated.embed.inta)
        plain.embed.hyper.copy_(updated.embed.hyper[:1])
        plain.embed.coeff.load_state_dict(updated.embed.coeff.state_dict())
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.3]],
        dtype=torch.float64,
    )
    embedding = torch.randn(3, 4, dtype=torch.float64)
    species = torch.tensor([1, 6, 8])
    before = _run(updated, pos, embedding, species)
    # The released output linear is zero, so the coefficient is unchanged.
    # The passed orbital still changes the density.
    assert float(updated.blocks[0].shared.out.weight.detach().abs().sum()) == 0.0
    assert not torch.allclose(before, _run(plain, pos, embedding, species))
    twice = _tiny(num_iterations=2)
    with torch.no_grad():
        twice.embed.rs.copy_(updated.embed.rs)
        twice.embed.inta.copy_(updated.embed.inta)
        twice.embed.hyper.copy_(updated.embed.hyper[0].unsqueeze(0).repeat(twice.num_iterations + 1, 1, 1, 1))
        twice.embed.coeff.load_state_dict(updated.embed.coeff.state_dict())
        for step in range(2):
            twice.blocks[step].load_state_dict(updated.blocks[0].state_dict())
    assert not torch.allclose(before, _run(twice, pos, embedding, species))


def test_later_contraction_is_unused_until_that_iteration():
    core0 = _tiny(num_iterations=0)
    core1 = _tiny(num_iterations=1)
    with torch.no_grad():
        core1.embed.rs.copy_(core0.embed.rs)
        core1.embed.inta.copy_(core0.embed.inta)
        core1.embed.hyper.zero_()
        core1.embed.hyper[0].copy_(core0.embed.hyper[0])
        core1.embed.coeff.load_state_dict(core0.embed.coeff.state_dict())
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.2, 0.1], [-0.2, 0.7, 0.3]], dtype=torch.float64)
    embedding = torch.randn(3, 4, dtype=torch.float64)
    species = torch.tensor([1, 8, 6])
    base = _run(core1, pos, embedding, species).detach()
    with torch.no_grad():
        core1.embed.hyper[1].add_(0.7)
    moved = _run(core1, pos, embedding, species)
    assert not torch.allclose(base, moved.detach())
    with torch.no_grad():
        core1.embed.hyper[1].zero_()
    restored = _run(core1, pos, embedding, species)
    assert_allclose(base.numpy(), restored.detach().numpy(), atol=1e-10)
    # A zero-iteration model reads only the first slice, which matches REANN's hyper.
    from enerzyme.models.presets.cartesian_tensor import reann

    plain = reann(
        dim_embedding=4,
        max_Za=10,
        cutoff_sr=4.0,
        lmax=1,
        nwave=2,
        num_iterations=0,
        nblock=1,
        hidden_dims=(4,),
        activation="tanh_like",
        layer_norm=False,
    ).double()
    with torch.no_grad():
        plain.embed.rs.copy_(core0.embed.rs)
        plain.embed.inta.copy_(core0.embed.inta)
        plain.embed.hyper.copy_(core0.embed.hyper[0])
        plain.embed.coeff.load_state_dict(core0.embed.coeff.state_dict())
    assert_allclose(
        _run(core0, pos, embedding, species).detach().numpy(),
        _run(plain, pos, embedding, species).detach().numpy(),
        atol=1e-10,
    )


def test_elemental_update_follows_the_receiver_and_trains_the_contraction():
    torch.manual_seed(1)
    core = _tiny(elemental=True, atomic_numbers=(1, 6, 8), num_iterations=1)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [0.1, 0.6, 0.4]],
        dtype=torch.float64,
    )
    embedding = torch.randn(3, 4, dtype=torch.float64)
    species = torch.tensor([1, 6, 8])
    base = _run(core, pos, embedding, species).detach()
    with torch.no_grad():
        core.blocks[0].by_species["1"].out.weight.fill_(0.05)
        core.blocks[0].by_species["1"].out.bias.fill_(0.5)
    changed = _run(core, pos, embedding, species)
    assert not torch.allclose(base, changed.detach())
    changed.sum().backward()
    assert core.embed.hyper.grad is not None
    assert float(core.embed.hyper.grad[0].abs().sum()) > 0.0
    assert float(core.embed.hyper.grad[1].abs().sum()) > 0.0
    assert core.blocks[0].by_species["1"].out.bias.grad is not None
    try:
        _run(core, pos, embedding, torch.tensor([1, 6, 7]))
    except ValueError as exc:
        assert "7" in str(exc)
    else:
        raise AssertionError("an unlisted element must be rejected")


def test_core_density_is_invariant():
    torch.manual_seed(3)
    core = _tiny(num_iterations=2)
    core.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.1, 1.0, 0.4], [-0.4, 0.3, 0.8]],
        dtype=torch.float64,
    )
    embedding = torch.randn(4, 4, dtype=torch.float64)
    species = torch.tensor([1, 6, 8, 1])
    base = _run(core, pos, embedding, species)
    for improper in (False, True):
        rotated = _run(core, pos @ _rotation(7 + int(improper), improper), embedding, species)
        assert_allclose(base.detach().numpy(), rotated.detach().numpy(), atol=1e-10)
    shifted = _run(core, pos + torch.tensor([2.0, -1.0, 0.5]), embedding, species)
    assert_allclose(base.detach().numpy(), shifted.detach().numpy(), atol=1e-10)


def _energy_layers():
    return [
        {"name": "RangeSeparation"},
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "lmax": 1,
                "nwave": 2,
                "num_iterations": 2,
                "nblock": 1,
                "hidden_dims": [4],
                "activation": "tanh_like",
                "layer_norm": False,
                "elemental": False,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "mlp",
                "hidden_dims": [4],
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


def test_layer_stack_forces_offsets_and_isolated_atom():
    from enerzyme.models.ff import build_model

    model = build_model(
        "equireann",
        layer_params=_energy_layers(),
        build_params={"dim_embedding": 4, "max_Za": 10, "cutoff_sr": 5.0},
        verbose=0,
    )
    model.double()
    n = 4
    idx_i, idx_j = _pairs(n)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 1.1, 0.2], [0.4, 0.3, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = {
        "Ra": pos,
        "Za": torch.tensor([1, 6, 8, 1]),
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (n, 3)
    force = torch.autograd.grad(out["E"].sum(), pos, retain_graph=True)[0]
    assert_allclose(out["Fa"].detach().numpy(), (-force).detach().numpy(), atol=1e-8)
    assert_allclose(out["Fa"].sum(0).detach().numpy(), torch.zeros(3).numpy(), atol=1e-8)

    rotation = _rotation(5)
    rotated = (pos.detach() @ rotation).requires_grad_(True)
    out_r = model({**batch, "Ra": rotated})
    assert_allclose(out["E"].detach().numpy(), out_r["E"].detach().numpy(), atol=1e-8)
    assert_allclose(
        out_r["Fa"].detach().numpy(),
        (out["Fa"].detach() @ rotation).numpy(),
        atol=1e-7,
        rtol=1e-6,
    )
    shifted = (pos.detach() + torch.tensor([1.5, -0.4, 0.2])).requires_grad_(True)
    out_t = model({**batch, "Ra": shifted})
    assert_allclose(out["E"].detach().numpy(), out_t["E"].detach().numpy(), atol=1e-8)

    eps = 1e-5
    plus = pos.detach().clone()
    minus = pos.detach().clone()
    plus[1, 0] = plus[1, 0] + eps
    minus[1, 0] = minus[1, 0] - eps
    e_plus = model({**batch, "Ra": plus.requires_grad_(True)})["E"]
    e_minus = model({**batch, "Ra": minus.requires_grad_(True)})["E"]
    finite = -(e_plus - e_minus) / (2 * eps)
    assert_allclose(finite.detach().numpy(), out["Fa"][1, 0].detach().numpy(), atol=1e-5, rtol=1e-4)

    offsets = torch.zeros(idx_i.numel(), 3, dtype=torch.float64)
    offsets[0, 0] = 0.25
    out_off = model({**batch, "Ra": pos.detach().requires_grad_(True), "offsets": offsets})
    assert not torch.allclose(out["E"], out_off["E"])

    alone = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64, requires_grad=True)
    out_a = model(
        {
            "Ra": alone,
            "Za": torch.tensor([6]),
            "batch_seg": torch.zeros(1, dtype=torch.long),
            "idx_i": torch.zeros(0, dtype=torch.long),
            "idx_j": torch.zeros(0, dtype=torch.long),
        }
    )
    assert torch.isfinite(out_a["E"]).all()
    assert_allclose(out_a["Fa"].detach().numpy(), torch.zeros(1, 3).numpy(), atol=1e-8)

    model.train()
    pos2 = pos.detach().requires_grad_(True)
    loss = model({**batch, "Ra": pos2})["E"].sum()
    loss.backward()
    assert model.embed.rs.grad is not None
    assert float(model.embed.rs.grad.abs().sum()) > 0.0
    assert float(model.embed.hyper.grad.abs().sum()) > 0.0


def test_response_stack_rotates_the_dipole():
    from enerzyme.models.ff import build_model

    layers = _energy_layers()
    layers[3]["params"]["keep_feature"] = True
    layers.insert(
        -2,
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Qa"],
                "head_type": "dense",
                "use_bias": False,
                "keep_feature": True,
            },
        },
    )
    layers.insert(-2, {"name": "AtomicCharge2Dipole"})
    model = build_model(
        "equireann",
        layer_params=layers,
        build_params={"dim_embedding": 4, "max_Za": 10, "cutoff_sr": 5.0},
        verbose=0,
    )
    model.double()
    model.eval()
    n = 3
    idx_i, idx_j = _pairs(n)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [-0.2, 0.7, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = {
        "Ra": pos,
        "Za": torch.tensor([1, 6, 8]),
        "batch_seg": torch.zeros(n, dtype=torch.long),
        "idx_i": idx_i,
        "idx_j": idx_j,
    }
    out = model(batch)
    assert out["M2"].shape == (1, 3)
    rotation = _rotation(9)
    rotated = (pos.detach() @ rotation).requires_grad_(True)
    out_r = model({**batch, "Ra": rotated})
    assert_allclose(out["E"].detach().numpy(), out_r["E"].detach().numpy(), atol=1e-8)
    assert_allclose(
        out_r["M2"].detach().numpy(),
        (out["M2"].detach() @ rotation).numpy(),
        atol=1e-6,
        rtol=1e-5,
    )
