"""BOTNet Core registration, readout, and conservative energy."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "botnet_layers_example.yaml"


def _rotation(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _tiny_layers():
    return [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "mace", "trainable": False, "polynomial_p": 6},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "max_ell": 1,
                "num_interactions": 2,
                "variant": "agnostic",
                "output_mode": "layers",
                "MLP_irreps": "4x0e",
                "radial_MLP": [8],
                "gate": "silu",
            },
        },
        {
            "name": "BOTNetEnergyReadout",
            "params": {"MLP_irreps": "4x0e", "gate": "silu"},
        },
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


def _forward(model, pos, atomic_numbers, batch_seg):
    idx_i, idx_j = _edges(pos.shape[0])
    return model(
        {
            "Ra": pos,
            "Za": atomic_numbers,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
        }
    )


def test_registration_published_widths_and_yaml():
    from enerzyme.models.botnet import (
        PUBLISHED_3BPA_HIDDEN_IRREPS,
        PUBLISHED_BUILD_PARAMS,
        PUBLISHED_CORE_PARAMS,
    )
    from enerzyme.models.botnet.readout import BOTNetEnergyReadout, _ScalarLinear, _ScalarMLP
    from enerzyme.models.ff import build_model, get_ff_core

    for name in ("botnet", "BOTNet", "body_ordered", "body-ordered"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "botnet"
        assert build_params["cutoff_sr"] == 4.0
        assert build_params["num_rbf"] == 8
        assert layers[1]["params"]["flavor"] == "mace"
        assert layers[1]["params"]["polynomial_p"] == 6
        assert layers[3]["params"]["variant"] == "agnostic"
        assert layers[4]["name"] == "BOTNetEnergyReadout"

    assert PUBLISHED_CORE_PARAMS["num_interactions"] == 5
    assert PUBLISHED_CORE_PARAMS["max_ell"] == 3
    assert PUBLISHED_BUILD_PARAMS["polynomial_p"] == 6
    assert "80x0o" in PUBLISHED_3BPA_HIDDEN_IRREPS

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    assert ff["architecture"] == "botnet"
    assert ff["layers"][1]["params"]["polynomial_p"] == 6

    model = build_model(
        "botnet",
        layer_params=_tiny_layers(),
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 4.0,
            "avg_num_neighbors": 2.0,
        },
        verbose=0,
    )
    assert model.__class__.__name__ == "ComposableCore"
    assert model.uniform_irreps
    assert not hasattr(model, "products")
    readout = next(layer for layer in model.post_sequence if isinstance(layer, BOTNetEnergyReadout))
    assert isinstance(readout.heads[0], _ScalarLinear)
    assert isinstance(readout.heads[-1], _ScalarMLP)


def test_readout_sums_linear_layers_and_a_final_mlp():
    from enerzyme.models.presets.irreps import botnet
    from enerzyme.models.botnet.readout import BOTNetEnergyReadout

    torch.manual_seed(0)
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=3,
        radial_MLP=[8],
        MLP_irreps="4x0e",
        gate="silu",
        output_mode="layers",
    )
    core.eval()
    n = 3
    idx_i, idx_j = _edges(n)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.1, 0.6, 0.2]])
    features = core.get_output(
        Za=torch.tensor([6, 1, 8]),
        vij_sr=pos[idx_j] - pos[idx_i],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.randn(idx_i.numel(), 3),
        atom_embedding=torch.randn(n, 4),
    )["atom_feature"]
    readout = BOTNetEnergyReadout(built_layers=[core])
    readout.eval()
    energy = readout.get_output(atom_feature=features)["Ea"]
    manual = (
        readout.heads[0](features[:, :, 0])
        + readout.heads[1](features[:, :, 1])
        + readout.heads[2](features[:, :, 2])
    )
    assert torch.allclose(energy, manual)
    single = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=1,
        radial_MLP=[8],
        output_mode="feature",
    )
    assert single.last_readout == "linear"
    assert single.num_output_blocks == 1
    head = BOTNetEnergyReadout(built_layers=[single])
    assert len(head.heads) == 1
    assert type(head.heads[0]).__name__ == "_ScalarLinear"


def test_molecules_do_not_mix_and_charge_is_optional():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(4)
    core = botnet(
        max_Za=8,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        radial_MLP=[8],
    )
    core.eval()

    def pack(seed, n):
        generator = torch.Generator().manual_seed(seed)
        idx_i, idx_j = _edges(n)
        pos = torch.randn(n, 3, generator=generator)
        return {
            "Za": torch.randint(1, 8, (n,), generator=generator),
            "vij_sr": pos[idx_j] - pos[idx_i],
            "idx_i_sr": idx_i,
            "idx_j_sr": idx_j,
            "rbf": torch.randn(idx_i.numel(), 3, generator=generator),
            "atom_embedding": torch.randn(n, 4, generator=generator),
        }

    left = pack(5, 3)
    right = pack(6, 4)
    separate_l = core.get_output(**left)["atom_feature"]
    separate_r = core.get_output(**right)["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    both = core.get_output(
        Za=torch.cat([left["Za"], right["Za"]]),
        vij_sr=torch.cat([left["vij_sr"], right["vij_sr"]]),
        idx_i_sr=torch.cat([left["idx_i_sr"], right["idx_i_sr"] + n_left]),
        idx_j_sr=torch.cat([left["idx_j_sr"], right["idx_j_sr"] + n_left]),
        rbf=torch.cat([left["rbf"], right["rbf"]]),
        atom_embedding=torch.cat([left["atom_embedding"], right["atom_embedding"]]),
    )["atom_feature"]
    assert torch.allclose(both[:n_left], separate_l, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], separate_r, atol=1e-6, rtol=1e-6)

    zeros = torch.zeros_like(left["atom_embedding"])
    plain = core.get_output(**left)["atom_feature"]
    with_zero = core.get_output(**left, charge_embedding=zeros, spin_embedding=zeros)["atom_feature"]
    assert torch.allclose(plain, with_zero)
    shifted = core.get_output(
        **left, charge_embedding=torch.randn_like(left["atom_embedding"])
    )["atom_feature"]
    assert not torch.allclose(plain, shifted)


def test_stack_energy_is_conservative():
    from enerzyme.models.ff import build_model

    torch.manual_seed(7)
    model = build_model(
        "botnet",
        layer_params=_tiny_layers(),
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 4.0,
            "avg_num_neighbors": 3.0,
        },
        verbose=0,
    ).double()
    model.eval()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.15, 0.8, 0.05],
            [0.2, 0.1, 0.85],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 6, 1])
    batch = torch.zeros(4, dtype=torch.long)
    out = _forward(model, pos, atomic_numbers, batch)
    assert out["E"].shape == (1,)
    assert out["Fa"].shape == (4, 3)
    assert torch.isfinite(out["E"]).all()
    assert torch.isfinite(out["Fa"]).all()

    shifted = (pos.detach() + torch.tensor([1.5, -0.4, 0.2])).requires_grad_(True)
    moved = _forward(model, shifted, atomic_numbers, batch)
    assert torch.allclose(out["E"], moved["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"], moved["Fa"], atol=1e-6, rtol=1e-6)

    rotation = _rotation(8).double()
    rotated_pos = (pos.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, atomic_numbers, batch)
    assert torch.allclose(out["E"], rotated["E"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    analytical = out["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = pos.detach().clone()
            minus = pos.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            energy_plus = _forward(model, plus.requires_grad_(True), atomic_numbers, batch)["E"]
            energy_minus = _forward(model, minus.requires_grad_(True), atomic_numbers, batch)["E"]
            numerical[atom, coord] = -(energy_plus - energy_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-4, rtol=1e-4)


def test_element_dependent_variant_runs():
    from enerzyme.models.presets.irreps import botnet

    torch.manual_seed(9)
    core = botnet(
        max_Za=6,
        dim_embedding=4,
        num_rbf=3,
        max_ell=1,
        num_interactions=2,
        variant="element",
        radial_MLP=[8],
    )
    core.eval()
    assert core.interaction_cls_first == "AgnosticNonlinearInteractionBlock"
    assert core.interaction_cls == "ResidualElementDependentInteractionBlock"
    n = 3
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3)
    out = core.get_output(
        Za=torch.tensor([1, 6, 1]),
        vij_sr=pos[idx_j] - pos[idx_i],
        idx_i_sr=idx_i,
        idx_j_sr=idx_j,
        rbf=torch.rand(idx_i.numel(), 3),
        atom_embedding=torch.randn(n, 4),
    )["atom_feature"]
    assert torch.isfinite(out).all()
    assert out.shape[-1] == 2
