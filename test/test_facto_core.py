"""FACTO Core: registration, equivariance, energy conservation, batch isolation."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "facto_layers_example.yaml"


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _edges(positions: torch.Tensor, cutoff: float):
    number = positions.shape[0]
    receivers, senders = [], []
    for i in range(number):
        for j in range(number):
            if i == j:
                continue
            if torch.linalg.vector_norm(positions[j] - positions[i]) < cutoff:
                receivers.append(i)
                senders.append(j)
    if not receivers:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    return torch.tensor(receivers), torch.tensor(senders)


def _tiny_core(**overrides):
    from enerzyme.models.facto import FACTOCore

    params = dict(
        max_Za=10,
        cutoff_sr=6.0,
        r_short=1.0,
        preset="custom",
        scalar_dim=8,
        num_groups=2,
        tensor_channels=(4, 2),
        num_radial=4,
        num_blocks=2,
        tensor_transport=True,
    )
    params.update(overrides)
    return FACTOCore(**params)


def _core_inputs(positions: torch.Tensor, atomic_numbers: torch.Tensor, cutoff: float = 6.0):
    idx_i, idx_j = _edges(positions, cutoff)
    if idx_i.numel() == 0:
        return {
            "Za": atomic_numbers,
            "Dij_sr": positions.new_zeros(0),
            "vij_sr": positions.new_zeros(0, 3),
            "idx_i_sr": idx_i,
            "idx_j_sr": idx_j,
        }
    displacement = positions[idx_j] - positions[idx_i]
    return {
        "Za": atomic_numbers,
        "Dij_sr": torch.linalg.vector_norm(displacement, dim=-1),
        "vij_sr": displacement,
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
    }


def test_registration_presets_and_yaml():
    from enerzyme.models.facto import PAPER_PRESETS, RESIDUAL_SCALE
    from enerzyme.models.facto.readout import FACTOEnergyReadout
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.zbl import ZBLRepulsionEnergyLayer

    for name in ("facto", "FACTO"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "FACTOCore"
        assert build_params["cutoff_sr"] == 6.0
        assert build_params["cutoff_fn"] == "facto"
        assert layers[1]["params"]["preset"] == "S"
        assert layers[2]["name"] == "FACTOEnergyReadout"
        assert layers[3]["name"] == "ZBLRepulsionEnergy"

    assert PAPER_PRESETS["S"]["tensor_transport"] is False
    assert PAPER_PRESETS["M"]["tensor_transport"] is False
    assert PAPER_PRESETS["L"]["tensor_transport"] is True
    assert PAPER_PRESETS["XL"]["tensor_channels"] == (512, 256, 128, 64, 32, 16)
    assert abs(RESIDUAL_SCALE - 40.0 ** -0.5) < 1e-15

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=dict(ff["build_params"], max_Za=12),
        verbose=0,
    )
    assert model.__class__.__name__ == "FACTOCore"
    assert model.scalar_dim == 16
    assert model.num_blocks == 3
    assert model.lmax == 2
    assert model.feature_irreps == "16x0e"
    assert model.blocks[0].injection is not None
    assert model.blocks[-1].injection is None
    assert model.blocks[0].transport is not None
    readout = next(layer for layer in model.post_sequence if isinstance(layer, FACTOEnergyReadout))
    assert readout.projection.bias is None
    assert any(isinstance(layer, ZBLRepulsionEnergyLayer) for layer in model.post_sequence)

    try:
        _tiny_core(preset="nope")
    except ValueError as exc:
        assert "preset" in str(exc).lower() or "Unknown" in str(exc)
    else:
        raise AssertionError("unknown preset should fail")
    try:
        _tiny_core(scalar_dim=7, num_groups=2)
    except ValueError:
        pass
    else:
        raise AssertionError("indivisible groups should fail")


def test_paper_small_backbone_forward_and_last_block_is_scalar_only():
    from enerzyme.models.facto import FACTOCore

    torch.manual_seed(0)
    core = FACTOCore(max_Za=8, preset="S")
    core.eval()
    assert core.scalar_dim == 384
    assert core.num_radial == 32
    assert core.tensor_channels == (128, 64, 32, 16)
    assert core.blocks[-1].update_tensors is False
    assert all(block.transport is None for block in core.blocks)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [0.2, 1.5, 0.3]])
    atomic_numbers = torch.tensor([8, 1, 1])
    out = core.get_output(**_core_inputs(positions, atomic_numbers))
    assert out["atom_feature"].shape == (3, 384, 5)
    assert out["isolated_atom_feature"].shape == (3, 384, 5)
    assert torch.isfinite(out["atom_feature"]).all()


def test_isolated_atom_matches_the_neighbor_free_state_and_energy_offset():
    from enerzyme.models.facto.readout import FACTOEnergyReadout, concat_block_states

    torch.manual_seed(1)
    core = _tiny_core().eval()
    atomic_numbers = torch.tensor([6, 1, 8])
    positions = torch.tensor([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 50.0, 0.2]])
    out = core.get_output(**_core_inputs(positions, atomic_numbers))
    assert torch.allclose(
        out["atom_feature"], out["isolated_atom_feature"], atol=1e-6, rtol=1e-6
    )
    packed = concat_block_states(out["atom_feature"])
    assert packed.shape == (3, core.scalar_dim * core.num_blocks)

    readout = FACTOEnergyReadout(built_layers=[core], max_Za=core.max_Za)
    with torch.no_grad():
        readout.shift.copy_(torch.linspace(-0.2, 0.4, core.max_Za + 1))
        readout.reference.copy_(torch.linspace(0.1, 0.5, core.max_Za + 1))
    energy = readout.get_Ea(
        atom_feature=out["atom_feature"],
        isolated_atom_feature=out["isolated_atom_feature"],
        Za=atomic_numbers,
    )
    expected = readout.reference[atomic_numbers] + readout.shift[atomic_numbers]
    assert torch.allclose(energy, expected, atol=1e-6, rtol=1e-6)

    close = torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0], [0.1, 1.4, 0.2]])
    interacting = core.get_output(**_core_inputs(close, atomic_numbers))
    assert not torch.allclose(
        interacting["atom_feature"],
        interacting["isolated_atom_feature"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_molecules_do_not_exchange_messages():
    torch.manual_seed(2)
    core = _tiny_core().eval()
    left_pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.2, 1.3, 0.1]])
    right_pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, -0.2, 0.3]])
    left_z = torch.tensor([6, 1, 1])
    right_z = torch.tensor([8, 1])
    left = core.get_output(**_core_inputs(left_pos, left_z))["atom_feature"]
    right = core.get_output(**_core_inputs(right_pos, right_z))["atom_feature"]
    joined_pos = torch.cat([left_pos, right_pos + torch.tensor([30.0, -10.0, 4.0])], dim=0)
    joined_z = torch.cat([left_z, right_z])
    both = core.get_output(**_core_inputs(joined_pos, joined_z))["atom_feature"]
    assert torch.allclose(both[:3], left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[3:], right, atol=1e-6, rtol=1e-6)


def test_transport_changes_the_representation():
    # Stored tensors start at zero, so the first block's transport is a no-op.
    # The next tensor block carries those injected features into the last
    # scalar update. Three blocks are the smallest stack where that happens.
    torch.manual_seed(3)
    core = _tiny_core(tensor_transport=True, num_blocks=3).eval()
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.2, 1.5, 0.4]])
    atomic_numbers = torch.tensor([6, 6, 1])
    kwargs = _core_inputs(positions, atomic_numbers)
    baseline = core.get_output(**kwargs)["atom_feature"]
    with torch.no_grad():
        for block in core.blocks:
            if block.transport is None:
                continue
            for parameter in block.transport.parameters():
                parameter.zero_()
    ablated = core.get_output(**kwargs)["atom_feature"]
    assert torch.allclose(baseline[:, :, 0], ablated[:, :, 0], atol=1e-6, rtol=1e-6)
    assert (baseline - ablated).abs().max() > 1e-7


def _stack(**overrides):
    from enerzyme.models.ff import build_model

    layers = [
        {"name": "RangeSeparation"},
        {
            "name": "Core",
            "params": {
                "preset": "custom",
                "scalar_dim": 8,
                "num_groups": 2,
                "tensor_channels": [4, 2],
                "num_radial": 4,
                "num_blocks": 3,
                "tensor_transport": True,
                "r_short": 1.0,
            },
        },
        {"name": "FACTOEnergyReadout"},
        {"name": "ZBLRepulsionEnergy"},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]
    build = {
        "max_Za": 10,
        "cutoff_sr": 6.0,
        "cutoff_fn": "facto",
        "Hartree_in_E": 1.0,
        "Bohr_in_R": 0.5291772105638411,
    }
    build.update(overrides)
    return build_model("facto", layer_params=layers, build_params=build, verbose=0)


def _forward(model, positions, atomic_numbers, batch):
    number = positions.shape[0]
    receivers, senders = [], []
    for i in range(number):
        for j in range(number):
            if i != j:
                receivers.append(i)
                senders.append(j)
    return model(
        {
            "Ra": positions,
            "Za": atomic_numbers,
            "idx_i": torch.tensor(receivers, dtype=torch.long),
            "idx_j": torch.tensor(senders, dtype=torch.long),
            "batch_seg": batch,
        }
    )


def test_stack_is_invariant_and_forces_are_conservative():
    torch.manual_seed(4)
    model = _stack().double().eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.35, 0.15, 0.0], [0.2, 1.4, 0.25], [2.2, 0.4, 1.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([8, 1, 1, 6])
    batch = torch.zeros(4, dtype=torch.long)
    base = _forward(model, positions, atomic_numbers, batch)
    assert base["E"].shape == (1,)
    assert base["Fa"].shape == (4, 3)
    assert torch.isfinite(base["Fa"]).all()

    shift = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    moved = _forward(model, (positions.detach() + shift).requires_grad_(True), atomic_numbers, batch)
    assert torch.allclose(base["E"], moved["E"], atol=1e-8, rtol=1e-8)
    assert torch.allclose(base["Fa"], moved["Fa"], atol=1e-8, rtol=1e-8)

    rotation = _rotation(5).double()
    rotated_pos = (positions.detach() @ rotation).requires_grad_(True)
    rotated = _forward(model, rotated_pos, atomic_numbers, batch)
    assert torch.allclose(base["E"], rotated["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(base["Fa"] @ rotation, rotated["Fa"], atol=1e-6, rtol=1e-6)

    inverted_pos = (-positions.detach()).requires_grad_(True)
    inverted = _forward(model, inverted_pos, atomic_numbers, batch)
    assert torch.allclose(base["E"], inverted["E"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(-base["Fa"], inverted["Fa"], atol=1e-6, rtol=1e-6)

    analytical = base["Fa"].detach()
    step = 1e-5
    numerical = torch.zeros_like(analytical)
    for atom in range(4):
        for coord in range(3):
            plus = positions.detach().clone()
            minus = positions.detach().clone()
            plus[atom, coord] += step
            minus[atom, coord] -= step
            e_plus = _forward(model, plus.requires_grad_(True), atomic_numbers, batch)["E"]
            e_minus = _forward(model, minus.requires_grad_(True), atomic_numbers, batch)["E"]
            numerical[atom, coord] = -(e_plus - e_minus).detach() / (2 * step)
    assert torch.allclose(analytical, numerical, atol=2e-5, rtol=1e-5)


def test_two_molecules_keep_separate_total_energies():
    torch.manual_seed(5)
    model = _stack().double().eval()
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.1, 0.0],
            [0.2, 1.3, 0.05],
            [25.0, 0.0, 0.0],
            [26.2, 0.3, 0.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = torch.tensor([6, 1, 1, 8, 1])
    batch = torch.tensor([0, 0, 0, 1, 1])
    together = _forward(model, positions, atomic_numbers, batch)
    left = _forward(
        model,
        positions[:3].detach().requires_grad_(True),
        atomic_numbers[:3],
        torch.zeros(3, dtype=torch.long),
    )
    right = _forward(
        model,
        positions[3:].detach().requires_grad_(True),
        atomic_numbers[3:],
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(together["E"][0], left["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(together["E"][1], right["E"][0], atol=1e-8, rtol=1e-8)
    assert torch.allclose(together["Fa"][:3], left["Fa"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(together["Fa"][3:], right["Fa"], atol=1e-7, rtol=1e-7)
