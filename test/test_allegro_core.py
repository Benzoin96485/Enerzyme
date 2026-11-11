"""Allegro Core: pair embedding, channelwise tensor product, energy and forces."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "allegro_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotation(seed: int = 0, proper: bool = True) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    positive = torch.det(q) > 0
    if proper != bool(positive):
        q = q.clone()
        q[:, 0] *= -1
    return q


def _core(**overrides):
    from enerzyme.models.allegro import AllegroCore

    params = dict(
        dim_edge=4,
        avg_num_neighbors=2.0,
        num_layers=2,
        l_max=1,
        parity=True,
        num_scalar_features=4,
        num_tensor_features=2,
        scalar_embed_mlp_depth=1,
        scalar_embed_mlp_width=4,
        latent_mlp_depth=1,
        latent_mlp_width=4,
    )
    params.update(overrides)
    return AllegroCore(**params)


def test_pair_embedding_mixes_center_neighbor_and_radius():
    from enerzyme.models.layers.pair_embedding import PairChemicalRadialLayer

    layer = PairChemicalRadialLayer(max_Za=10, num_rbf=3, dim_edge=4)
    za = torch.tensor([6, 1, 8])
    idx_i, idx_j = _edges(3)
    rbf = torch.randn(idx_i.numel(), 3)
    out = layer.get_output(za, rbf, idx_i, idx_j)["edge_embedding"]
    assert out.shape == (idx_i.numel(), 4)
    swapped = layer.get_output(za, rbf, idx_j, idx_i)["edge_embedding"]
    assert not torch.allclose(out, swapped)
    try:
        PairChemicalRadialLayer(max_Za=10, num_rbf=3, dim_edge=3)
    except ValueError:
        pass
    else:
        raise AssertionError("odd dim_edge should be rejected")


def test_layer_irreps_end_on_scalars_and_start_with_0e():
    from enerzyme.models.allegro.irreps import allegro_layer_irreps

    full = allegro_layer_irreps(l_max=1, parity=True, num_layers=2)
    natural = allegro_layer_irreps(l_max=2, parity=False, num_layers=1)
    assert len(full) == 2
    assert str(full[-1][1]) == "1x0e"
    for incoming, outgoing in full:
        assert outgoing[0].ir == o3.Irrep("0e")
        assert incoming[0].ir == o3.Irrep("0e")
    assert all(ir.p == (-1) ** ir.l for _, ir in natural[0][1])
    try:
        allegro_layer_irreps(l_max=1, parity=True, num_layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("num_layers=0 should be rejected")


def test_channel_product_is_equivariant_and_matches_wigner():
    from enerzyme.models.allegro.interaction import ChannelTensorProduct

    torch.manual_seed(0)
    base = o3.Irreps("1x1o")
    product = ChannelTensorProduct(
        base, base, o3.Irreps("1x0e"), mul=2, path_channel_coupling=True
    ).double()
    product.eval()
    with torch.no_grad():
        product.tp.weight.copy_(torch.tensor([0.4, -0.7], dtype=torch.float64))
    n = 5
    left = product.irreps_in1.randn(n, -1).double()
    right = product.irreps_in2.randn(n, -1).double()
    out = product(left, right)
    assert out.shape == (n, 2)

    w3j = o3.wigner_3j(1, 1, 0).double()
    norm = product.tp.instructions[0].path_weight
    manual = []
    for row in range(n):
        channels = []
        for channel, weight in enumerate(product.tp.weight.detach()):
            x1 = left[row, channel * 3 : (channel + 1) * 3]
            x2 = right[row, channel * 3 : (channel + 1) * 3]
            contracted = torch.einsum("ijk,i,j->k", w3j, x1, x2).reshape(-1)
            channels.append(norm * weight * contracted)
        manual.append(torch.cat(channels))
    manual = torch.stack(manual)
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-6)

    rotation = _rotation(1).double()
    d_in = product.irreps_in1.D_from_matrix(rotation)
    d_out = product.irreps_out.D_from_matrix(rotation)
    rotated = product(left @ d_in, right @ d_in)
    assert torch.allclose(rotated, out @ d_out, atol=1e-6, rtol=1e-6)

    shared = ChannelTensorProduct(
        base, base, o3.Irreps("1x0e+1x1e+1x2e"), mul=3, path_channel_coupling=False
    )
    assert shared.path_weight.shape == (shared.num_paths,)
    assert shared.tp.weight_numel == shared.num_paths * 3


def test_weighted_channels_are_equivariant():
    from enerzyme.models.allegro.interaction import WeightedSphericalChannels

    irreps = o3.Irreps.spherical_harmonics(1)
    lifted = WeightedSphericalChannels(irreps, mul=2, weight_individual_irreps=True).double()
    basis = irreps.randn(4, -1).double()
    weights = torch.randn(4, lifted.weight_numel, dtype=torch.float64)
    out = lifted(basis, weights)
    rotation = _rotation(2, proper=False).double()
    d_in = irreps.D_from_matrix(rotation)
    d_out = lifted.irreps_out.D_from_matrix(rotation)
    rotated = lifted(basis @ d_in, weights)
    assert torch.allclose(rotated, out @ d_out, atol=1e-6, rtol=1e-6)
    # One weight per irrep: the l=1 block of channel 0 scales by that weight.
    with torch.no_grad():
        weights.zero_()
        weights[:, 1] = 2.0  # channel 0, second irrep (1o), flat [channel, irrep]
    scaled = lifted(basis, weights)
    assert torch.allclose(scaled[:, 2:5], 2.0 * basis[:, 1:4], atol=1e-6, rtol=1e-6)


def test_edge_energy_reduce_uses_allegro_scale():
    from enerzyme.models.layers.reduce import EdgeEnergyReduceLayer

    layer = EdgeEnergyReduceLayer(avg_num_neighbors=4.0, pair_symmetry=True)
    energy = torch.tensor([1.0, 3.0, 2.0])
    centers = torch.tensor([0, 0, 1])
    atomic = layer.get_output(energy, centers, torch.zeros(3))["Ea"]
    scale = 1.0 / math.sqrt(4.0) / math.sqrt(2.0)
    assert torch.allclose(atomic, torch.tensor([4.0, 2.0, 0.0]) * scale)
    raw = EdgeEnergyReduceLayer(avg_num_neighbors=1.0, pair_symmetry=False)
    assert torch.allclose(
        raw.get_output(energy, centers, torch.zeros(3))["Ea"],
        torch.tensor([4.0, 2.0, 0.0]),
    )


def test_registration_preset_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import EdgeReadout

    for name in ("allegro", "Allegro"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "AllegroCore"
        assert build_params["num_rbf"] == 8
        assert layers[1]["params"]["flavor"] == "nequip"
        assert layers[2]["name"] == "PairChemicalRadial"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    layers = []
    for layer in ff["layers"]:
        if layer["name"] != "Core":
            layers.append(layer)
            continue
        layers.append(
            {
                "name": "Core",
                "params": dict(
                    layer["params"],
                    num_layers=1,
                    num_scalar_features=4,
                    num_tensor_features=2,
                    scalar_embed_mlp_width=4,
                    latent_mlp_width=4,
                ),
            }
        )
    model = build_model(
        ff["architecture"],
        layer_params=layers,
        build_params=dict(ff["build_params"], dim_edge=4, num_rbf=3, max_Za=10),
        verbose=0,
    )
    assert model.dim_edge == 4
    assert model.num_layers == 1
    readout = next(layer for layer in model.post_sequence if isinstance(layer, EdgeReadout))
    assert readout.head_type == "variance_preserving"
    assert readout._input_fields == {"edge_feature"}
    assert model.readout_hidden_width == 4
    assert readout._hinted_hidden_width == 4

    small = _core()
    assert small.dim_feature_out == 4 * (2 + 1)
    assert small.feature_irreps == "12x0e"
    try:
        _core(num_layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("num_layers=0 should be rejected")


def test_core_scalars_are_invariant_and_molecules_stay_separate():
    torch.manual_seed(1)
    core = _core().double()
    core.eval()
    n = 4
    idx_i, idx_j = _edges(n)
    generator = torch.Generator().manual_seed(2)
    pos = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    vij = pos[idx_j] - pos[idx_i]
    embedding = torch.randn(idx_i.numel(), 4, generator=generator, dtype=torch.float64)

    def run(pos_now, embed):
        index_i, index_j = _edges(pos_now.shape[0])
        return core.get_output(
            edge_embedding=embed,
            idx_i_sr=index_i,
            idx_j_sr=index_j,
            vij_sr=pos_now[index_j] - pos_now[index_i],
        )["edge_feature"]

    reference = run(pos, embedding)
    assert reference.shape == (idx_i.numel(), 12)
    assert torch.allclose(run(pos @ _rotation(3).double(), embedding), reference, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        run(pos @ _rotation(4, proper=False).double(), embedding),
        reference,
        atol=1e-6,
        rtol=1e-6,
    )

    left = run(pos[:3], embedding[:6])
    shift = torch.tensor([8.0, -3.0, 1.0], dtype=torch.float64)
    right_pos = pos[:3] + shift
    right_embed = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    right = core.get_output(
        edge_embedding=right_embed,
        idx_i_sr=_edges(3)[0],
        idx_j_sr=_edges(3)[1],
        vij_sr=right_pos[_edges(3)[1]] - right_pos[_edges(3)[0]],
    )["edge_feature"]
    both_pos = torch.cat([pos[:3], right_pos], dim=0)
    both_i = torch.cat([_edges(3)[0], _edges(3)[0] + 3])
    both_j = torch.cat([_edges(3)[1], _edges(3)[1] + 3])
    both = core.get_output(
        edge_embedding=torch.cat([embedding[:6], right_embed], dim=0),
        idx_i_sr=both_i,
        idx_j_sr=both_j,
        vij_sr=both_pos[both_j] - both_pos[both_i],
    )["edge_feature"]
    assert torch.allclose(both[:6], left, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[6:], right, atol=1e-6, rtol=1e-6)
    empty = core.get_output(
        edge_embedding=embedding[:0],
        idx_i_sr=idx_i[:0],
        idx_j_sr=idx_j[:0],
        vij_sr=vij[:0],
    )["edge_feature"]
    assert empty.shape == (0, 12)


def _tiny_layers():
    return [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
        },
        {"name": "PairChemicalRadial"},
        {
            "name": "Core",
            "params": {
                "num_layers": 1,
                "l_max": 1,
                "parity": True,
                "num_scalar_features": 4,
                "num_tensor_features": 2,
                "scalar_embed_mlp_depth": 1,
                "scalar_embed_mlp_width": 4,
                "latent_mlp_depth": 1,
                "latent_mlp_width": 4,
            },
        },
        {
            "name": "EdgeReadout",
            "params": {
                "output_fields": ["Eij"],
                "head_type": "variance_preserving",
                "hidden_layers_depth": 1,
                "hidden_layers_width": 4,
                "nonlinearity": "silu",
                "use_bias": False,
            },
        },
        {"name": "EdgeEnergyReduce", "params": {"pair_symmetry": True}},
        {"name": "EnergyReduce"},
        {"name": "Force"},
    ]


def _forward_stack(model, pos, za, batch_seg):
    idx_i, idx_j = _edges(pos.shape[0])
    return model(
        {
            "Ra": pos,
            "Za": za,
            "idx_i": idx_i,
            "idx_j": idx_j,
            "batch_seg": batch_seg,
        }
    )


def test_allegro_force_matches_energy_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = build_model(
        "allegro",
        layer_params=_tiny_layers(),
        build_params={
            "dim_edge": 4,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 5.0,
            "avg_num_neighbors": 2.0,
        },
        verbose=0,
    ).double()
    model.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.3, 0.9, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    za = torch.tensor([6, 1, 1], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    out = _forward_stack(model, pos, za, batch)
    analytical = out["Fa"].detach()
    assert out["E"].shape == (1,)
    assert analytical.shape == (3, 3)
    assert torch.isfinite(analytical).all()
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
            e_plus = _forward_stack(model, plus, za, batch)["E"].sum()
            e_minus = _forward_stack(model, minus, za, batch)["E"].sum()
            numerical[atom, coord] = -(e_plus - e_minus) / (2 * step)
    assert torch.allclose(analytical, numerical, atol=1e-4, rtol=1e-4)
