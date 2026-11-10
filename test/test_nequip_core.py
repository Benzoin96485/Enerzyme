"""NequIP Core: radial basis, convolution equivariance, energy/force stack."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "nequip_layers_example.yaml"


def _edges(n: int):
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                idx_i.append(i)
                idx_j.append(j)
    return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(q) < 0:
        q = q.clone()
        q[:, 0] *= -1
    return q


def _graph(n: int, n_rbf: int, width: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3, generator=generator)
    vij = pos[idx_j] - pos[idx_i]
    return {
        "atom_embedding": torch.randn(n, width, generator=generator),
        "rbf": torch.randn(idx_i.numel(), n_rbf, generator=generator),
        "idx_i_sr": idx_i,
        "idx_j_sr": idx_j,
        "vij_sr": vij,
        "pos": pos,
    }


def test_bessel_nequip_matches_sinc_envelope():
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 5.0
    degree = 6
    layer = BesselRBFLayer(
        num_rbf=4,
        cutoff_sr=cutoff,
        flavor="nequip",
        trainable=False,
        polynomial_p=degree,
    )
    assert layer.apply_cutoff_fn is False
    distances = torch.tensor([0.0, 0.4, 1.5, 4.9, 5.0, 6.2])
    got = layer.get_rbf(distances)
    a = -(degree + 1) * (degree + 2) / 2
    b = degree * (degree + 2)
    c = -degree * (degree + 1) / 2
    scale = 2.0 * math.pi / (cutoff * cutoff)
    expected = []
    for dist in distances.tolist():
        x = dist / cutoff
        if x >= 1.0:
            expected.append(torch.zeros(4))
            continue
        env = 1.0 + a * x**degree + b * x ** (degree + 1) + c * x ** (degree + 2)
        row = []
        for n in range(1, 5):
            row.append(scale * n * torch.sinc(torch.tensor(n * x)) * env)
        expected.append(torch.stack(row))
    expected = torch.stack(expected)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6)

    try:
        BesselRBFLayer(num_rbf=2, cutoff_sr=cutoff, flavor="nequip", polynomial_p=1)
    except ValueError:
        pass
    else:
        raise AssertionError("polynomial_p < 2 should be rejected")


def test_nequip_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.readout import SimpleReadout

    for name in ("nequip", "NequIP"):
        core_cls, build_params, layers = get_ff_core(name)
        assert core_cls.__name__ == "NequIPCore"
        assert build_params["num_rbf"] == 8
        assert layers[1]["params"]["flavor"] == "nequip"
        assert layers[1]["name"] == "BesselRBF"

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    layers = ff["layers"]
    core_params = next(layer for layer in layers if layer["name"] == "Core")
    core_params = {
        "name": "Core",
        "params": dict(core_params["params"], num_layers=2, num_features=4, radial_mlp_width=8),
    }
    small_layers = [
        core_params if layer["name"] == "Core" else layer for layer in layers
    ]
    model = build_model(
        ff["architecture"],
        layer_params=small_layers,
        build_params=dict(ff["build_params"], dim_embedding=4, num_rbf=3, max_Za=20),
        verbose=0,
    )
    assert model.__class__.__name__ == "NequIPCore"
    assert model.num_layers == 2
    assert model.dim_feature_out == 4
    assert model.feature_irreps == "4x0e"
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head_type == "dense"
    assert readout.head.bias is None


def test_preset_sets_depth_and_explicit_args_win():
    from enerzyme.models.nequip.core import NequIPCore, _resolve

    paper = _resolve("paper", {})
    assert paper["radial_mlp_depth"] == 2
    assert paper["parity"] is True
    small = _resolve("S", {"num_features": 4, "radial_mlp_width": 8})
    assert small["num_layers"] == 2
    assert small["l_max"] == 1
    assert small["parity"] is False
    assert small["num_features"] == 4

    core = NequIPCore(
        dim_embedding=4,
        num_rbf=3,
        preset="S",
        num_features=4,
        radial_mlp_width=8,
    )
    assert core.num_layers == 2
    assert core.l_max == 1
    assert core.parity is False
    assert len(core.convs) == 2


def test_conv_is_rotation_and_inversion_equivariant():
    from enerzyme.models.nequip.interaction import NequIPConv

    torch.manual_seed(0)
    irreps_in = o3.Irreps("4x0e+3x1o")
    hidden = o3.Irreps("4x0e+4x0o+3x1o+3x1e")
    edge = o3.Irreps.spherical_harmonics(1)
    conv = NequIPConv(
        irreps_in=irreps_in,
        feature_irreps_hidden=hidden,
        irreps_edge_attr=edge,
        irreps_node_attr=o3.Irreps("4x0e"),
        num_rbf=4,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        avg_num_neighbors=2.0,
        use_sc=True,
        resnet=False,
        nonlinearity_type="gate",
    ).double()
    conv.eval()
    n = 5
    features = irreps_in.randn(n, -1).double()
    attrs = torch.randn(n, 4, dtype=torch.float64)
    idx_i, idx_j = _edges(n)
    pos = torch.randn(n, 3, dtype=torch.float64)
    vij = pos[idx_j] - pos[idx_i]
    rbf = torch.randn(idx_i.numel(), 4, dtype=torch.float64)
    sph = o3.SphericalHarmonics(edge, normalize=True, normalization="component")
    out = conv(features, attrs, idx_j, idx_i, sph(vij), rbf)

    rotation = _rotation(1).double()
    d_in = irreps_in.D_from_matrix(rotation)
    d_out = conv.irreps_out.D_from_matrix(rotation)
    rotated = conv(features @ d_in, attrs, idx_j, idx_i, sph(vij @ rotation), rbf)
    assert torch.allclose(out @ d_out, rotated, atol=1e-5, rtol=1e-5)

    inversion = -torch.eye(3, dtype=torch.float64)
    d_in_p = irreps_in.D_from_matrix(inversion)
    d_out_p = conv.irreps_out.D_from_matrix(inversion)
    inverted = conv(features @ d_in_p, attrs, idx_j, idx_i, sph(vij @ inversion), rbf)
    assert torch.allclose(out @ d_out_p, inverted, atol=1e-5, rtol=1e-5)


def test_core_scalars_are_invariant_and_molecules_stay_separate():
    from enerzyme.models.nequip import NequIPCore

    torch.manual_seed(1)
    width, n_rbf = 4, 3
    core = NequIPCore(
        dim_embedding=width,
        num_rbf=n_rbf,
        num_layers=2,
        l_max=1,
        parity=True,
        num_features=4,
        radial_mlp_depth=1,
        radial_mlp_width=8,
        avg_num_neighbors=1.0,
    ).double()
    core.eval()
    graph = _graph(5, n_rbf, width, seed=2)
    for key in ("atom_embedding", "rbf", "vij_sr", "pos"):
        graph[key] = graph[key].double()

    def _run(pos, embedding=None):
        idx_i, idx_j = graph["idx_i_sr"], graph["idx_j_sr"]
        return core.get_output(
            atom_embedding=graph["atom_embedding"] if embedding is None else embedding,
            rbf=graph["rbf"],
            idx_i_sr=idx_i,
            idx_j_sr=idx_j,
            vij_sr=pos[idx_j] - pos[idx_i],
        )["atom_feature"]

    base = _run(graph["pos"])
    assert base.shape == (5, 4)
    assert core.feature_irreps == "4x0e"
    rotation = _rotation(3).double()
    assert torch.allclose(_run(graph["pos"] @ rotation), base, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        _run(graph["pos"] + torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)),
        base,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(_run(-graph["pos"]), base, atol=1e-5, rtol=1e-5)

    left = _graph(3, n_rbf, width, seed=4)
    right = _graph(4, n_rbf, width, seed=5)
    for graph_i in (left, right):
        for key in ("atom_embedding", "rbf", "vij_sr"):
            graph_i[key] = graph_i[key].double()
    separate_l = core.get_output(
        atom_embedding=left["atom_embedding"],
        rbf=left["rbf"],
        idx_i_sr=left["idx_i_sr"],
        idx_j_sr=left["idx_j_sr"],
        vij_sr=left["vij_sr"],
    )["atom_feature"]
    separate_r = core.get_output(
        atom_embedding=right["atom_embedding"],
        rbf=right["rbf"],
        idx_i_sr=right["idx_i_sr"],
        idx_j_sr=right["idx_j_sr"],
        vij_sr=right["vij_sr"],
    )["atom_feature"]
    n_left = left["atom_embedding"].shape[0]
    both = core.get_output(
        atom_embedding=torch.cat([left["atom_embedding"], right["atom_embedding"]]),
        rbf=torch.cat([left["rbf"], right["rbf"]]),
        idx_i_sr=torch.cat([left["idx_i_sr"], right["idx_i_sr"] + n_left]),
        idx_j_sr=torch.cat([left["idx_j_sr"], right["idx_j_sr"] + n_left]),
        vij_sr=torch.cat([left["vij_sr"], right["vij_sr"]]),
    )["atom_feature"]
    assert torch.allclose(both[:n_left], separate_l, atol=1e-6, rtol=1e-6)
    assert torch.allclose(both[n_left:], separate_r, atol=1e-6, rtol=1e-6)


def test_isolated_atom_is_zero_until_self_connection_includes_the_first_layer():
    from enerzyme.models.nequip import NequIPCore

    torch.manual_seed(2)
    kwargs = dict(
        dim_embedding=4,
        num_rbf=3,
        num_layers=3,
        l_max=1,
        num_features=4,
        radial_mlp_width=8,
    )
    plain = NequIPCore(**kwargs)
    plain.eval()
    embedding = torch.randn(1, 4)
    empty = dict(
        atom_embedding=embedding,
        rbf=torch.zeros(0, 3),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
    )
    alone = plain.get_output(**empty)["atom_feature"]
    assert alone.shape == (1, 4)
    assert torch.count_nonzero(alone) == 0

    torch.manual_seed(2)
    with_first = NequIPCore(**kwargs, sc_first=True)
    with_first.eval()
    shifted = with_first.get_output(**empty)["atom_feature"]
    assert torch.isfinite(shifted).all()
    assert not torch.allclose(shifted, torch.zeros_like(shifted))

    graph = _graph(4, 3, 4, seed=7)
    without = plain.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
    )["atom_feature"]
    torch.manual_seed(2)
    no_sc = NequIPCore(**kwargs, use_sc=False)
    no_sc.eval()
    ablated = no_sc.get_output(
        atom_embedding=graph["atom_embedding"],
        rbf=graph["rbf"],
        idx_i_sr=graph["idx_i_sr"],
        idx_j_sr=graph["idx_j_sr"],
        vij_sr=graph["vij_sr"],
    )["atom_feature"]
    assert not torch.allclose(without, ablated, atol=1e-5, rtol=1e-5)


def _tiny_layers():
    return [
        {"name": "RangeSeparation"},
        {
            "name": "BesselRBF",
            "params": {"flavor": "nequip", "trainable": False, "polynomial_p": 6},
        },
        {"name": "RandomAtomEmbedding"},
        {
            "name": "Core",
            "params": {
                "num_layers": 2,
                "l_max": 1,
                "parity": True,
                "num_features": 4,
                "radial_mlp_depth": 1,
                "radial_mlp_width": 8,
            },
        },
        {
            "name": "SimpleReadout",
            "params": {
                "output_fields": ["Ea"],
                "head_type": "dense",
                "use_bias": False,
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


def test_nequip_force_matches_energy_finite_difference():
    from enerzyme.models.ff import build_model

    torch.manual_seed(6)
    model = build_model(
        "nequip",
        layer_params=_tiny_layers(),
        build_params={
            "dim_embedding": 4,
            "num_rbf": 3,
            "max_Za": 10,
            "cutoff_sr": 5.0,
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
    energy = out["E"].sum()
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
    assert torch.isfinite(energy)
