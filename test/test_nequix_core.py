"""Nequix: registration, radial window, RMSNorm, equivariance, optional upstream parity."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml
from e3nn import o3

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "enerzyme" / "config" / "nequix_layers_example.yaml"


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
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def test_mptrj_vocabulary_matches_published_list():
    from enerzyme.models.nequix import MPTRJ_ATOMIC_NUMBERS

    assert len(MPTRJ_ATOMIC_NUMBERS) == 89
    assert MPTRJ_ATOMIC_NUMBERS[0] == 1
    assert MPTRJ_ATOMIC_NUMBERS[-1] == 94
    assert 83 in MPTRJ_ATOMIC_NUMBERS
    assert 84 not in MPTRJ_ATOMIC_NUMBERS
    assert 89 in MPTRJ_ATOMIC_NUMBERS


def test_nequip_envelope_is_polynomial_p6():
    from enerzyme.models.cutoff import PolynomialEnvelope, nequip_envelope

    distance = torch.tensor([0.0, 1.0, 3.0, 6.0, 6.5])
    got = nequip_envelope(distance, cutoff=6.0, cuton=0.0)
    expected = PolynomialEnvelope(cutoff=6.0, exponent=6)(distance).reshape(-1)
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(got[0], torch.ones(()), atol=1e-6)
    assert torch.allclose(got[-2:], torch.zeros(2), atol=1e-6)
    assert got[2] > 0 and got[2] < 1


def test_bessel_times_nequip_envelope_matches_nequix_basis():
    from enerzyme.models.cutoff import nequip_envelope
    from enerzyme.models.layers.rbf import BesselRBFLayer

    cutoff = 6.0
    num_rbf = 4
    p = 6.0
    layer = BesselRBFLayer(
        num_rbf=num_rbf,
        cutoff_sr=cutoff,
        flavor="equflash",
        trainable=False,
        apply_cutoff_fn=False,
    )
    distances = torch.tensor([0.0, 0.4, 2.5, 5.5, 6.0])
    basis = layer.get_rbf(distances)
    window = nequip_envelope(distances, cutoff)
    frequencies = torch.arange(1, num_rbf + 1) * math.pi
    expected = []
    for dist, env in zip(distances, window):
        if dist == 0:
            radial = (2.0 / cutoff) * (frequencies / cutoff)
        else:
            radial = (2.0 / cutoff) * torch.sin(frequencies * dist / cutoff) / dist
        # r >= cutoff is removed by the envelope, including the exact boundary.
        expected.append(radial * env)
    assert torch.allclose(basis * window[:, None], torch.stack(expected), atol=1e-5)


def _reference_rms(irreps: o3.Irreps, features: torch.Tensor, centering: bool) -> torch.Tensor:
    """Independent transcription of the eSEN / Nequix RMSNorm."""
    chunks = []
    norms = []
    n_blocks = len(irreps)
    for mul, ir, sl in zip(
        [mul for mul, _ in irreps],
        [ir for _, ir in irreps],
        irreps.slices(),
    ):
        values = features[:, sl].reshape(-1, mul, ir.dim)
        if centering and ir.l == 0:
            values = values - values.mean(dim=-2, keepdim=True)
        chunks.append(values)
        weight = 1.0 / (ir.dim * n_blocks)
        block = (values.square() * weight).sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        norms.append(block)
    scale = torch.cat(norms, dim=-1).sum(dim=-1, keepdim=True)
    scale = torch.pow(scale + 1e-12, -0.5)
    return torch.cat(
        [(values * scale).reshape(features.shape[0], mul * ir.dim) for values, (mul, ir) in zip(chunks, irreps)],
        dim=-1,
    )


def test_rms_layer_norm_matches_reference_formula():
    from enerzyme.models.e3nn_nn import RMSLayerNorm

    irreps = o3.Irreps("8x0e + 4x1o + 2x2e")
    features = torch.randn(5, irreps.dim)
    for centering in (False, True):
        layer = RMSLayerNorm(irreps, centering=centering, std_balance_degrees=True)
        with torch.no_grad():
            for weight in layer.affine_weight:
                weight.uniform_(0.5, 1.5)
            if centering:
                for bias in layer.affine_bias:
                    if bias is not None:
                        bias.uniform_(-0.2, 0.2)
        raw = _reference_rms(irreps, features, centering)
        # Apply the same affine the module uses, in block order.
        chunks = []
        offset = 0
        for index, (mul, ir) in enumerate(irreps):
            width = mul * ir.dim
            block = raw[:, offset : offset + width].reshape(-1, mul, ir.dim)
            block = block * layer.affine_weight[index].detach()[:, None]
            if centering and ir.l == 0:
                block = block + layer.affine_bias[index].detach()[:, None]
            chunks.append(block.reshape(-1, width))
            offset += width
        expected = torch.cat(chunks, dim=-1)
        got = layer(features)
        assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_one_hot_vocabulary_and_unknown_species():
    from enerzyme.models.layers.atom_embedding import OneHotAtomEmbedding

    layer = OneHotAtomEmbedding(max_Za=8, dim_embedding=3, atomic_numbers=[1, 6, 8])
    encoded = layer.get_embedding(torch.tensor([8, 1, 6]))
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded, torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    with pytest.raises(ValueError, match="outside the one-hot"):
        layer.get_embedding(torch.tensor([7]))


def test_nequix_registration_and_yaml_build():
    from enerzyme.models.ff import build_model, get_ff_core
    from enerzyme.models.layers.atom_embedding import OneHotAtomEmbedding
    from enerzyme.models.layers.readout import SimpleReadout

    core_cls, build_params, layers = get_ff_core("nequix")
    assert core_cls.__name__ == "NequixCore"
    assert build_params["cutoff_fn"] == "nequip"
    assert build_params["dim_embedding"] == 89
    assert layers[1]["params"]["flavor"] == "equflash"
    assert layers[2]["name"] == "OneHotAtomEmbedding"
    assert layers[3]["params"]["avg_num_neighbors"] == pytest.approx(
        57.413687022442645
    )
    assert layers[4]["params"]["use_bias"] is False

    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    assert model.__class__.__name__ == "NequixCore"
    assert model.dim_feature_out == 8
    assert model.feature_irreps == "8x0e"
    assert model.num_layers == 2
    embedding = next(layer for layer in model.pre_sequence if isinstance(layer, OneHotAtomEmbedding))
    assert embedding.dim_embedding == 4
    readout = next(layer for layer in model.post_sequence if isinstance(layer, SimpleReadout))
    assert readout.head.bias is None
    assert readout.head.dim_feature_in == 8


def _small_conv(**overrides):
    from enerzyme.models.nequix.interaction import NequixConvolution

    params = dict(
        irreps_in="4x0e",
        irreps_out="4x0e + 2x1o",
        irreps_sh=o3.Irreps.spherical_harmonics(1),
        num_rbf=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        mlp_init_scale=4.0,
        avg_num_neighbors=1.0,
        layer_norm=True,
    )
    params.update(overrides)
    return NequixConvolution(**params)


def test_published_gate_keeps_data_scalars_then_gates():
    from enerzyme.models.nequix import PUBLISHED_MP_CORE_PARAMS
    from enerzyme.models.nequix.interaction import NequixGate, pre_gate_irreps

    hidden = o3.Irreps(PUBLISHED_MP_CORE_PARAMS["hidden_irreps"])
    pre = pre_gate_irreps(hidden)
    n_data = hidden.count("0e")
    n_gates = hidden.num_irreps - n_data
    assert pre[0] == (n_data + n_gates, o3.Irrep("0e"))
    assert all(ir.l > 0 for _, ir in list(pre)[1:])
    gate = NequixGate("3x0e + 1x1o")
    assert gate.extra_dim == 2 and gate.gate_dim == 1
    base = torch.tensor([[1.0, 2.0, 0.0, 0.4, 0.0, 0.0]])
    gated = base.clone()
    gated[0, 2] = 1.5
    base_out = gate(base)
    gated_out = gate(gated)
    assert torch.allclose(base_out[:, :2], gated_out[:, :2])
    assert not torch.allclose(base_out[:, 2:], gated_out[:, 2:])
    with pytest.raises(ValueError, match="odd scalar"):
        pre_gate_irreps("1x0o + 1x1e")
    with pytest.raises(ValueError, match="trailing scalars"):
        NequixGate("1x1o + 2x0e")


def test_linear2_keeps_sorted_unmerged_paths():
    from enerzyme.models.nequix.interaction import NequixConvolution

    conv = NequixConvolution(
        "4x0e + 2x1o",
        "4x0e",
        o3.Irreps.spherical_harmonics(1),
        num_rbf=4,
        radial_mlp_size=8,
        radial_mlp_layers=1,
        mlp_init_scale=4.0,
        avg_num_neighbors=1.0,
        layer_norm=False,
    )
    assert conv.linear_2.irreps_in == conv.tp.irreps_out
    assert conv.linear_2.irreps_in != conv.tp.irreps_out.regroup()
    assert conv.linear_1.bias_numel == 0 and conv.skip.bias_numel == 0


def test_convolution_hidden_features_rotate_with_edges():
    torch.manual_seed(0)
    conv = _small_conv()
    conv.eval()
    features = torch.randn(4, conv.irreps_in.dim)
    idx_i, idx_j = _edges(4)
    vij = torch.randn(idx_i.numel(), 3)
    radial = torch.randn(idx_i.numel(), 4)
    edge_sh = conv_sh(conv, vij)
    reference = conv(features, idx_i, idx_j, radial, edge_sh)
    rotation = _rotation(1)
    rotated = conv(features, idx_i, idx_j, radial, conv_sh(conv, vij @ rotation.T))
    transform = conv.irreps_out.D_from_matrix(rotation)
    assert torch.allclose(reference @ transform.T, rotated, atol=1e-4, rtol=1e-4)


def conv_sh(conv, vij: torch.Tensor) -> torch.Tensor:
    return o3.spherical_harmonics(
        conv.tp.irreps_in2,
        vij,
        normalize=True,
        normalization="component",
    )


def test_core_scalar_output_is_rotation_invariant_and_isolated_atom_is_finite():
    from enerzyme.models.nequix import NequixCore

    torch.manual_seed(2)
    core = NequixCore(
        dim_embedding=3,
        num_rbf=4,
        num_layers=2,
        lmax=1,
        hidden_irreps="4x0e+2x1o",
        radial_mlp_size=8,
        avg_num_neighbors=2.0,
    )
    core.eval()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 1.1, 0.0]])
    embedding = torch.eye(3)
    idx_i, idx_j = _edges(3)
    vij = pos[idx_j] - pos[idx_i]
    radial = torch.rand(idx_i.numel(), 4)
    reference = core.get_output(embedding, radial, idx_i, idx_j, vij)["atom_feature"]
    assert reference.shape == (3, 4)
    rotation = _rotation(3)
    rotated = core.get_output(embedding, radial, idx_i, idx_j, vij @ rotation.T)["atom_feature"]
    assert torch.allclose(reference, rotated, atol=1e-4, rtol=1e-4)
    alone = core.get_output(
        atom_embedding=embedding[:1],
        rbf=torch.zeros(0, 4),
        idx_i_sr=torch.zeros(0, dtype=torch.long),
        idx_j_sr=torch.zeros(0, dtype=torch.long),
        vij_sr=torch.zeros(0, 3),
    )["atom_feature"]
    assert alone.shape == (1, 4)
    assert torch.isfinite(alone).all()


def test_stack_energy_is_invariant_and_forces_rotate():
    from enerzyme.models.ff import build_model

    torch.manual_seed(4)
    with EXAMPLE.open() as stream:
        ff = yaml.safe_load(stream)["Modelhub"]["internal_FFs"]["FF01"]
    model = build_model(
        ff["architecture"],
        layer_params=ff["layers"],
        build_params=ff["build_params"],
        verbose=0,
    )
    model.eval()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.0, 0.3], [0.4, 0.2, 1.2]],
        dtype=torch.float32,
    )
    idx_i, idx_j = _edges(pos.shape[0])
    species = torch.tensor([1, 6, 7, 8])

    def run(positions: torch.Tensor):
        positions = positions.detach().requires_grad_(True)
        output = model(
            {
                "Ra": positions,
                "Za": species,
                "idx_i": idx_i,
                "idx_j": idx_j,
                "batch_seg": torch.zeros(positions.shape[0], dtype=torch.long),
            }
        )
        return output["E"], output["Fa"]

    energy, forces = run(pos)
    assert energy.shape == (1,)
    assert forces.shape == pos.shape
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()
    rotation = _rotation(5)
    energy_rot, forces_rot = run(pos @ rotation.T)
    assert torch.allclose(energy, energy_rot, atol=1e-4, rtol=1e-4)
    assert torch.allclose(forces @ rotation.T, forces_rot, atol=1e-4, rtol=1e-4)
    energy_shift, forces_shift = run(pos + torch.tensor([0.4, -0.3, 0.2]))
    assert torch.allclose(energy, energy_shift, atol=1e-4, rtol=1e-4)
    assert torch.allclose(forces, forces_shift, atol=1e-4, rtol=1e-4)


def test_convolution_matches_nequix_torch_when_installed():
    """Parity with atomicarchitects/nequix when that package can be constructed.

    e3nn 0.5.6 has no ``Irreps.filter``, which the torch port's gate calls.
    The published training path also keeps duplicate tensor-product blocks
    unmerged into ``linear_2``; the torch port regroups them, so only the
    convolution messages (before that linear) are compared.
    """
    nequix_conv = pytest.importorskip("nequix.torch_impl.model").NequixTorchConvolution
    if not hasattr(o3.Irreps, "filter"):
        pytest.skip("installed e3nn Irreps has no filter(); the torch port cannot build its gate")
    torch.manual_seed(6)
    irreps_in = o3.Irreps("4x0e")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(1)
    reference = nequix_conv(
        irreps_in,
        irreps_out,
        irreps_sh,
        n_species=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        mlp_init_scale=4.0,
        avg_n_neighbors=2.5,
        index_weights=False,
        layer_norm=True,
        kernel=False,
    )
    ours = _small_conv(avg_num_neighbors=2.5)
    with torch.no_grad():
        ours.linear_1.weight.copy_(reference.linear_1.weight)
        for left, right in zip(ours.radial_mlp.layers, reference.radial_mlp.layers):
            left.weights.copy_(right.weights)
    features = torch.randn(3, irreps_in.dim)
    senders = torch.tensor([0, 1, 2, 0])
    receivers = torch.tensor([1, 2, 0, 1])
    radial = torch.randn(4, 4)
    vectors = torch.randn(4, 3)
    edge_sh = o3.spherical_harmonics(irreps_sh, vectors, True, normalization="component")
    from torch_scatter import scatter_sum

    def aggregated(conv, idx_i, idx_j):
        sent = conv.tp(conv.linear_1(features)[idx_j], edge_sh, conv.radial_mlp(radial))
        pooled = scatter_sum(sent, idx_i, dim=0, dim_size=features.shape[0])
        sorter = getattr(conv, "sort", None)
        if sorter is not None:
            pooled = sorter(pooled)
        return pooled / math.sqrt(2.5)

    assert torch.allclose(
        aggregated(ours, receivers, senders),
        aggregated(reference, receivers, senders),
        atol=1e-5,
        rtol=1e-5,
    )
