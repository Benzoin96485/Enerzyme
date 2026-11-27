"""EQNorm norm, convolution wiring, and parity with yzchen08/eqnorm."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet

from enerzyme.models.eqnorm.interaction import EqnormConv, eqnorm_tp_paths
from enerzyme.models.eqnorm.layernorm import EqnormRMSNorm


UPSTREAM = Path(os.environ.get("EQNORM_SRC", "/tmp/eqnorm-src"))


def _reference_channel_norm(irreps: o3.Irreps, features: torch.Tensor, centering: bool) -> torch.Tensor:
    """Channel-mean form of the EQNorm RMS, without the reference padding trick."""
    parts = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        parts.append(features[:, offset : offset + width].reshape(-1, mul, ir.dim))
        offset += width
    if centering:
        parts[0] = parts[0] - parts[0].mean(dim=1, keepdim=True)
    max_mul = max(mul for mul, _ in irreps)
    rows = []
    for channel in range(max_mul):
        total = features.new_zeros(features.shape[0])
        count = 0
        for (mul, _), block in zip(irreps, parts):
            if channel < mul:
                total = total + block[:, channel, :].square().mean(dim=-1)
                count += 1
        rows.append(total / count)
    scale = (torch.stack(rows, dim=1).mean(dim=1, keepdim=True) + 1e-12).pow(-0.5)
    return scale, parts


def test_rms_matches_channel_mean_formula():
    irreps = o3.Irreps("8x0e + 4x1o + 2x2e")
    features = torch.randn(6, irreps.dim)
    for centering in (False, True):
        layer = EqnormRMSNorm(irreps, centering=centering, affine=True)
        with torch.no_grad():
            for weight in layer.affine_weight:
                weight.uniform_(0.5, 1.5)
            if centering:
                layer.affine_bias.uniform_(-0.2, 0.2)
        scale, parts = _reference_channel_norm(irreps, features, centering)
        chunks = []
        for index, ((mul, ir), block) in enumerate(zip(irreps, parts)):
            weight = layer.affine_weight[index].detach().view(1, mul, 1)
            scaled = block * scale[:, None, :] * weight
            if centering and ir.l == 0:
                scaled = scaled + layer.affine_bias.detach().view(1, mul, 1)
            chunks.append(scaled.reshape(features.shape[0], mul * ir.dim))
        expected = torch.cat(chunks, dim=-1)
        assert torch.allclose(layer(features), expected, atol=1e-5, rtol=1e-5)


def test_rms_output_rotates_with_the_feature():
    irreps = o3.Irreps("4x0e + 3x1o + 2x2e")
    layer = EqnormRMSNorm(irreps, centering=False)
    features = torch.randn(5, irreps.dim)
    rotation = o3.rand_matrix()
    transform = irreps.D_from_matrix(rotation)
    rotated = layer(features @ transform.T)
    assert torch.allclose(layer(features) @ transform.T, rotated, atol=1e-5, rtol=1e-5)


def test_rms_rejects_a_missing_degree():
    with pytest.raises(ValueError, match="one irrep"):
        EqnormRMSNorm("4x0e + 2x2e")


def _load_upstream(which: str):
    if not (UPSTREAM / "eqnorm" / "layernorm.py").is_file():
        pytest.skip(f"yzchen08/eqnorm checkout not found at {UPSTREAM}")
    root = str(UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)
    import eqnorm.layernorm  # noqa: F401  registers the package for relative imports

    filename = {
        "mptrj": "eqnorm-mptrj.py",
        "omat": "eqnorm-omat.py",
        "max": "eqnorm-max-mptrj.py",
    }[which]
    spec = importlib.util.spec_from_file_location(
        f"eqnorm_ref_{which}", UPSTREAM / "eqnorm" / filename
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "eqnorm"
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rms_matches_upstream_module():
    if not (UPSTREAM / "eqnorm" / "layernorm.py").is_file():
        pytest.skip(f"yzchen08/eqnorm checkout not found at {UPSTREAM}")
    root = str(UPSTREAM)
    if root not in sys.path:
        sys.path.insert(0, root)
    from eqnorm.layernorm import RMSLayerNorm as UpstreamNorm

    irreps = o3.Irreps("8x0e + 4x1o + 2x2e")
    features = torch.randn(4, irreps.dim)
    for centering in (False, True):
        reference = UpstreamNorm(irreps, affine=True, centering=centering)
        ours = EqnormRMSNorm(irreps, affine=True, centering=centering)
        assert torch.allclose(ours.balance_weight, reference.balance_weight, atol=1e-6)
        ours.load_state_dict(reference.state_dict())
        assert torch.allclose(ours(features), reference(features), atol=1e-5, rtol=1e-5)


def _copy_conv(reference, ours: EqnormConv) -> None:
    ours.load_state_dict(reference.state_dict())


def test_path_order_matches_upstream_get_path():
    module = _load_upstream("mptrj")
    irreps_in = o3.Irreps("4x0e+2x1o")
    irreps_sh = o3.Irreps.spherical_harmonics(1)
    irreps_out = o3.Irreps("8x0e+2x1o")
    ours, our_instructions = eqnorm_tp_paths(irreps_in, irreps_sh, irreps_out)
    reference, ref_instructions = module.get_path(irreps_in, irreps_sh, irreps_out)
    assert ours == reference
    assert our_instructions == ref_instructions


def test_mptrj_conv_matches_upstream():
    module = _load_upstream("mptrj")
    torch.manual_seed(0)
    reference = module.E3Conv(
        "4x0e",
        "1x0e+1x1o",
        "6x0e+2x1o",
        num_basis=4,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=2.5,
        mode="nonscalar",
    )
    ours = EqnormConv(
        "4x0e",
        "1x0e+1x1o",
        "6x0e+2x1o",
        num_rbf=4,
        invariant_layers=1,
        invariant_neurons=8,
        normalization="constant",
        avg_num_neighbors=2.5,
    )
    _copy_conv(reference, ours)
    features = torch.randn(3, 4)
    idx_i = torch.tensor([0, 1, 2, 1])
    idx_j = torch.tensor([1, 2, 0, 0])
    vij = torch.randn(4, 3)
    radial = torch.rand(4, 4)
    edge_sh = o3.spherical_harmonics(
        "1x0e+1x1o", vij, normalize=True, normalization="component"
    )
    data = {
        "node_hiddens": features,
        "edge_index": torch.stack([idx_i, idx_j]),
        "edge_sh": edge_sh,
        "edge_attr": radial,
    }
    reference(data)
    got = ours(features, idx_i, idx_j, radial, edge_sh)
    assert torch.allclose(got, data["node_hiddens"], atol=1e-5, rtol=1e-5)


def test_degree_and_envelope_match_upstream_variants():
    torch.manual_seed(1)
    features = torch.randn(3, 4)
    idx_i = torch.tensor([0, 0, 1])
    idx_j = torch.tensor([1, 2, 2])
    vij = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    radial = torch.rand(3, 4)
    edge_sh = o3.spherical_harmonics(
        "1x0e+1x1o", vij, normalize=True, normalization="component"
    )

    omat = _load_upstream("omat")
    reference = omat.E3Conv(
        "4x0e",
        "1x0e+1x1o",
        "4x0e+2x1o",
        num_basis=4,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        mode="nonscalar",
    )
    ours = EqnormConv(
        "4x0e",
        "1x0e+1x1o",
        "4x0e+2x1o",
        num_rbf=4,
        invariant_layers=1,
        invariant_neurons=8,
        normalization="degree",
        avg_num_neighbors=None,
    )
    _copy_conv(reference, ours)
    data = {
        "node_hiddens": features.clone(),
        "edge_index": torch.stack([idx_i, idx_j]),
        "edge_sh": edge_sh,
        "edge_attr": radial,
    }
    reference(data)
    got = ours(features, idx_i, idx_j, radial, edge_sh)
    assert torch.allclose(got, data["node_hiddens"], atol=1e-5, rtol=1e-5)

    maximum = _load_upstream("max")
    reference = maximum.E3Conv(
        "4x0e",
        "1x0e+1x1o",
        "4x0e+2x1o",
        num_basis=4,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=4.0,
        mode="scalar",
    )
    ours = EqnormConv(
        "4x0e",
        "1x0e+1x1o",
        "4x0e+2x1o",
        num_rbf=4,
        invariant_layers=1,
        invariant_neurons=8,
        normalization="constant",
        avg_num_neighbors=4.0,
    )
    _copy_conv(reference, ours)
    distance = torch.linalg.vector_norm(vij, dim=-1)
    envelope = maximum.PolynomialCutoff(r_max=5.0, p=6)(distance)
    data = {
        "node_hiddens": features.clone(),
        "edge_index": torch.stack([idx_i, idx_j]),
        "edge_sh": edge_sh,
        "edge_attr": radial,
        "envelope": envelope.unsqueeze(-1),
    }
    reference(data)
    got = ours(features, idx_i, idx_j, radial, edge_sh, envelope)
    assert torch.allclose(got, data["node_scalars"], atol=1e-5, rtol=1e-5)


def test_zero_envelope_matches_an_empty_neighborhood():
    torch.manual_seed(2)
    conv = EqnormConv(
        "4x0e",
        "1x0e+1x1o",
        "4x0e",
        num_rbf=3,
        invariant_layers=1,
        invariant_neurons=5,
        normalization="constant",
        avg_num_neighbors=9.0,
    )
    features = torch.randn(2, 4)
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 0])
    radial = torch.rand(2, 3)
    vij = torch.randn(2, 3)
    edge_sh = o3.spherical_harmonics(
        "1x0e+1x1o", vij, normalize=True, normalization="component"
    )
    no_edges = conv(
        features,
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 3),
        torch.zeros(0, conv.irreps_sh.dim),
    )
    assert torch.allclose(
        conv(features, idx_i, idx_j, radial, edge_sh, torch.zeros(2)),
        no_edges,
        atol=1e-5,
        rtol=1e-5,
    )


def test_radial_network_is_the_e3nn_fully_connected_net():
    conv = EqnormConv(
        "2x0e",
        "1x0e",
        "2x0e",
        num_rbf=4,
        invariant_layers=2,
        invariant_neurons=6,
        normalization="constant",
        avg_num_neighbors=1.0,
    )
    assert isinstance(conv.fc, FullyConnectedNet)
    assert conv.fc.hs == [4, 6, 6, conv.tp.weight_numel]
    assert conv.linear_1.bias_numel == 0
    assert conv.linear_2.irreps_in == conv.tp.irreps_out.simplify()
