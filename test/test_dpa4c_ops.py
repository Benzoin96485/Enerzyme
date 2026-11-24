"""DPA4C harmonics, Grams, bispectrum, radial map, and fitting residual."""

import math

import torch
from numpy.testing import assert_allclose


def test_degree_profile_and_published_widths():
    from enerzyme.models.dpa4c.geometry import (
        bispectrum_feature_count,
        derive_bispectrum_ranks,
        derive_degree_channels,
        invariant_width,
    )

    assert derive_degree_channels(8, 2) == [8, 4, 4]
    assert derive_degree_channels(32, 2) == [32, 8, 4]
    assert derive_degree_channels(64, 3) == [64, 8, 4, 1]
    assert derive_degree_channels(128, 4) == [128, 16, 8, 1, 1]
    ranks = derive_bispectrum_ranks(derive_degree_channels(32, 2))
    assert ranks == [4, 2]
    assert invariant_width(32, 2) == 144
    assert invariant_width(8, 2) == 70
    assert bispectrum_feature_count(2, [4, 2]) == 24

    try:
        derive_degree_channels(24, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("channels outside the profile set should fail")
    try:
        derive_degree_channels(8, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("lmax outside {2, 3, 4} should fail")


def test_harmonics_addition_theorem_and_stf_isometry():
    from enerzyme.models.dpa4c.geometry import (
        cartesian_harmonics,
        legendre,
        packed_l2_to_stf,
    )

    torch.manual_seed(0)
    left = torch.randn(6, 3, dtype=torch.float64)
    right = torch.randn(6, 3, dtype=torch.float64)
    left = left / left.norm(dim=-1, keepdim=True)
    right = right / right.norm(dim=-1, keepdim=True)
    basis_left = cartesian_harmonics(left, 4)
    basis_right = cartesian_harmonics(right, 4)
    cosine = (left * right).sum(-1)
    for degree in range(5):
        block_left = basis_left[:, degree ** 2 : (degree + 1) ** 2]
        block_right = basis_right[:, degree ** 2 : (degree + 1) ** 2]
        dots = (block_left * block_right).sum(-1)
        torch.testing.assert_close(dots, legendre(degree, cosine), atol=1e-10, rtol=1e-10)

    packed_a = torch.randn(5, 5, dtype=torch.float64)
    packed_b = torch.randn(5, 5, dtype=torch.float64)
    matrix_a = packed_l2_to_stf(packed_a)
    matrix_b = packed_l2_to_stf(packed_b)
    frobenius = (matrix_a * matrix_b).sum(dim=(-1, -2))
    torch.testing.assert_close(frobenius, (packed_a * packed_b).sum(-1), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        matrix_a.diagonal(dim1=-2, dim2=-1).sum(-1),
        torch.zeros(5, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_envelope_is_flat_at_the_cutoff_and_raw_bessel_is_not():
    from enerzyme.models.mp.basis.envelope import C3CutoffEnvelope
    from enerzyme.models.mp.basis.radial import BesselC3RadialBasis

    radius = torch.tensor([0.0, 6.0], dtype=torch.float64, requires_grad=True)
    envelope = C3CutoffEnvelope(6.0, exponent=5).double()
    values = envelope(radius)
    grad = torch.autograd.grad(values.sum(), radius)[0]
    assert_allclose(values[0].detach().numpy(), 1.0, atol=1e-12)
    assert_allclose(values[1].detach().numpy(), 0.0, atol=1e-12)
    assert_allclose(grad[1].detach().numpy(), 0.0, atol=1e-8)

    raw = BesselC3RadialBasis(6.0, n_radial=4, apply_envelope=False).double()
    covered = BesselC3RadialBasis(6.0, n_radial=4, apply_envelope=True).double()
    covered.load_state_dict(raw.state_dict())
    far = torch.tensor([6.0], dtype=torch.float64)
    assert torch.count_nonzero(raw(far)) > 0
    assert_allclose(covered(far).detach().numpy(), 0.0, atol=1e-12)


def test_swiglu_width_and_pair_bounds():
    from enerzyme.models.dpa4c.radial import (
        OrderedPairFiLM,
        swiglu_gate_first,
        swiglu_hidden_width,
    )

    assert swiglu_hidden_width(8) == 24
    assert swiglu_hidden_width(32) == 88
    assert swiglu_hidden_width(64) == 176  # 8 * ceil(64 / 3)
    pre = torch.tensor([[1.0, -1.0, 2.0, 4.0]])
    gate, value = pre.chunk(2, dim=-1)
    torch.testing.assert_close(swiglu_gate_first(pre), torch.nn.functional.silu(gate) * value)

    film = OrderedPairFiLM(8, radial_modes=2)
    center = torch.randn(5, 8)
    neighbor = torch.randn(5, 8)
    scale, shift, mixing = film(center, neighbor)
    assert scale.shape == (5, 8)
    assert shift.shape == (5, 8)
    assert mixing.shape == (5, 8, 2)
    assert torch.all(scale > 0) and torch.all(scale < 2)
    assert torch.all((shift - center - neighbor).abs() < 1)


def test_vector_tensor_matches_manual_contraction():
    from enerzyme.models.mp.equivariant.body_order import vector_tensor_invariants

    torch.manual_seed(1)
    vector = torch.randn(3, 3, 4, dtype=torch.float64)
    packed = torch.randn(3, 5, 2, dtype=torch.float64)
    cubic, quartic = vector_tensor_invariants(vector, packed)
    assert cubic.shape == (3, 4 * 4 * 2)
    assert quartic.shape == (3, 2 * 4)
    assert torch.isfinite(cubic).all() and torch.isfinite(quartic).all()
    # One probe pair: v^T Q v and |Q v|^2.
    from enerzyme.models.dpa4c.geometry import packed_l2_to_stf

    matrix = packed_l2_to_stf(packed[0, :, 0])
    vector_0 = vector[0, :, 0]
    manual_cubic = -vector_0.dot(matrix @ vector_0) / math.sqrt(5.0)
    manual_quartic = (matrix @ vector_0).square().sum()
    # Layout (κ1, κ2, η) with η fastest, so (0, 0, 0) is entry 0.
    torch.testing.assert_close(cubic[0, 0], manual_cubic, atol=1e-10, rtol=1e-10)
    # Quartic is η-major then κ, so η=0, κ=0 is entry 0.
    torch.testing.assert_close(quartic[0, 0], manual_quartic, atol=1e-10, rtol=1e-10)


def test_gram_half_vectorization_preserves_frobenius():
    from enerzyme.models.dpa4c.readout import InvariantReadout

    readout = InvariantReadout(8, 2).double()
    block = torch.randn(2, 3, 4, dtype=torch.float64)
    gram = torch.matmul(block.transpose(-1, -2), block)
    emitted = readout._grams(
        [torch.zeros(2, 1, 8, dtype=torch.float64), block, torch.randn(2, 5, 4, dtype=torch.float64)]
    )[0]
    frobenius = gram.square().sum(dim=(-1, -2)).sqrt()
    torch.testing.assert_close(emitted.norm(dim=-1), frobenius, atol=1e-10, rtol=1e-10)


def test_mlp_identity_residual_matches_widths():
    from torch.nn import Module

    from enerzyme.models.layers.readout import SimpleReadout

    class _Previous(Module):
        dim_feature_out = 4
        feature_irreps = "4x0e"

    plain = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Previous()],
        head_type="mlp",
        hidden_dims=[4],
        activation_fn="swish",
        activation_params={"initial_beta": 1.0, "learnable": False},
        identity_residual=False,
        use_bias_out=False,
    )
    residual = SimpleReadout(
        output_fields={"Ea"},
        built_layers=[_Previous()],
        head_type="mlp",
        hidden_dims=[4],
        activation_fn="swish",
        activation_params={"initial_beta": 1.0, "learnable": False},
        identity_residual=True,
        use_bias_out=False,
    )
    residual.head[0].layer.load_state_dict(plain.head[0].state_dict())
    residual.head[1].load_state_dict(plain.head[1].state_dict())
    feature = torch.randn(3, 4)
    assert not torch.allclose(plain.get_output(feature)["Ea"], residual.get_output(feature)["Ea"])
