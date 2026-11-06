"""Algebraic checks for FACTO, which has no public reference implementation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def test_facto_polynomial_matches_closed_form_and_derivative():
    from enerzyme.models.cutoff import facto_pair_window, facto_polynomial_cutoff

    cutoff = 6.0
    distance = torch.tensor([0.0, 1e-6, 0.5, 3.0, 5.9, 6.0, 7.0], dtype=torch.float64)
    window = facto_polynomial_cutoff(distance, cutoff)
    assert window[0].item() == 1.0
    assert window[-2].item() == 0.0
    assert window[-1].item() == 0.0

    interior = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    radius = interior * cutoff
    radius.requires_grad_(True)
    values = facto_polynomial_cutoff(radius, cutoff)
    expected = (
        1
        - 84 * interior**6
        + 216 * interior**7
        - 189 * interior**8
        + 56 * interior**9
    )
    assert torch.allclose(values, expected, atol=1e-12, rtol=0)
    values.sum().backward()
    analytic = 504 * interior**5 * (interior - 1) ** 3 / cutoff
    assert torch.allclose(radius.grad, analytic, atol=1e-10, rtol=1e-10)
    assert torch.all(radius.grad < 0)

    short = 1.0
    grid = torch.tensor([0.0, 0.2, 1.0, 2.0, 6.0], dtype=torch.float64)
    chi = facto_pair_window(grid, cutoff, short)
    outer = facto_polynomial_cutoff(grid, cutoff)
    inner = facto_polynomial_cutoff(grid, short)
    assert torch.allclose(chi, outer * (1 - inner), atol=1e-12, rtol=0)
    assert chi[0].item() == 0.0
    # At r = Rs the inner window is closed, so χ equals the outer cutoff.
    assert torch.allclose(chi[2], outer[2], atol=1e-12, rtol=0)
    assert chi[-1].item() == 0.0


def test_gaussian_centers_width_and_filters_vanish_with_the_window():
    from enerzyme.models.facto.geometry import GAUSSIAN_WIDTH_NUMERATOR, FACTOGeometry

    geometry = FACTOGeometry(
        scalar_dim=8, n_radial=5, channels=(4, 2), cutoff=6.0, r_short=1.0
    )
    assert torch.allclose(geometry.centers, torch.tensor([0, 0.25, 0.5, 0.75, 1.0]))
    assert geometry.width == GAUSSIAN_WIDTH_NUMERATOR / 4
    assert geometry.scalar_filter.bias is not None
    assert all(layer.bias is None for layer in geometry.tensor_filters)

    distance = torch.tensor([0.0, 0.4, 2.0, 6.0])
    displacement = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 2.0, 0.0], [6.0, 0.0, 0.0]]
    )
    edge = geometry(distance, displacement)
    assert edge.rho.shape == (4, 8)
    assert edge.xi[0].shape == (4, 4)
    assert edge.harmonics[0].shape == (4, 3)
    assert edge.harmonics[1].shape == (4, 5)
    assert torch.allclose(edge.chi[[0, 3]], torch.zeros(2), atol=1e-12)
    assert torch.allclose(edge.rho[[0, 3]], torch.zeros(2, 8), atol=1e-6)
    assert torch.allclose(edge.xi[0][[0, 3]], torch.zeros(2, 4), atol=1e-6)
    assert torch.isfinite(edge.harmonics[0]).all()
    # Between Rs and Rc the inner window is closed, so χ is the outer cutoff.
    from enerzyme.models.cutoff import facto_polynomial_cutoff

    assert torch.allclose(
        edge.chi[2], facto_polynomial_cutoff(distance[2], 6.0), atol=1e-12
    )


def test_descriptor_table_is_31_and_chemically_typed():
    from enerzyme.models.facto.descriptors import (
        DESCRIPTOR_COLUMNS,
        DESCRIPTOR_DIM,
        elemental_descriptor_table,
    )

    table = elemental_descriptor_table(94)
    assert len(DESCRIPTOR_COLUMNS) == DESCRIPTOR_DIM == 31
    assert table.shape == (95, 31)
    assert torch.count_nonzero(table[0]) == 0
    columns = {name: index for index, name in enumerate(DESCRIPTOR_COLUMNS)}
    hydrogen = table[1]
    carbon = table[6]
    iron = table[26]
    assert abs(hydrogen[columns["z/100"]].item() - 0.01) < 1e-6
    assert abs(hydrogen[columns["period/7"]].item() - 1 / 7) < 1e-6
    assert abs(hydrogen[columns["group/18"]].item() - 1 / 18) < 1e-6
    assert abs(hydrogen[columns["pauling_en/4"]].item() - 2.2 / 4) < 1e-6
    assert carbon[columns["block_p"]].item() == 1.0
    assert carbon[columns["nonmetal"]].item() == 1.0
    assert iron[columns["block_d"]].item() == 1.0
    assert iron[columns["transition"]].item() == 1.0
    block = table[1:, columns["block_s"] : columns["block_f"] + 1]
    series = table[1:, columns["alkali"] :]
    assert torch.equal(block.sum(dim=-1), torch.ones(94))
    assert torch.equal(series.sum(dim=-1), torch.ones(94))
    try:
        elemental_descriptor_table(95)
    except ValueError as exc:
        assert "94" in str(exc)
    else:
        raise AssertionError("Z above 94 should be rejected")


def test_coupled_rms_uses_scalar_variance_in_the_tensor_denominator():
    from enerzyme.models.facto.interaction import CoupledRMSNorm

    torch.manual_seed(0)
    norm = CoupledRMSNorm(4, (3,), epsilon=1e-6)
    with torch.no_grad():
        norm.scalar_gain.copy_(torch.tensor([1.0, 2.0, 0.5, 1.5]))
        norm.tensor_gain[0].copy_(torch.tensor([0.5, 1.0, 1.5]))
    scalars = torch.randn(2, 4)
    tensors = torch.randn(2, 3, 3)
    scalars_out, tensors_out = norm(scalars, [tensors])
    variance = scalars.square().mean(dim=-1, keepdim=True)
    expected_h = norm.scalar_gain * scalars / torch.sqrt(variance + norm.epsilon)
    angular = tensors.square().mean(dim=1)
    expected_t = norm.tensor_gain[0] * tensors / torch.sqrt(
        variance + angular + norm.epsilon
    ).unsqueeze(1)
    assert torch.allclose(scalars_out, expected_h, atol=1e-6, rtol=1e-6)
    assert torch.allclose(tensors_out[0], expected_t, atol=1e-6, rtol=1e-6)

    zeros = torch.zeros_like(tensors)
    _, zero_out = norm(scalars, [zeros])
    assert torch.allclose(zero_out[0], torch.zeros_like(zeros), atol=1e-6)


def test_factorized_attention_matches_the_edge_sum():
    from enerzyme.models.facto.interaction import FactorizedScalarAttention

    torch.manual_seed(1)
    attention = FactorizedScalarAttention(8, 2).double()
    scalars = torch.randn(4, 8, dtype=torch.float64)
    rho = torch.randn(5, 8, dtype=torch.float64)
    idx_i = torch.tensor([0, 0, 1, 2, 3])
    idx_j = torch.tensor([1, 2, 0, 3, 1])
    got = attention(scalars, rho, idx_i, idx_j)

    query = attention.query(scalars).view(4, 2, 4)
    source = F.silu(attention.key(scalars)) * attention.value(scalars)
    aggregated = torch.zeros(4, 2, 4, dtype=torch.float64)
    radial = rho.view(5, 2, 4)
    for edge in range(5):
        aggregated[idx_i[edge]] += radial[edge] * source[idx_j[edge]]
    mixed = (F.silu(query) * aggregated).reshape(4, 8)
    expected = F.silu(attention.mix(mixed))
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)

    empty_i = torch.zeros(0, dtype=torch.long)
    empty = attention(scalars, rho[:0], empty_i, empty_i)
    assert torch.isfinite(empty).all()
    assert empty.shape == (4, 8)


def test_feedback_injection_and_transport_match_the_paper_reductions():
    from enerzyme.models.facto.interaction import (
        GeometricTensorInjection,
        TensorToScalarFeedback,
        TensorTransport,
    )

    torch.manual_seed(2)
    scalar_dim = 6
    channels = (4,)
    intermediate = torch.randn(3, scalar_dim, dtype=torch.float64)
    tensors = torch.randn(3, 3, 4, dtype=torch.float64)
    feedback = TensorToScalarFeedback(scalar_dim, channels).double()
    got = feedback([tensors], intermediate)
    left = feedback.left[0](tensors)
    right = feedback.right[0](tensors)
    invariant = (left * right).sum(dim=1)
    expected = feedback.mix(invariant) * feedback.to_gate(intermediate)
    expected = expected + feedback.to_residual(intermediate)
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)
    assert feedback.left[0].bias is None

    zero = feedback([torch.zeros_like(tensors)], intermediate)
    bias_only = feedback.to_residual(intermediate)
    bias_only = feedback.mix(torch.zeros(3, 4, dtype=torch.float64)) * feedback.to_gate(
        intermediate
    ) + feedback.to_residual(intermediate)
    assert torch.allclose(zero, bias_only, atol=1e-10, rtol=1e-10)

    injection = GeometricTensorInjection(scalar_dim, channels).double()
    xi = torch.randn(2, 4, dtype=torch.float64)
    harmonics = torch.randn(2, 3, dtype=torch.float64)
    idx_i = torch.tensor([0, 1])
    idx_j = torch.tensor([1, 2])
    injected = injection(intermediate, [xi], [harmonics], idx_i, idx_j)[0]
    manual = torch.zeros(3, 3, 4, dtype=torch.float64)
    source = injection.project[0](intermediate)
    for edge in range(2):
        manual[idx_i[edge]] += (source[idx_j[edge]] * xi[edge]).unsqueeze(0) * harmonics[
            edge
        ].unsqueeze(-1)
    assert torch.allclose(injected, manual, atol=1e-10, rtol=1e-10)

    transport = TensorTransport(scalar_dim, channels).double()
    chi = torch.tensor([0.4, 0.8], dtype=torch.float64)
    carried = transport([tensors], intermediate, chi, [harmonics], idx_i, idx_j)[0]
    manual = torch.zeros_like(manual)
    aligned = transport.align[0](tensors)
    payload = transport.message[0](tensors)
    gate = transport.gate[0](intermediate)
    for edge in range(2):
        score = (aligned[idx_j[edge]] * harmonics[edge].unsqueeze(-1)).sum(dim=0)
        gated = chi[edge] * gate[idx_j[edge]] * score
        manual[idx_i[edge]] += gated.unsqueeze(0) * payload[idx_j[edge]]
    assert torch.allclose(carried, manual, atol=1e-10, rtol=1e-10)
    assert transport.align[0].bias is None
    assert transport.message[0].bias is None


def test_residual_scale_is_the_paper_constant():
    from enerzyme.models.facto.interaction import RESIDUAL_SCALE

    assert abs(RESIDUAL_SCALE - 40.0 ** -0.5) < 1e-15
    assert abs(RESIDUAL_SCALE - 1 / math.sqrt(40.0)) < 1e-15
