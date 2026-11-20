"""TensorNet Cartesian algebra and TorchMD-Net pure-PyTorch op parity."""

from __future__ import annotations

import torch


def _rotation(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.det(matrix) < 0:
        matrix = matrix.clone()
        matrix[:, 0] *= -1
    return matrix


def _reference_decompose(tensor):
    """Transcription of torchmdnet.models.tensornet._decompose_tensor."""
    skew = 0.5 * (tensor - tensor.transpose(1, 2))
    symmetric = tensor - skew
    iso = tensor.diagonal(dim1=1, dim2=2).mean(-1)
    eye = torch.eye(3, device=tensor.device, dtype=tensor.dtype)[None, :, :, None]
    symmetric = symmetric - iso[:, None, None, :] * eye
    return iso, skew, symmetric


def _reference_matmul_o3(node, message):
    left = torch.matmul(message.permute(0, 3, 1, 2), node.permute(0, 3, 1, 2))
    right = torch.matmul(node.permute(0, 3, 1, 2), message.permute(0, 3, 1, 2))
    return left.permute(0, 2, 3, 1) + right.permute(0, 2, 3, 1)


def _reference_skew(vector):
    n, _, width = vector.shape
    zero = torch.zeros((n, width), device=vector.device, dtype=vector.dtype)
    stacked = torch.stack(
        (
            zero,
            -vector[..., 2, :],
            vector[..., 1, :],
            vector[..., 2, :],
            zero,
            -vector[..., 0, :],
            -vector[..., 1, :],
            vector[..., 0, :],
            zero,
        ),
        dim=1,
    )
    return stacked.view(n, 3, 3, width)


def test_decompose_matches_torchmd_and_roundtrips():
    from enerzyme.models.tensornet.ops import compose_tensor, decompose_tensor

    torch.manual_seed(0)
    tensor = torch.randn(5, 3, 3, 4)
    iso, skew, traceless = decompose_tensor(tensor)
    ref_iso, ref_skew, ref_traceless = _reference_decompose(tensor)
    assert torch.allclose(iso, ref_iso)
    assert torch.allclose(skew, ref_skew)
    assert torch.allclose(traceless, ref_traceless)
    assert torch.allclose(compose_tensor(iso, skew, traceless), tensor, atol=1e-6)
    assert torch.allclose(skew, -skew.transpose(1, 2), atol=1e-6)
    assert torch.allclose(traceless, traceless.transpose(1, 2), atol=1e-6)
    assert torch.allclose(
        traceless.diagonal(dim1=1, dim2=2).sum(-1), torch.zeros_like(iso), atol=1e-6
    )


def test_skew_roundtrip_matches_torchmd_layout():
    from enerzyme.models.tensornet.ops import skew_tensor_to_vector, vector_to_skew_tensor

    torch.manual_seed(1)
    vector = torch.randn(6, 3, 3)
    skew = vector_to_skew_tensor(vector)
    assert torch.allclose(skew, _reference_skew(vector))
    assert torch.allclose(skew_tensor_to_vector(skew), vector, atol=1e-6)
    axis = torch.tensor([[[1.0], [0.0], [0.0]]])
    matrix = vector_to_skew_tensor(axis)[0, :, :, 0]
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert torch.allclose(matrix, expected)


def test_unit_outer_product_is_traceless_quadrupole():
    from enerzyme.models.tensornet.interaction import aggregate_edge_tensors

    hat = torch.tensor([[1.0, 0.0, 0.0]])
    processed = torch.ones(1, 3, 1)
    receiver = torch.zeros(1, dtype=torch.long)
    iso, skew, traceless = aggregate_edge_tensors(hat, processed, receiver, 2)
    assert torch.allclose(iso[0], torch.tensor([1.0]))
    assert torch.allclose(iso[1], torch.tensor([0.0]))
    assert torch.allclose(skew[1], torch.zeros(3, 3, 1))
    expected = torch.tensor([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    assert torch.allclose(traceless[0, :, :, 0].diagonal(), expected, atol=1e-6)
    assert torch.allclose(traceless[0, :, :, 0].sum(), torch.zeros(()), atol=1e-6)


def test_isotropic_norm_is_three_times_square():
    from enerzyme.models.tensornet.ops import frobenius_squared, identity_tensor, scalar_invariants

    iso = torch.tensor([[0.5, -2.0]])
    eye = identity_tensor(iso)
    assert torch.allclose(frobenius_squared(eye), 3.0 * iso.square())
    invariants, recovered, skew, traceless = scalar_invariants(eye)
    assert torch.allclose(recovered, iso)
    assert torch.count_nonzero(skew) == 0
    assert torch.count_nonzero(traceless) == 0
    assert invariants.shape == (1, 6)
    assert torch.allclose(invariants[:, :2], 3.0 * iso.square())


def test_o3_product_matches_torchmd_and_rotates():
    from enerzyme.models.tensornet.ops import tensor_anticommutator, tensor_matmul

    torch.manual_seed(2)
    node = torch.randn(3, 3, 3, 2)
    message = torch.randn(3, 3, 3, 2)
    assert torch.allclose(
        tensor_anticommutator(node, message), _reference_matmul_o3(node, message)
    )
    rotation = _rotation(3)

    def _rotate(tensor):
        # X' = R.T @ X @ R, the image of a matrix when row vectors map as p @ R.
        return torch.einsum("ia,nacf,cj->nijf", rotation.T, tensor, rotation)

    rotated = tensor_anticommutator(_rotate(node), _rotate(message))
    assert torch.allclose(
        _rotate(tensor_anticommutator(node, message)), rotated, atol=1e-5
    )
    assert not torch.allclose(tensor_matmul(node, message), tensor_anticommutator(node, message))
