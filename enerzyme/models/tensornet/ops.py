"""Cartesian 3×3 tensor algebra for TensorNet.

Tensors are stored ``[N, 3, 3, F]`` (channel last), the layout used by
TorchMD-Net's pure PyTorch TensorNet path. ``I`` is the isotropic
coefficient ``[N, F]``, ``A`` is skew-symmetric, and ``S`` is symmetric
and traceless. The squared Frobenius weight of ``I * Id`` is ``3 I²``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def identity_tensor(iso: Tensor) -> Tensor:
    """Broadcast a per-channel scalar to ``iso * I_3`` with shape ``[N, 3, 3, F]``."""
    eye = torch.eye(3, device=iso.device, dtype=iso.dtype)
    return iso[:, None, None, :] * eye[None, :, :, None]


def decompose_tensor(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Split a full Cartesian tensor into ``(I, A, S)``.

    ``I`` has shape ``[N, F]``. ``A`` and ``S`` keep ``[N, 3, 3, F]``.
    """
    skew = 0.5 * (tensor - tensor.transpose(1, 2))
    symmetric = tensor - skew
    iso = tensor.diagonal(dim1=1, dim2=2).mean(dim=-1)
    traceless = symmetric - identity_tensor(iso)
    return iso, skew, traceless


def compose_tensor(iso: Tensor, skew: Tensor, traceless: Tensor) -> Tensor:
    """Rebuild ``[N, 3, 3, F]`` from irreducible components."""
    return identity_tensor(iso) + skew + traceless


def frobenius_squared(tensor: Tensor) -> Tensor:
    """Squared Frobenius norm over the 3×3 axes. Shape ``[N, F]``."""
    return tensor.square().sum(dim=(1, 2))


def scalar_invariants(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``cat(3 I², ||A||², ||S||²)`` and the three components.

    ``3 I²`` is ``||I * Id||²_F``, which is the isotropic term TorchMD-Net
    concatenates before the scalar LayerNorm.
    """
    iso, skew, traceless = decompose_tensor(tensor)
    invariants = torch.cat(
        (3.0 * iso.square(), frobenius_squared(skew), frobenius_squared(traceless)),
        dim=-1,
    )
    return invariants, iso, skew, traceless


def vector_to_skew_tensor(vector: Tensor) -> Tensor:
    """Map ``[N, 3, F]`` to the skew matrix with ``A v = w × v``."""
    x = vector[:, 0]
    y = vector[:, 1]
    z = vector[:, 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=1),
            torch.stack((z, zero, -x), dim=1),
            torch.stack((-y, x, zero), dim=1),
        ),
        dim=1,
    )


def skew_tensor_to_vector(tensor: Tensor) -> Tensor:
    """Invert :func:`vector_to_skew_tensor`. Output shape ``[N, 3, F]``."""
    x = 0.5 * (tensor[:, 2, 1, :] - tensor[:, 1, 2, :])
    y = 0.5 * (tensor[:, 0, 2, :] - tensor[:, 2, 0, :])
    z = 0.5 * (tensor[:, 1, 0, :] - tensor[:, 0, 1, :])
    return torch.stack((x, y, z), dim=1)


def symmetric_traceless(tensor: Tensor) -> Tensor:
    """Symmetrize a ``[N, 3, 3, F]`` tensor and remove its isotropic part."""
    symmetric = 0.5 * (tensor + tensor.transpose(1, 2))
    iso = tensor.diagonal(dim1=1, dim2=2).mean(dim=-1)
    return symmetric - identity_tensor(iso)


def tensor_matmul(left: Tensor, right: Tensor) -> Tensor:
    """Batched matrix product on channel-last 3×3 tensors."""
    product = torch.matmul(left.permute(0, 3, 1, 2), right.permute(0, 3, 1, 2))
    return product.permute(0, 2, 3, 1)


def tensor_anticommutator(left: Tensor, right: Tensor) -> Tensor:
    """``right @ left + left @ right``. O(3) interaction product."""
    return tensor_matmul(right, left) + tensor_matmul(left, right)


def unit_vectors(vij: Tensor, dist: Tensor) -> Tensor:
    """``vij / ||vij||``, with the zero vector left at zero (self pairs)."""
    positive = dist > 0
    hat = vij / dist.clamp_min(1e-12).unsqueeze(-1)
    return torch.where(positive.unsqueeze(-1), hat, torch.zeros_like(vij))
