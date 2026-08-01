"""Shared e3nn / Jd Wigner-D backend for edge-frame SO(3) rotations.

Canonical packed layout is e3nn degree-major with ``m = -l..+l``
(index ``l^2 + l + m``). Consumers adapt this tensor:

* :class:`~enerzyme.models.so3.rotation.SO3_Rotation` — row/col masks + optional rescale
* :class:`~enerzyme.models.so3.rotation_fused.SO3RotationFused` — m-primary fuse
* :class:`~enerzyme.models.so3.wigner_quaternion.WignerDCalculator` — quaternion frames → R → packed W

Adapted from fairchem v1 eSCN / e3nn 0.4.0 (MIT).
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from e3nn import o3

# Borrowed from e3nn @ 0.4.0:
# https://github.com/e3nn/e3nn/blob/0.4.0/e3nn/o3/_wigner.py#L10
_Jd = torch.load(
    os.path.join(os.path.dirname(__file__), "Jd.pt"),
    map_location="cpu",
    weights_only=False,
)


def max_wigner_lmax() -> int:
    """Largest spherical degree supported by the packaged ``Jd.pt`` tables."""
    return int(len(_Jd) - 1)


def _check_lmax(lval: int) -> None:
    if not 0 <= int(lval) <= max_wigner_lmax():
        raise NotImplementedError(
            f"wigner D maximum l implemented is {max_wigner_lmax()}, got l={lval}"
        )


def z_rot_mat(angle: torch.Tensor, lval: int) -> torch.Tensor:
    """Z-axis rotation matrix of size ``(2l+1, 2l+1)`` batched over ``angle``."""
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * lval + 1, 2 * lval + 1))
    inds = torch.arange(0, 2 * lval + 1, 1, device=device)
    reversed_inds = torch.arange(2 * lval, -1, -1, device=device)
    frequencies = torch.arange(lval, -lval - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M


def wigner_D(
    lval: int,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Single-degree Wigner-D block ``(N, 2l+1, 2l+1)`` from ZYZ Euler angles."""
    _check_lmax(lval)
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[lval].to(dtype=alpha.dtype, device=alpha.device)
    Xa = z_rot_mat(alpha, lval)
    Xb = z_rot_mat(beta, lval)
    Xc = z_rot_mat(gamma, lval)
    return Xa @ J @ Xb @ J @ Xc


def rotation_matrix_to_euler(
    edge_rot_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ZYZ Euler angles from edge rotation matrices ``(E, 3, 3)``.

    Matches the eSCN / Equiformer recipe: project the local y-axis, then recover γ.
    """
    x = edge_rot_mat[:, :, 1]
    alpha, beta = o3.xyz_to_angles(x)
    R = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
    R = torch.bmm(R, edge_rot_mat)
    gamma = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return alpha, beta, gamma


def wigner_from_rotation_matrix(
    edge_rot_mat: torch.Tensor,
    end_lmax: int,
    start_lmax: int = 0,
) -> torch.Tensor:
    """Block-diagonal packed Wigner-D ``(E, D, D)`` with ``D = (end_lmax+1)^2 - start_lmax^2``.

    Layout is e3nn degree-major packed coefficients. Gradients through
    ``edge_rot_mat`` are preserved for EnergyReduce+Force stacks.
    """
    end_lmax = int(end_lmax)
    start_lmax = int(start_lmax)
    if start_lmax < 0 or end_lmax < start_lmax:
        raise ValueError(
            f"Invalid l range: start_lmax={start_lmax}, end_lmax={end_lmax}"
        )
    _check_lmax(end_lmax)

    alpha, beta, gamma = rotation_matrix_to_euler(edge_rot_mat)
    size = (end_lmax + 1) ** 2 - start_lmax**2
    # Assemble via padded adds (out-of-place) so the edge-frame graph stays connected.
    wigner = edge_rot_mat.new_zeros(len(alpha), size, size)
    offset = 0
    for lval in range(start_lmax, end_lmax + 1):
        block = wigner_D(lval, alpha, beta, gamma)
        sz = block.shape[-1]
        pad_before = offset
        pad_after = size - offset - sz
        wigner = wigner + torch.nn.functional.pad(
            block, (pad_before, pad_after, pad_before, pad_after)
        )
        offset += sz
    return wigner
