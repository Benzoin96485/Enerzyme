# Adapted from IQuestLab/UBio-MolFM (MIT)
# https://github.com/IQuestLab/UBio-MolFM
"""On-the-fly Wigner-D frames for E2Former-V2 SO2 / EAAS attention.

Uses the shared e3nn/Jd backend in :mod:`enerzyme.models.so3.wigner_jd` together
with UBio-MolFM's Euler convention (z-align absolute positions). Do **not** mix
with eSCN ``init_edge_rot_mat`` frames unless a numerical parity test exists.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..so3.wigner_jd import wigner_D


def init_edge_rot_euler_angles(
    edge_distance_vec: Tensor,
    training: bool = True,
    ts: float = 1e-5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """ZYZ Euler angles that align the local z-axis with ``edge_distance_vec``.

    During training a random roll (gamma) is sampled; at eval gamma is zero for
    deterministic inference.
    """
    edge_vec_0 = edge_distance_vec
    edge_vec_0_distance = torch.norm(edge_vec_0, dim=1)
    mask = edge_vec_0_distance < ts
    if edge_vec_0_distance.numel() > 0 and bool(torch.min(edge_vec_0_distance) < ts):
        edge_vec_0_distance = torch.where(
            edge_vec_0_distance < ts,
            torch.ones_like(edge_vec_0_distance),
            edge_vec_0_distance,
        )

    xyz = edge_vec_0 / edge_vec_0_distance.view(-1, 1)
    # Boolean OR (not +): keep mask as bool for safe advanced indexing.
    mask = mask | xyz[:, 1].abs().isclose(xyz.new_ones(1))

    # float32 normalize can leave |y| slightly outside [-1, 1]; clamp for acos.
    y_clamped = xyz[:, 1].clamp(-1.0, 1.0)

    beta = xyz.new_zeros(xyz.shape[0])
    beta[~mask] = torch.acos(y_clamped[~mask])
    beta[mask] = torch.acos(y_clamped[mask]).detach()

    alpha = torch.zeros_like(beta)
    alpha[~mask] = torch.atan2(xyz[~mask, 0], xyz[~mask, 2])
    alpha[mask] = torch.atan2(xyz[mask, 0], xyz[mask, 2]).detach()

    if training:
        gamma = torch.rand_like(alpha) * 2 * torch.pi
    else:
        gamma = torch.zeros_like(alpha)

    # Intrinsic → extrinsic swap (UBio-MolFM / fairchem convention).
    return -gamma, -beta, -alpha


def eulers_to_wigner(
    eulers: Tuple[Tensor, Tensor, Tensor],
    start_lmax: int,
    end_lmax: int,
    l3_sequential: Optional[Sequence] = None,
) -> Tuple[Tensor, Tensor]:
    """Block-diagonal Wigner-D and its transpose (inverse for real D).

    When ``l3_sequential`` is set (SO2 TP path multiplicity), ``wigner_inv`` is
    sized to the expanded SO2 feature layout rather than ``(L+1)^2``.
    """
    alpha, beta, gamma = eulers
    size = int((end_lmax + 1) ** 2) - int(start_lmax**2)
    wigner = torch.zeros(
        len(alpha), size, size, device=alpha.device, dtype=alpha.dtype
    )
    start = 0
    for lval in range(start_lmax, end_lmax + 1):
        block = wigner_D(lval, alpha, beta, gamma)
        end = start + block.size()[1]
        wigner[:, start:end, start:end] = block
        start = end

    if l3_sequential is not None:
        s = sum((2 * tmp_l3 + 1) * tmp_l3_cnt for tmp_l3, tmp_l3_cnt in l3_sequential)
        start = 0
        wigner_inv = torch.zeros(
            len(alpha), s, s, device=alpha.device, dtype=alpha.dtype
        )
        for tmp_l3, tmp_l3_cnt in l3_sequential:
            block = wigner_D(tmp_l3, alpha, beta, gamma)
            for _ in range(tmp_l3_cnt):
                wigner_inv[
                    :, start : start + tmp_l3 * 2 + 1, start : start + tmp_l3 * 2 + 1
                ] = block
                start += tmp_l3 * 2 + 1
    else:
        wigner_inv = wigner
    return wigner, torch.transpose(wigner_inv, 1, 2).contiguous()


def build_so2_wigner_frames(
    node_pos: Tensor,
    exp_node_pos: Tensor,
    lmax: int,
    l3_sequential: Optional[List] = None,
    training: bool = False,
    wigner_dtype: torch.dtype = torch.float64,
) -> dict:
    """Compute ``wigner`` / ``wigner_inv`` (and exp-side) for SO2 attention.

    Positions should already be COM-centered (see ``graph.center_positions_by_batch``).
    """
    pos_w = node_pos.to(dtype=wigner_dtype)
    exp_w = exp_node_pos.to(dtype=wigner_dtype)
    out_dtype = node_pos.dtype

    wigner, wigner_inv = eulers_to_wigner(
        init_edge_rot_euler_angles(pos_w, training=training),
        0,
        lmax,
        l3_sequential,
    )
    wigner = wigner.to(dtype=out_dtype)
    wigner_inv = wigner_inv.to(dtype=out_dtype)

    if exp_node_pos.data_ptr() == node_pos.data_ptr() or torch.equal(
        exp_node_pos, node_pos
    ):
        wigner_exp, wigner_inv_exp = wigner, wigner_inv
    else:
        wigner_exp, wigner_inv_exp = eulers_to_wigner(
            init_edge_rot_euler_angles(exp_w, training=training),
            0,
            lmax,
            l3_sequential,
        )
        wigner_exp = wigner_exp.to(dtype=out_dtype)
        wigner_inv_exp = wigner_inv_exp.to(dtype=out_dtype)

    return {
        "wigner": wigner,
        "wigner_inv": wigner_inv,
        "wigner_exp": wigner_exp,
        "wigner_inv_exp": wigner_inv_exp,
    }
