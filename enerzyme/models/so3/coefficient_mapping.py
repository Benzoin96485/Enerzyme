"""Coefficient indexing helpers for spherical harmonic embeddings (l <-> m).

Adapted from fairchem v1 eSCN (Passaro & Zitnick, 2023; MIT license).
"""

from __future__ import annotations

import torch


class CoefficientMapping:
    """Maps between degree-major and order-major layouts of SH coefficients.

    Args:
        lmax_list: Maximum degree ``l`` for each resolution.
        mmax_list: Maximum order ``m`` for each resolution.
        device: Device for index tensors.
    """

    def __init__(
        self,
        lmax_list: list[int],
        mmax_list: list[int],
        device: torch.device,
    ) -> None:
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.device = device

        self.l_harmonic = torch.tensor([], device=self.device).long()
        self.m_harmonic = torch.tensor([], device=self.device).long()
        self.m_complex = torch.tensor([], device=self.device).long()

        self.res_size = torch.zeros([self.num_resolutions], device=self.device).long()
        offset = 0
        for i in range(self.num_resolutions):
            for lval in range(self.lmax_list[i] + 1):
                mmax = min(self.mmax_list[i], lval)
                m = torch.arange(-mmax, mmax + 1, device=self.device).long()
                self.m_complex = torch.cat([self.m_complex, m], dim=0)
                self.m_harmonic = torch.cat(
                    [self.m_harmonic, torch.abs(m).long()], dim=0
                )
                self.l_harmonic = torch.cat(
                    [self.l_harmonic, m.fill_(lval).long()], dim=0
                )
            self.res_size[i] = len(self.l_harmonic) - offset
            offset = len(self.l_harmonic)

        num_coefficients = len(self.l_harmonic)
        self.to_m = torch.zeros(
            [num_coefficients, num_coefficients], device=self.device
        )
        self.m_size = torch.zeros([max(self.mmax_list) + 1], device=self.device).long()

        offset = 0
        for m in range(max(self.mmax_list) + 1):
            idx_r, idx_i = self.complex_idx(m)

            for idx_out, idx_in in enumerate(idx_r):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_r)
            self.m_size[m] = int(len(idx_r))

            for idx_out, idx_in in enumerate(idx_i):
                self.to_m[idx_out + offset, idx_in] = 1.0
            offset = offset + len(idx_i)

        self.to_m = self.to_m.detach()

    def complex_idx(self, m: int, lmax: int = -1):
        """Return masks for real / imaginary coefficients of order ``m``."""
        if lmax == -1:
            lmax = max(self.lmax_list)

        indices = torch.arange(len(self.l_harmonic), device=self.device)
        mask_r = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(m))
        mask_idx_r = torch.masked_select(indices, mask_r)

        mask_idx_i = torch.tensor([], device=self.device).long()
        if m != 0:
            mask_i = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_complex.eq(-m))
            mask_idx_i = torch.masked_select(indices, mask_i)

        return mask_idx_r, mask_idx_i

    def coefficient_idx(self, lmax: int, mmax: int) -> torch.Tensor:
        """Return indices of coefficients with degree ≤ ``lmax`` and order ≤ ``mmax``."""
        mask = torch.bitwise_and(self.l_harmonic.le(lmax), self.m_harmonic.le(mmax))
        indices = torch.arange(len(mask), device=self.device)
        return torch.masked_select(indices, mask)
