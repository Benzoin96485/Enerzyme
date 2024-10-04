import os, math
from typing import Optional
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from ... import segment_sum_coo
from .. import BaseFFLayer
from ...functional import softplus_inverse
from ...cutoff import smooth_transition


"""
computes D4 dispersion energy
HF      s6=1.00000000, s8=1.61679827, a1=0.44959224, a2=3.35743605
"""


class GrimmeD4EnergyLayer(BaseFFLayer):
    def __init__(
        self,
        cutoff: Optional[float] = None,
        s6: float = 1.00000000,
        s8: float = 1.61679827,
        a1: float = 0.44959224,
        a2: float = 3.35743605,
        g_a: float = 3.0,
        g_c: float = 2.0,
        k2: float = 1.3333333333333333,  # 4/3
        k4: float = 4.10451,
        k5: float = 19.08857,
        k6: float = 254.5553148552,  # 2*11.28174**2
        kn: float = 7.5,
        wf: float = 6.0,
        max_Za: int = 87,
        Hartree_in_E: float=1, Bohr_in_R: float=0.5291772108
    ) -> None:
        """ Initializes the D4DispersionEnergy class. """
        super().__init__(output_fields={"E_disp_a"})
        # Grimme's D4 dispersion is only parametrized up to Rn (Z=86)
        assert max_Za <= 87
        # trainable parameters
        self.register_parameter(
            "_s6", Parameter(softplus_inverse(s6), requires_grad=False)
        )  # s6 is usually not fitted (correct long-range)
        self.register_parameter(
            "_s8", Parameter(softplus_inverse(s8), requires_grad=True)
        )
        self.register_parameter(
            "_a1", Parameter(softplus_inverse(a1), requires_grad=True)
        )
        self.register_parameter(
            "_a2", Parameter(softplus_inverse(a2), requires_grad=True)
        )
        self.register_parameter(
            "_scaleq", Parameter(softplus_inverse(1.0), requires_grad=True)
        )  # for scaling charges of reference systems
        # D4 constants
        self.Zmax = max_Za
        self.Hartree_in_E = Hartree_in_E
        self.Bohr_in_R = Bohr_in_R  # factor of 0.5 prevents double counting
        self.set_cutoff(cutoff)
        self.g_a = g_a
        self.g_c = g_c
        self.k2 = k2
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.kn = kn
        self.wf = wf
        # load D4 data
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grimme_d4_tables")
        self.register_buffer(
            "refsys",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refsys.pth"))[:max_Za],
        )
        self.register_buffer(
            "zeff", torch.load(os.path.join(directory, "zeff.pth"))[:max_Za]  # [Zmax]
        )
        self.register_buffer(
            "refh",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refh.pth"))[:max_Za],
        )
        self.register_buffer(
            "sscale", torch.load(os.path.join(directory, "sscale.pth"))  # [18]
        )
        self.register_buffer(
            "secaiw", torch.load(os.path.join(directory, "secaiw.pth"))  # [18,23]
        )
        self.register_buffer(
            "gam", torch.load(os.path.join(directory, "gam.pth"))[:max_Za]  # [Zmax]
        )
        self.register_buffer(
            "ascale",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "ascale.pth"))[:max_Za],
        )
        self.register_buffer(
            "alphaiw",  # [Zmax,max_nref,23]
            torch.load(os.path.join(directory, "alphaiw.pth"))[:max_Za],
        )
        self.register_buffer(
            "hcount",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "hcount.pth"))[:max_Za],
        )
        self.register_buffer(
            "casimir_polder_weights",  # [23]
            torch.load(os.path.join(directory, "casimir_polder_weights.pth"))[:max_Za],
        )
        self.register_buffer(
            "rcov", torch.load(os.path.join(directory, "rcov.pth"))[:max_Za]  # [Zmax]
        )
        self.register_buffer(
            "en", torch.load(os.path.join(directory, "en.pth"))[:max_Za]  # [Zmax]
        )
        self.register_buffer(
            "ncount_mask",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "ncount_mask.pth"))[:max_Za],
        )
        self.register_buffer(
            "ncount_weight",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "ncount_weight.pth"))[:max_Za],
        )
        self.register_buffer(
            "cn",  # [Zmax,max_nref,max_ncount]
            torch.load(os.path.join(directory, "cn.pth"))[:max_Za],
        )
        self.register_buffer(
            "fixgweights",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "fixgweights.pth"))[:max_Za],
        )
        self.register_buffer(
            "refq",  # [Zmax,max_nref]
            torch.load(os.path.join(directory, "refq.pth"))[:max_Za],
        )
        self.register_buffer(
            "sqrt_r4r2",  # [Zmax]
            torch.load(os.path.join(directory, "sqrt_r4r2.pth"))[:max_Za],
        )
        self.register_buffer(
            "alpha",  # [Zmax,max_nref,23]
            torch.load(os.path.join(directory, "alpha.pth"))[:max_Za],
        )
        self.max_nref = self.refsys.size(-1)
        self._compute_refc6()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def set_cutoff(self, cutoff: Optional[float] = None) -> None:
        """ Can be used to change the cutoff. """
        if cutoff is None:
            self.cutoff = None
            self.cuton = None
        else:
            self.cutoff = cutoff / self.Bohr_in_R
            self.cuton = self.cutoff - self.Bohr_in_R

    def _compute_refc6(self) -> None:
        """
        Function to compute the refc6 tensor. Important: If the charges of
        reference systems are scaled and the scaleq parameter changes (e.g.
        during training), then the refc6 tensor must be recomputed for correct
        results.
        """
        with torch.no_grad():
            allZ = torch.arange(self.Zmax)
            is_ = self.refsys[allZ, :]
            iz = self.zeff[is_]
            refh = self.refh[allZ, :] * F.softplus(self._scaleq)
            qref = iz
            qmod = iz + refh
            ones_like_qmod = torch.ones_like(qmod)
            qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
            alpha = (
                self.sscale[is_].view(-1, self.max_nref, 1)
                * self.secaiw[is_]
                * torch.where(
                    qmod > 1e-8,
                    torch.exp(
                        self.g_a
                        * (1 - torch.exp(self.gam[is_] * self.g_c * (1 - qref / qmod_)))
                    ),
                    math.exp(self.g_a) * ones_like_qmod,
                ).view(-1, self.max_nref, 1)
            )
            alpha = torch.max(
                self.ascale[allZ, :].view(-1, self.max_nref, 1)
                * (
                    self.alphaiw[allZ, :, :]
                    - self.hcount[allZ, :].view(-1, self.max_nref, 1) * alpha
                ),
                torch.zeros_like(alpha),
            )
            alpha_expanded = alpha.view(
                alpha.size(0), 1, alpha.size(1), 1, -1
            ) * alpha.view(1, alpha.size(0), 1, alpha.size(1), -1)
            self.register_buffer(
                "refc6",
                3.0
                / math.pi
                * torch.sum(
                    alpha_expanded * self.casimir_polder_weights.view(1, 1, 1, 1, -1),
                    -1,
                ),
                persistent=False,
            )

    def get_E_disp_a(self, Za: Tensor, Qa: Tensor, Dij_lr: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        # initialization of Zi/Zj and unit conversion
        Dij_lr_ = Dij_lr / self.Bohr_in_R  # convert distances to Bohr
        Zi = Za[idx_i]
        Zj = Za[idx_j]

        # calculate coordination numbers
        rco = self.k2 * (self.rcov[Zi] + self.rcov[Zj])
        den = self.k4 * torch.exp(
            -((torch.abs(self.en[Zi] - self.en[Zj]) + self.k5) ** 2) / self.k6
        )
        tmp = den * 0.5 * (1.0 + torch.erf(-self.kn * (Dij_lr_ - rco) / rco))
        if self.cutoff is not None:
            tmp = tmp * smooth_transition(Dij_lr_, self.cutoff, self.cuton)

        covcn = segment_sum_coo(tmp, idx_i, dim_size=len(Za))

        # calculate gaussian weights
        gweights = torch.sum(
            self.ncount_mask[Za]
            * torch.exp(
                -self.wf
                * self.ncount_weight[Za]
                * (covcn.view(-1, 1, 1) - self.cn[Za]) ** 2
            ),
            -1,
        )
        norm = torch.sum(gweights, -1, True)
        # norm_ is used to prevent nans in backwards pass
        norm_ = torch.where(norm > 1e-8, norm, torch.ones_like(norm))
        gweights = torch.where(norm > 1e-8, gweights / norm_, self.fixgweights[Za])

        # calculate dispersion energy
        iz = self.zeff[Za].view(-1, 1)
        refq = self.refq[Za] * F.softplus(self._scaleq)
        qref = iz + refq
        qmod = iz + Qa.view(-1, 1).expand(-1, self.refq.size(1))
        ones_like_qmod = torch.ones_like(qmod)
        qmod_ = torch.where(qmod > 1e-8, qmod, ones_like_qmod)
        zeta = (
            torch.where(
                qmod > 1e-8,
                torch.exp(
                    self.g_a
                    * (
                        1
                        - torch.exp(
                            self.gam[Za].view(-1, 1) * self.g_c * (1 - qref / qmod_)
                        )
                    )
                ),
                math.exp(self.g_a) * ones_like_qmod,
            )
            * gweights
        )
        if zeta.device.type == "cpu":  # indexing is faster on CPUs
            zetai = zeta[idx_i]
            zetaj = zeta[idx_j]
        else:  # gathering is faster on GPUs
            zetai = torch.gather(zeta, 0, idx_i.view(-1, 1).expand(-1, zeta.size(1)))
            zetaj = torch.gather(zeta, 0, idx_j.view(-1, 1).expand(-1, zeta.size(1)))
        refc6ij = self.refc6[Zi, Zj, :, :]
        zetaij = zetai.view(zetai.size(0), zetai.size(1), 1) * zetaj.view(
            zetaj.size(0), 1, zetaj.size(1)
        )
        c6ij = torch.sum((refc6ij * zetaij).view(refc6ij.size(0), -1), -1)
        sqrt_r4r2ij = math.sqrt(3) * self.sqrt_r4r2[Zi] * self.sqrt_r4r2[Zj]
        a1 = F.softplus(self._a1)
        a2 = F.softplus(self._a2)
        r0 = a1 * sqrt_r4r2ij + a2
        if self.cutoff is None:
            oor6 = 1 / (Dij_lr_ ** 6 + r0 ** 6)
            oor8 = 1 / (Dij_lr_ ** 8 + r0 ** 8)
        else:
            cut2 = self.cutoff ** 2
            cut6 = cut2 ** 3
            cut8 = cut2 * cut6
            tmp6 = r0 ** 6
            tmp8 = r0 ** 8
            cut6tmp6 = cut6 + tmp6
            cut8tmp8 = cut8 + tmp8
            tmpc = Dij_lr_ / self.cutoff - 1
            oor6 = (
                1 / (Dij_lr_ ** 6 + tmp6) - 1 / cut6tmp6 + 6 * cut6 / cut6tmp6 ** 2 * tmpc
            )
            oor8 = (
                1 / (Dij_lr_ ** 8 + tmp8) - 1 / cut8tmp8 + 8 * cut8 / cut8tmp8 ** 2 * tmpc
            )
            oor6 = torch.where(Dij_lr_ < self.cutoff, oor6, torch.zeros_like(oor6))
            oor8 = torch.where(Dij_lr_ < self.cutoff, oor8, torch.zeros_like(oor8))
        s6 = F.softplus(self._s6)
        s8 = F.softplus(self._s8)
        pairwise = -c6ij * (s6 * oor6 + s8 * sqrt_r4r2ij ** 2 * oor8) * self.Hartree_in_E / 2
        edisp = segment_sum_coo(pairwise, idx_i, dim_size=len(Za))
        return edisp
