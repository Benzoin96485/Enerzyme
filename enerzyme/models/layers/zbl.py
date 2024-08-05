from typing import Optional, Literal
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init
import torch.nn.functional as F
from ..functional import softplus_inverse, segment_sum
from ..cutoff import CUTOFF_REGISTER


class ZBLRepulsionEnergyLayer(Module):
    """
    Short-range repulsive potential with learnable parameters inspired by the
    Ziegler-Biersack-Littmark (ZBL) potential described in Ziegler, J.F.,
    Biersack, J.P., and Littmark, U., "The stopping and range of ions in
    solids".

    Arguments:
        a0 (float):
            Bohr radius in chosen length units (default value corresponds to
            lengths in Angstrom).
        ke (float):
            Coulomb constant in chosen unit system (default value corresponds to
            lengths in Angstrom and energy in electronvolt).
    """
    def __init__(
        self, Bohr_in_R: float=0.5291772105638411, Hartree_in_E: float=1, cutoff_sr: Optional[float]=None,
        cutoff_fn: Optional[Literal["polynomial", "bump"]]=None
    ) -> None:
        """ Initializes the ZBLRepulsionEnergy class. """
        super().__init__()
        self.a0 = Bohr_in_R
        self.kehalf = 0.5 * Bohr_in_R * Hartree_in_E
        if cutoff_fn is not None:
            self.cutoff_fn = CUTOFF_REGISTER[cutoff_fn]
            self.cutoff_sr = cutoff_sr
        self.register_parameter("_adiv", Parameter(torch.Tensor(1)))
        self.register_parameter("_apow", Parameter(torch.Tensor(1)))
        self.register_parameter("_c1", Parameter(torch.Tensor(1)))
        self.register_parameter("_c2", Parameter(torch.Tensor(1)))
        self.register_parameter("_c3", Parameter(torch.Tensor(1)))
        self.register_parameter("_c4", Parameter(torch.Tensor(1)))
        self.register_parameter("_a1", Parameter(torch.Tensor(1)))
        self.register_parameter("_a2", Parameter(torch.Tensor(1)))
        self.register_parameter("_a3", Parameter(torch.Tensor(1)))
        self.register_parameter("_a4", Parameter(torch.Tensor(1)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters to the default ZBL potential. """
        init.constant_(self._adiv, softplus_inverse(1 / (0.8854 * self.a0)))
        init.constant_(self._apow, softplus_inverse(0.23))
        init.constant_(self._c1, softplus_inverse(0.18180))
        init.constant_(self._c2, softplus_inverse(0.50990))
        init.constant_(self._c3, softplus_inverse(0.28020))
        init.constant_(self._c4, softplus_inverse(0.02817))
        init.constant_(self._a1, softplus_inverse(3.20000))
        init.constant_(self._a2, softplus_inverse(0.94230))
        init.constant_(self._a3, softplus_inverse(0.40280))
        init.constant_(self._a4, softplus_inverse(0.20160))

    def get_zbf_energy(
        self,
        Za: Tensor,
        Dij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        cutoff_values: Optional[Tensor]=None,
    ) -> Tensor:
        """
        Evaluate the short-range repulsive potential.
        P: Number of atom pairs.

        Arguments:
            N (int):
                Number of atoms.
            Zf (FloatTensor [N]):
                Nuclear charges of atoms (as floating point values).
            rij (FloatTensor [P]):
                Pairwise interatomic distances.
            cutoff_values (FloatTensor [P]):
                Values of a cutoff function for the distances rij.
            idx_i (LongTensor [P]):
                Index of atom i for all atomic pairs ij. Each pair must be
                specified as both ij and ji.
            idx_j (LongTensor [P]):
                Same as idx_i, but for atom j.

        Returns:
            e (FloatTensor [N]):
                Atomic contributions to the total repulsive energy.
        """
        if cutoff_values is None:
            cutoff_values = self.cutoff_fn(Dij, cutoff=self.cutoff_sr)
        # calculate ZBL parameters
        Zf = Za.type_as(self._a1)
        z = Zf ** F.softplus(self._apow)
        a = (z[idx_i] + z[idx_j]) * F.softplus(self._adiv)
        a1 = F.softplus(self._a1) * a
        a2 = F.softplus(self._a2) * a
        a3 = F.softplus(self._a3) * a
        a4 = F.softplus(self._a4) * a
        c1 = F.softplus(self._c1)
        c2 = F.softplus(self._c2)
        c3 = F.softplus(self._c3)
        c4 = F.softplus(self._c4)
        # normalize c coefficients (necessary to get asymptotically correct
        # behaviour for r -> 0)
        csum = c1 + c2 + c3 + c4
        c1 = c1 / csum
        c2 = c2 / csum
        c3 = c3 / csum
        c4 = c4 / csum
        # compute interactions
        zizj = Zf[idx_i] * Zf[idx_j]
        f = (
            c1 * torch.exp(-a1 * Dij)
            + c2 * torch.exp(-a2 * Dij)
            + c3 * torch.exp(-a3 * Dij)
            + c4 * torch.exp(-a4 * Dij)
        ) * cutoff_values
        return segment_sum(self.kehalf * f * zizj / Dij, idx_i)