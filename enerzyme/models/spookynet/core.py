import math
from typing import Dict, List, Optional, Tuple, Literal
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear
import torch.nn.functional as F
from .interaction import InteractionModule
from ..layers import DistanceLayer, RangeSeparationLayer, BaseFFCore, BaseAtomEmbedding, BaseElectronEmbedding, BaseRBF, ChargeConservationLayer
from ..activation import ACTIVATION_KEY_TYPE


DEFAULT_BUILD_PARAMS = {'dim_embedding': 64,
 'num_rbf': 16,
 'max_Za': 86,
 'cutoff_sr': 5.291772105638412,
 'Hartree_in_E': 1,
 'Bohr_in_R': 0.5291772108,
 'activation_fn': 'swish'}
DEFAULT_LAYER_PARAMS = [{'name': 'RangeSeparation', 'params': {'cutoff_fn': 'bump'}},
 {'name': 'ExponentialBernsteinRBF',
  'params': {'no_basis_at_infinity': False,
   'init_alpha': 0.944863062918464,
   'exp_weighting': False,
   'learnable_shape': True}},
 {'name': 'NuclearEmbedding',
  'params': {'zero_init': True, 'use_electron_config': True}},
 {'name': 'ElectronicEmbedding',
  'params': {'num_residual': 1, 'attribute': 'charge'}},
 {'name': 'ElectronicEmbedding',
  'params': {'num_residual': 1, 'attribute': 'spin'}},
 {'name': 'Core',
  'params': {'num_modules': 3,
   'num_residual_pre': 1,
   'num_residual_local_x': 1,
   'num_residual_local_s': 1,
   'num_residual_local_p': 1,
   'num_residual_local_d': 1,
   'num_residual_local': 1,
   'num_residual_nonlocal_q': 1,
   'num_residual_nonlocal_k': 1,
   'num_residual_nonlocal_v': 1,
   'num_residual_post': 1,
   'num_residual_output': 1,
   'use_irreps': True,
   'dropout_rate': 0.0}},
 {'name': 'AtomicAffine',
  'params': {'shifts': {'Ea': {'values': 0, 'learnable': True},
    'Qa': {'values': 0, 'learnable': True}},
   'scales': {'Ea': {'values': 1, 'learnable': True},
    'Qa': {'values': 1, 'learnable': True}}}},
 {'name': 'ChargeConservation'},
 {'name': 'AtomicCharge2Dipole'},
 {'name': 'ZBLRepulsionEnergy'},
 {'name': 'ElectrostaticEnergy', 'params': {'flavor': 'SpookyNet'}},
 {'name': 'GrimmeD4Energy', 'params': {'learnable': True}},
 {'name': 'EnergyReduce'},
 {'name': 'Force'}]


class SpookyNetCore(BaseFFCore):
    def __str__(self) -> str:
        return """
###############################################
# SpookyNet (Nat. Commun., 2021, 12(1): 7273) #
###############################################
"""

    def __init__(
        self, dim_embedding: int, num_rbf: int, num_modules: int, num_residual_pre: int,
        num_residual_local_x: int, num_residual_local_s: int, num_residual_local_p: int, 
        num_residual_local_d: int, num_residual_local: int,
        num_residual_nonlocal_q: int, num_residual_nonlocal_k: int, num_residual_nonlocal_v: int,
        num_residual_post: int, num_residual_output: int, activation_fn: ACTIVATION_KEY_TYPE, use_irreps: bool, dropout_rate: float=0.0
    ) -> None:
        super().__init__()
        self.interaction = ModuleList(
            [
                InteractionModule(
                    dim_embedding=dim_embedding,
                    num_rbf=num_rbf,
                    num_residual_pre=num_residual_pre,
                    num_residual_local_x=num_residual_local_x,
                    num_residual_local_s=num_residual_local_s,
                    num_residual_local_p=num_residual_local_p,
                    num_residual_local_d=num_residual_local_d,
                    num_residual_local=num_residual_local,
                    num_residual_nonlocal_q=num_residual_nonlocal_q,
                    num_residual_nonlocal_k=num_residual_nonlocal_k,
                    num_residual_nonlocal_v=num_residual_nonlocal_v,
                    num_residual_post=num_residual_post,
                    num_residual_output=num_residual_output,
                    activation_fn=activation_fn,
                )   
                for _ in range(num_modules)
            ]
        )
        self.output = Linear(dim_embedding, 2, bias=False)
        self.use_irreps = use_irreps
        self._sqrt2 = math.sqrt(2.0)
        self._sqrt3 = math.sqrt(3.0)
        self._sqrt3half = 0.5 * self._sqrt3
        self.module_keep_prob = 1 - dropout_rate
        self.calculate_distance: DistanceLayer = None
        self.range_separation: RangeSeparationLayer = None

    def build(self, built_layers: List[Module]) -> None:
        # build necessary fixed pre-core layers
        self.calculate_distance = DistanceLayer()
        self.calculate_distance.with_vector_on("vij_lr")
        self.calculate_distance.reset_field_name(Dij="Dij_lr")
        self.pre_sequence.append(self.calculate_distance)

        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                # reset pre-core layers
                if isinstance(layer, RangeSeparationLayer):
                    self.range_separation = layer
                    self.range_separation.reset_field_name(idx_i_lr="idx_i", idx_j_lr="idx_j")
                elif isinstance(layer, BaseAtomEmbedding):
                    self.atom_embedding = layer
                elif isinstance(layer, BaseElectronEmbedding):
                    if layer.attribute == "charge":
                        self.charge_embedding = layer
                    elif layer.attribute == "spin":
                        self.spin_embedding = layer
                elif isinstance(layer, BaseRBF):
                    self.radial_basis_function = layer
                # build pre-core sequence
                self.pre_sequence.append(layer)
            else: 
                # build post-core sequence
                if isinstance(layer, ChargeConservationLayer):
                    self.charge_conservation = layer
                self.post_sequence.append(layer)

    def _atomic_properties_static(self, Dij_sr: Tensor, vij_sr: Tensor, batch_seg: Optional[Tensor]=None) -> Tuple[Tensor, Tensor, Tensor, int]:
        pij = vij_sr / Dij_sr.unsqueeze(-1)
        if self.use_irreps:  # irreducible representation
            try:
                from e3nn.o3 import spherical_harmonics as sh
                # strictly reproduction
                dij = sh(2, pij[:, [1,2,0]], normalize=True, normalization="norm")[:, [0,3,1,2,4]] 
            except ImportError:
                dij = torch.stack(
                    [
                        self._sqrt3 * pij[:, 0] * pij[:, 1],  # xy
                        self._sqrt3 * pij[:, 0] * pij[:, 2],  # xz
                        self._sqrt3 * pij[:, 1] * pij[:, 2],  # yz
                        0.5 * (3 * pij[:, 2] * pij[:, 2] - 1.0),  # z2
                        self._sqrt3half
                        * (pij[:, 0] * pij[:, 0] - pij[:, 1] * pij[:, 1]),  # x2-y2
                    ],
                    dim=-1,
                )
        else:  # reducible Cartesian functions
            dij = torch.stack(
                [
                    pij[:, 0] * pij[:, 0],  # x2
                    pij[:, 1] * pij[:, 1],  # y2
                    pij[:, 2] * pij[:, 2],  # z2
                    self._sqrt2 * pij[:, 0] * pij[:, 1],  # x*y
                    self._sqrt2 * pij[:, 0] * pij[:, 2],  # x*z
                    self._sqrt2 * pij[:, 1] * pij[:, 2],  # y*z
                ],
                dim=-1,
            )
        if batch_seg is None:
            num_batch = 1
        else:
            num_batch = batch_seg[-1]
        if num_batch > 1:
            one_hot = F.one_hot(batch_seg).to(
                dtype=Dij_sr.dtype, device=Dij_sr.device
            )
            mask = one_hot @ one_hot.transpose(-1, -2)
        else:
            mask = None
        return pij, dij, mask, num_batch
    
    def _atomic_properties_dynamic(
        self, atom_embedding: Tensor, charge_embedding: Tensor, spin_embedding: Tensor, num_batch: int,
        rbf: Tensor, pij: Tensor, dij: Tensor, idx_i_sr: Tensor, idx_j_sr: Tensor, mask: Tensor, batch_seg: Optional[Tensor]=None):
        x = atom_embedding + charge_embedding + spin_embedding
        dropout_mask = torch.ones((num_batch, 1), dtype=x.dtype, device=x.device)
        f = x.new_zeros(x.size())
        for module in self.interaction:
            x, y = module(
                x, rbf, pij, dij, idx_i_sr, idx_j_sr, num_batch, batch_seg, mask
            )
            # apply dropout mask
            if self.training and self.module_keep_prob < 1.0:
                y = y * dropout_mask[batch_seg]
                dropout_mask = dropout_mask * torch.bernoulli(self.keep_prob * torch.ones_like(dropout_mask))
            f = f + y
        out = self.output(f)
        ea = out.narrow(-1, 0, 1).squeeze(-1)  # atomic energy
        qa = out.narrow(-1, 1, 1).squeeze(-1)  # partial charge
        return ea, qa

    def get_output(
        self, Dij_sr: Tensor, vij_sr: Tensor, idx_i_sr: Tensor, idx_j_sr: Tensor, 
        rbf: Tensor, atom_embedding: Tensor, charge_embedding: Tensor, spin_embedding: Tensor, batch_seg: Optional[Tensor]=None
    ) -> Dict[Literal["Ea", "Qa"], Tensor]:
        pij, dij, mask, num_batch = self._atomic_properties_static(Dij_sr, vij_sr, batch_seg)
        ea, qa = self._atomic_properties_dynamic(
            atom_embedding, charge_embedding, spin_embedding, num_batch, rbf, pij, dij, idx_i_sr, idx_j_sr, mask, batch_seg
        )
        return {"Ea": ea, "Qa": qa}
