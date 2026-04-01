from typing import Dict, List, Optional, Literal, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from e3nn.util.jit import compile_mode
from e3nn.o3 import Irreps, SphericalHarmonics
from .interaction import INTERACTION_CLASSES, EquivariantProductBasisBlock, LinearReadoutBlock, NonLinearReadoutBlock
from ..layers import BaseFFCore, DistanceLayer, RangeSeparationLayer, BaseAtomEmbedding, BaseElectronEmbedding, BaseRBF, ChargeConservationLayer

DEFAULT_BUILD_PARAMS = {
    'r_max': 5.0,
    'atomic_numbers': [1, 6, 7, 8, 15, 16],
}
DEFAULT_LAYER_PARAMS = [{
    'name': 'Core',
    'params': {
        'num_bessel': 8,
        'num_polynomial_cutoff': 5,
        'interaction_cls': "RealAgnosticResidualInteractionBlock",
        'interaction_cls_first': "RealAgnosticResidualInteractionBlock",
        'max_ell': 3,
        'correlation': 3,
        'num_interactions': 2,
        'MLP_irreps': "16x0e",
        'radial_MLP': [64, 64, 64],
        'hidden_irreps': "128x0e + 128x1o",
        'gate': "silu",
        'avg_num_neighbors': 8.0,
    }
}]
GATE_FUNCTIONS = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}


class MACEWrapper(BaseFFCore):
    def __init__(self, 
        atomic_numbers: List[int], 
        r_max: float, 
        num_bessel: int, 
        num_polynomial_cutoff: int, 
        interaction_cls: str, 
        interaction_cls_first: str, 
        max_ell: int, 
        correlation: int, 
        num_interactions: int, 
        MLP_irreps: str, 
        radial_MLP: List[int], 
        hidden_irreps: str, 
        gate: str, 
        avg_num_neighbors: float
    ):
        try:
            from mace.modules.models import ScaleShiftMACE
            from mace.modules import interaction_classes, gate_dict
            from mace.tools import get_atomic_number_table_from_zs
        except ImportError:
            raise ImportError("External FF: MACE is not installed. Please install it with `pip install mace-torch`.")
        
        super().__init__(input_fields={"Ra", "Za", "batch_seg"}, output_fields={"E", "Fa"})
        self.z_table = get_atomic_number_table_from_zs(atomic_numbers)
        self.r_max = r_max
        self.model = ScaleShiftMACE(
            atomic_inter_scale=1.0, 
            atomic_inter_shift=0.0, 
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_classes[interaction_cls],
            interaction_cls_first=interaction_classes[interaction_cls_first],
            num_interactions=num_interactions,
            num_elements=len(self.z_table),
            hidden_irreps=Irreps(hidden_irreps),
            MLP_irreps=Irreps(MLP_irreps),
            atomic_energies=np.zeros(len(self.z_table)),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=self.z_table.zs,
            correlation=correlation,
            gate=gate_dict[gate],
            radial_MLP=radial_MLP
        )
    
    def __str__(self) -> str:
        return """
#################################################
# Wrapped MACE (NeurIPS 2022, arXiv:2206.07697) #
#################################################
"""

    def build(self, built_layers) -> None:
        pass

    def get_output(self, Ra: Tensor, Za: Tensor, batch_seg: Tensor) -> Dict[str, Tensor]:
        from mace.data.neighborhood import get_neighborhood
        from mace.tools.utils import atomic_numbers_to_indices
        from mace.data.atomic_data import to_one_hot

        mace_data = dict()
        indices = atomic_numbers_to_indices(Za.cpu(), z_table=self.z_table)
        one_hot = to_one_hot(torch.tensor(indices, dtype=torch.long, device=Za.device).unsqueeze(-1), num_classes=len(self.z_table))
        mace_data["batch"] = batch_seg
        mace_data["ptr"] = [0]
        mace_data["edge_index"], mace_data["shifts"], mace_data["unit_shifts"] = None, 0, 0
        count = 0
        for i in range(batch_seg[-1] + 1):
            mask = batch_seg == i
            edge_index, shifts, unit_shifts = get_neighborhood(
                positions=Ra[mask].detach().cpu().numpy(), cutoff=self.r_max, pbc=None, cell=None
            )
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=Ra.device)
            if i == 0:
                mace_data["edge_index"] = edge_index
            else:
                mace_data["edge_index"] = torch.cat([mace_data["edge_index"], edge_index + count], dim=1)
            N = mask.sum().item()
            count += N
            mace_data["ptr"].append(count)
        mace_data["ptr"] = torch.tensor(mace_data["ptr"], dtype=torch.long, device=Ra.device)
        mace_data["positions"] = Ra
        mace_data["node_attrs"] = one_hot
        mace_data["batch"] = batch_seg
        mace_data["cell"] = None
        output = self.model(mace_data, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False, compute_hessian=False, training=self.model.training)
        return {"E": output["energy"], "Fa": output["forces"]}


@compile_mode("script")
class MACECore(BaseFFCore):
    def __init__(self, 
        max_Za: int, max_ell: int, dim_embedding: int, num_rbf: int,
        additional_hidden_irreps: str,
        interaction_cls_first: Literal["RealAgnosticResidualInteractionBlock"],
        interaction_cls: Literal["RealAgnosticResidualInteractionBlock"],
        correlation: Union[int, List[int]],
        num_interactions: int, 
        avg_num_neighbors: float,
        MLP_irreps: str, 
        radial_MLP: List[int], 
        gate: str,
        shallow_ensemble_size: int=1,
        output_mode: Literal["direct", "feature"]="direct",
    ):
        self.output_mode: Literal["direct", "feature"] = output_mode
        output_fields = {"Ea", "Qa"} if output_mode == "direct" else {"atom_feature"}
        super().__init__(
            input_fields={"Za", "vij_sr", "idx_i_sr", "idx_j_sr", "rbf", "atom_embedding", "charge_embedding", "spin_embedding"},
            output_fields=output_fields,
        )
        self.max_Za = max_Za
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        sh_irreps = Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        node_attrs_irreps = Irreps([(max_Za + 1, (0, 1))])
        node_feats_irreps = Irreps([(dim_embedding, (0, 1))])
        edge_feats_irreps = Irreps([(num_rbf, (0, 1))])
        hidden_irreps = Irreps(f"{dim_embedding}x0e+" + additional_hidden_irreps)
        interaction_irreps = (sh_irreps * dim_embedding).sort()[0].simplify()
        inter_first = INTERACTION_CLASSES[interaction_cls_first](
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter_first])
        self.shallow_ensemble_size = shallow_ensemble_size

        use_sc_first = False
        if "Residual" in interaction_cls_first:
            use_sc_first = True

        node_feats_irreps_out = inter_first.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=max_Za + 1,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])
        if self.output_mode == "direct":
            self.readouts = torch.nn.ModuleList([LinearReadoutBlock(hidden_irreps, shallow_ensemble_size)])

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = INTERACTION_CLASSES[interaction_cls](
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=max_Za + 1,
                use_sc=True,
            )
            self.products.append(prod)
            if self.output_mode == "direct":
                if i == num_interactions - 2:
                    self.readouts.append(
                        NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, GATE_FUNCTIONS[gate], shallow_ensemble_size)
                    )
                else:
                    self.readouts.append(LinearReadoutBlock(hidden_irreps, shallow_ensemble_size))
        

    def __str__(self) -> str:
        return """
###################################################
# Augmented MACE (NeurIPS 2022, arXiv:2206.07697) #
###################################################
"""
    
    def build(self, built_layers) -> None:
        self.calculate_distance = DistanceLayer()
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

    def get_output(self, 
            Za: Tensor, vij_sr: Tensor,
            idx_i_sr: Tensor, idx_j_sr: Tensor, rbf: Tensor,
            atom_embedding: Tensor, charge_embedding: Optional[Tensor]=None, spin_embedding: Optional[Tensor]=None
        ) -> Dict[str, Tensor]:
        # prepare initial attributes and features
        if charge_embedding is None:
            charge_embedding = torch.zeros_like(atom_embedding)
        if spin_embedding is None:
            spin_embedding = torch.zeros_like(atom_embedding)
        node_feats = atom_embedding + charge_embedding + spin_embedding
        node_attrs = F.one_hot(Za, num_classes=self.max_Za + 1).to(node_feats.dtype)
        edge_attrs = self.spherical_harmonics(vij_sr)
        edge_feats = rbf
        if self.output_mode == "feature":
            for interaction, product in zip(self.interactions, self.products):
                node_feats, sc = interaction(
                    node_attrs=node_attrs,
                    node_feats=node_feats,
                    edge_attrs=edge_attrs,
                    edge_feats=edge_feats,
                    idx_i_sr=idx_i_sr,
                    idx_j_sr=idx_j_sr,
                )
                node_feats = product(
                    node_feats=node_feats, sc=sc, node_attrs=node_attrs
                )
            return {"atom_feature": node_feats}

        node_properties_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                idx_i_sr=idx_i_sr,
                idx_j_sr=idx_j_sr,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            if self.shallow_ensemble_size > 1:
                node_properties_list.append(readout(node_feats).reshape(-1, 2, self.shallow_ensemble_size))
            else:
                node_properties_list.append(readout(node_feats))
        node_properties = torch.sum(
            torch.stack(node_properties_list, dim=0), dim=0
        )
        return {"Ea": node_properties[:, 0], "Qa": node_properties[:, 1]}