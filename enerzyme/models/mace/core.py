from typing import Dict, List
import numpy as np
import torch
from torch import Tensor
from ..layers import BaseFFCore

try:
    from e3nn.o3 import Irreps
    from mace.modules.models import ScaleShiftMACE
    from mace.modules import interaction_classes, gate_dict
    from mace.tools import get_atomic_number_table_from_zs
    from mace.data.neighborhood import get_neighborhood
    from mace.tools.utils import atomic_numbers_to_indices
    from mace.data.atomic_data import to_one_hot
except ImportError:
    raise ImportError("External FF: MACE is not installed. Please install it with `pip install mace-torch`.")


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

    def build(self, built_layers) -> None:
        pass

    def get_output(self, Ra: Tensor, Za: Tensor, batch_seg: Tensor) -> Dict[str, Tensor]:
        mace_data = dict()
        indices = atomic_numbers_to_indices(Za.cpu(), z_table=self.z_table)
        one_hot = to_one_hot(torch.tensor(indices, dtype=torch.long, device=Za.device).unsqueeze(-1), num_classes=len(self.z_table))
        mace_data["batch"] = batch_seg
        mace_data["ptr"] = [0]
        mace_data["edge_index"], mace_data["shifts"], mace_data["unit_shifts"] = None, None, None
        for i in range(batch_seg[-1] + 1):
            mask = batch_seg == i
            mace_data["ptr"].append(mask.sum().item())
            edge_index, shifts, unit_shifts = get_neighborhood(
                positions=Ra[mask].detach().cpu().numpy(), cutoff=self.r_max, pbc=None, cell=None
            )
            edge_index = torch.tensor(edge_index, dtype=torch.long, device=Ra.device)
            shifts = torch.tensor(shifts, dtype=Ra.dtype, device=Ra.device)
            unit_shifts = torch.tensor(unit_shifts, dtype=Ra.dtype, device=Ra.device)
            if i == 0:
                mace_data["edge_index"] = edge_index
                mace_data["shifts"] = shifts
                mace_data["unit_shifts"] = unit_shifts
            else:
                mace_data["edge_index"] = torch.cat([mace_data["edge_index"], edge_index], dim=1)
                mace_data["shifts"] = torch.cat([mace_data["shifts"], shifts], dim=0)
                mace_data["unit_shifts"] = torch.cat([mace_data["unit_shifts"], unit_shifts], dim=0)
        mace_data["ptr"] = torch.tensor(mace_data["ptr"], dtype=torch.long, device=Ra.device)
        mace_data["positions"] = Ra
        mace_data["node_attrs"] = one_hot
        mace_data["batch"] = batch_seg
        mace_data["cell"] = None
        output = self.model(mace_data, compute_force=True, compute_virials=False, compute_stress=False, compute_displacement=False, compute_hessian=False, training=self.model.training)
        return {"E": output["energy"], "Fa": output["forces"]}
