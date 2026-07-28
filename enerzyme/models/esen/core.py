import copy
from typing import Dict
import torch
from torch import Tensor
import fairchem.core.models.base
import fairchem.core.models.uma.escn_md
import fairchem.core.models.uma.escn_moe
from fairchem.core.common.registry import registry as fairchem_registry
from fairchem.core.datasets.atomic_data import AtomicData
from .flow_umabackbone import build_flow_backbone
from .utils import load_state_dict, match_state_dict
from ..layers import BaseFFCore

# Default vacuum padding (Angstrom) for aperiodic molecules when creating cell
_DEFAULT_VACUUM = 50.0


def _fields_to_atomic_data(
    Ra: Tensor,
    Za: Tensor,
    batch_seg: Tensor,
    task_name: str = "omol",
    vacuum: float = _DEFAULT_VACUUM,
    device: torch.device | None = None,
    charge: Tensor | None = None,
    spin: Tensor | None = None,
) -> AtomicData:
    """Convert Enerzyme format (Ra, Za, batch_seg) to UMA AtomicData format.

    Graph is built on-the-fly by UMA backbone (otf_graph=True).

    Args:
        Ra: Atomic positions, shape (N, 3).
        Za: Atomic numbers, shape (N,).
        batch_seg: Batch index per atom, shape (N,). Graph i has atoms where batch_seg == i.
        task_name: UMA task name for dataset embedding (e.g. 'omol', 'omat', 'oc20').
        vacuum: Extra padding (Angstrom) for molecule cell when constructing PBC box.
        device: Device for output tensors. If None, uses Ra's device.
        charge: Total charge per graph, shape (num_graphs,) or scalar. Default 0.
        spin: Spin multiplicity minus 1 per graph (S = mult - 1), shape (num_graphs,) or scalar.
            UMA expects multiplicity, so we pass spin = S + 1. Default 0 (multiplicity 1) for non-omol, 1 for omol.

    Returns:
        AtomicData with empty edge_index; backbone builds graph internally via otf_graph.

    Flow-matching inputs (e.g. ``Q_flow_a``, ``S_flow_a``, ``flow_t``) are embedded in
    pre-core layers and summed by :class:`enerzyme.models.layers.GatherAtomEmbedding` into
    ``atom_embedding`` before :class:`UMAFlowWrapperQS`; they are not stored on AtomicData.
    """
    if device is None:
        device = Ra.device
    dtype = Ra.dtype
    Ra = Ra.to(device=device, dtype=dtype)
    Za = Za.to(device=device)
    batch_seg = batch_seg.to(device=device)

    num_graphs = int(batch_seg.max().item()) + 1
    n_atoms = Ra.shape[0]

    # Per-graph atom counts
    natoms = torch.tensor(
        [(batch_seg == i).sum().item() for i in range(num_graphs)],
        dtype=torch.long,
        device=device,
    )

    # Build cell for each graph: cubic box enclosing atoms + vacuum padding
    # UMA with PBC expects a cell per graph for neighbor search
    cells = []
    for i in range(num_graphs):
        mask = batch_seg == i
        pos_i = Ra[mask]
        min_pos = pos_i.min(dim=0).values
        max_pos = pos_i.max(dim=0).values
        span = (max_pos - min_pos).max().item() + vacuum
        cell = torch.eye(3, dtype=dtype, device=device) * span
        cells.append(cell)
    cell = torch.stack(cells)  # (num_graphs, 3, 3)
    pbc = torch.ones(num_graphs, 3, dtype=torch.bool, device=device)

    if charge is not None:
        charge = charge.to(device=device)
        if charge.dim() == 0 or charge.shape[0] == 1:
            charge = charge.expand(num_graphs).long()
        else:
            charge = charge.long()
    else:
        charge = torch.zeros(num_graphs, dtype=torch.long, device=device)

    if spin is not None:
        spin = spin.to(device=device)
        if spin.dim() == 0 or spin.shape[0] == 1:
            spin = spin.expand(num_graphs).long()
        else:
            spin = spin.long()
        spin = spin + 1  # S = mult - 1 -> UMA expects multiplicity
    else:
        spin = (
            torch.ones(num_graphs, dtype=torch.long, device=device)
            if task_name == "omol"
            else torch.zeros(num_graphs, dtype=torch.long, device=device)
        )
    fixed = torch.zeros(n_atoms, dtype=torch.long, device=device)
    tags = torch.zeros(n_atoms, dtype=torch.long, device=device)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    cell_offsets = torch.empty(0, 3, dtype=dtype, device=device)
    nedges = torch.zeros(num_graphs, dtype=torch.long, device=device)

    # sid must have length num_graphs for AtomicData.validate(); no real IDs in Enerzyme format
    sid = [""] * num_graphs

    return AtomicData(
        pos=Ra,
        atomic_numbers=Za.long(),
        cell=cell,
        pbc=pbc,
        natoms=natoms,
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        nedges=nedges,
        charge=charge,
        spin=spin,
        fixed=fixed,
        tags=tags,
        batch=batch_seg,
        sid=sid,
        dataset=[task_name] * num_graphs,
    )


def _extract_backbone_state_dict(checkpoint_state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Extract backbone weights from full model checkpoint and strip prefix."""
    backbone_state = {}
    for k, v in checkpoint_state_dict.items():
        if "n_averaged" in k:
            continue
        # Handle: "backbone.xxx", "module.backbone.xxx"
        if ".backbone." in k:
            subkey = k.split(".backbone.", 1)[-1]
        elif k.startswith("backbone."):
            subkey = k[len("backbone.") :]
        else:
            continue
        backbone_state[subkey] = v
    return backbone_state


class UMAWrapperQS(BaseFFCore):
    def __init__(self, checkpoint_path: str, shallow_ensemble_size: int = 1, frozen_backbone: bool = False):
        super().__init__(
            input_fields={"Ra", "Za", "batch_seg", "Q", "S"},
            output_fields={"atom_feature"},
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg = checkpoint.model_config
        backbone_cfg = copy.deepcopy(cfg["backbone"])
        backbone_cls = fairchem_registry.get_model_class(backbone_cfg.pop("model"))
        self.backbone = backbone_cls(**backbone_cfg)
        self.backbone.otf_graph = True

        # Prefer EMA weights for inference when available
        state_dict = getattr(checkpoint, "ema_state_dict", None) or checkpoint.model_state_dict
        backbone_state = _extract_backbone_state_dict(state_dict)
        if backbone_state:
            matched = match_state_dict(self.backbone.state_dict(), backbone_state)
            load_state_dict(self.backbone, matched, strict=False)
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Backbone node_embedding l=0 scalar has dimension sphere_channels
        self.dim_feature_out = self.backbone.sphere_channels

    def __str__(self) -> str:
        return """
##############################################################
# Wrapped UMA for charge/spin prediction (arXiv: 2506.23971) #
##############################################################   
"""

    def build(self, built_layers) -> None:
        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        Q: Tensor,
        S: Tensor,
        batch_seg: Tensor,
    ) -> Dict[str, Tensor]:
        input_data = _fields_to_atomic_data(
            Ra, Za, batch_seg, charge=Q, spin=S
        )

        emb = self.backbone(input_data)
        # Use scalar (l=0) component of node_embedding: shape (N, sphere_channels)
        return {"atom_feature": emb["node_embedding"][:, 0, :]}


class UMAFlowWrapperQS(BaseFFCore):
    """UMA backbone with early fusion of pre-core ``atom_embedding`` (see references/flow_matching.md §3.1).

    Place :class:`enerzyme.models.layers.ScalarDenseEmbedding` (e.g. ``initial_weight: zero``) and
    :class:`enerzyme.models.layers.GraphScalarBroadcastEmbedding` before ``GatherAtomEmbedding`` so flow scalars become
    ``*_embedding`` fields summed into ``atom_embedding``, then added to ``x_message[:, 0, :]``
    before ``edge_degree_embedding``. Omit ``atom_embedding`` or pass zeros for standard UMA behavior.
    """

    def __init__(self, checkpoint_path: str, shallow_ensemble_size: int = 1, frozen_backbone: bool = False):
        super().__init__(
            input_fields={"Ra", "Za", "batch_seg", "Q", "S", "atom_embedding"},
            output_fields={"atom_feature"},
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        cfg = checkpoint.model_config
        backbone_cfg = copy.deepcopy(cfg["backbone"])
        model_name = backbone_cfg.pop("model")
        self.backbone = build_flow_backbone(model_name, backbone_cfg)
        self.backbone.otf_graph = True

        state_dict = getattr(checkpoint, "ema_state_dict", None) or checkpoint.model_state_dict
        backbone_state = _extract_backbone_state_dict(state_dict)
        if backbone_state:
            matched = match_state_dict(self.backbone.state_dict(), backbone_state)
            load_state_dict(self.backbone, matched, strict=False)
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dim_feature_out = self.backbone.sphere_channels

    def __str__(self) -> str:
        return """
#####################################################################
# UMA flow wrapper: q/s/t fused before equivariant interaction stack #
#####################################################################
"""

    def build(self, built_layers) -> None:
        pre_core = True
        for layer in built_layers:
            if layer is self:
                pre_core = False
                continue
            if pre_core:
                self.pre_sequence.append(layer)
            else:
                self.post_sequence.append(layer)

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        Q: Tensor,
        S: Tensor,
        batch_seg: Tensor,
        atom_embedding: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        input_data = _fields_to_atomic_data(Ra, Za, batch_seg, charge=Q, spin=S)
        emb = self.backbone(input_data, atom_embedding=atom_embedding)
        return {"atom_feature": emb["node_embedding"][:, 0, :]}
