import copy
from contextlib import nullcontext
from typing import Dict, Optional
import torch
from torch import Tensor
from torch import nn

from .runtime import (
    OMOL_TASK_NAME,
    ensure_pinned_fairchem,
    prepend_pinned_src_to_sys_path,
    resolve_uma_checkpoint,
)

prepend_pinned_src_to_sys_path()
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
    vacuum_pbc: bool = True,
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
        vacuum_pbc: If True (default, used by ``UMAWrapperQS``), build a cubic
            vacuum box with PBC. If False, match ``AtomicData.from_ase`` for
            isolated molecules (pbc all False, completed identity cell).

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

    if vacuum_pbc:
        # Cubic box enclosing atoms + vacuum padding. UMAWrapperQS uses this
        # so PBC neighbor search never sees periodic images of small molecules.
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
    else:
        # FAIRChemCalculator / AtomicData.from_ase for isolated molecules:
        # pbc all False; ASE get_cell(complete=True) on a missing cell is identity.
        cell = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(num_graphs, 3, 3)
            .contiguous()
        )
        pbc = torch.zeros(num_graphs, 3, dtype=torch.bool, device=device)

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


def _enable_energy_gradients(backbone, data: AtomicData) -> None:
    """Match ``MLIPPredictUnit._prepare_inference_gradients`` for autograd forces."""
    regress_config = getattr(backbone, "regress_config", None)
    if regress_config is None or regress_config.direct_forces:
        return
    if regress_config.forces or regress_config.stress:
        data["pos"].requires_grad_(True)
    if regress_config.stress:
        data["cell"].requires_grad_(True)


def _unwrap_task_tensor(output: dict, task_name: str, property_name: str) -> Tensor:
    raw = output[task_name]
    if isinstance(raw, dict):
        if property_name in raw:
            return raw[property_name]
        if task_name in raw:
            return raw[task_name]
    return raw


class UMAOmolWrapper(BaseFFCore):
    """Full UMA-s-1p2 OMol energy/force wrapper.

    Loads the complete ``HydraModel`` (backbone, DatasetSpecificMoEWrapper /
    MLP_EFS_Head, all checkpoint tasks, OMol normalizer and element
    references) via ``load_inference_model(..., use_ema=True)``. Each forward
    runs the backbone once and sends the same ``emb["node_embedding"]`` into
    the frozen OMol EFS head. Outputs are denormalized and have element
    references restored as in ``MLIPPredictUnit._process_outputs``.

    ``uma-s-1p2`` has ``direct_forces=False``, so forces are energy gradients.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        task_name: str = OMOL_TASK_NAME,
        frozen: bool = True,
        use_ema: bool = True,
        inference_settings: str = "batch",
        add_untrained_stress: bool = True,
        strict: bool = True,
    ):
        super().__init__(
            input_fields={"Ra", "Za", "batch_seg", "Q", "S"},
            output_fields={"E", "Fa"},
        )
        ensure_pinned_fairchem()
        from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
        from fairchem.core.units.mlip_unit.utils import (
            get_backbone_class_from_checkpoint,
            load_inference_model,
            tf32_context_manager,
        )

        ckpt_path = str(resolve_uma_checkpoint(checkpoint_path))
        settings = guess_inference_settings(inference_settings)
        if settings.tf32:
            raise ValueError(
                "UMAOmolWrapper requires eager FP32 (tf32=False). "
                f"Got inference_settings={inference_settings!r}."
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        backbone_cls = get_backbone_class_from_checkpoint(checkpoint)
        overrides = {"backbone": backbone_cls.build_inference_settings(settings)}
        self.uma, checkpoint = load_inference_model(
            ckpt_path,
            overrides=overrides,
            use_ema=use_ema,
            return_checkpoint=True,
            strict=strict,
            preloaded_checkpoint=checkpoint,
        )
        hydra_model = self.uma.module if hasattr(self.uma, "module") else self.uma
        hydra_model.setup_tasks(checkpoint.tasks_config)
        if add_untrained_stress and hasattr(hydra_model.backbone, "get_default_untrained_tasks"):
            extra_tasks = hydra_model.backbone.get_default_untrained_tasks(
                hydra_model.tasks, settings
            )
            if extra_tasks:
                hydra_model.add_tasks(extra_tasks)

        regress = hydra_model.backbone.regress_config
        if getattr(regress, "direct_forces", False):
            raise ValueError(
                "UMAOmolWrapper requires energy-conserving forces "
                "(direct_forces=False); uma-s-1p2 must not use direct forces."
            )

        self.hydra = hydra_model
        self.task_name = task_name
        self.inference_settings = settings
        self._tf32_context_manager = tf32_context_manager
        self._prepared = False
        self.checkpoint_path = ckpt_path
        self.dim_feature_out = hydra_model.backbone.sphere_channels

        energy_name = f"{task_name}_energy"
        forces_name = f"{task_name}_forces"
        if energy_name not in hydra_model.tasks:
            raise KeyError(
                f"Checkpoint is missing task {energy_name!r}. "
                f"Available: {sorted(hydra_model.tasks)}"
            )
        if forces_name not in hydra_model.tasks:
            from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task

            energy_task = hydra_model.tasks[energy_name]
            hydra_model.add_tasks(
                [
                    Task(
                        name=forces_name,
                        level="atom",
                        property="forces",
                        out_spec=OutputSpec(
                            dim=[3], dtype=settings.base_precision_dtype
                        ),
                        normalizer=energy_task.normalizer,
                        datasets=[task_name],
                        loss_fn=None,
                        element_references=None,
                        metrics=[],
                        train_on_free_atoms=True,
                        eval_on_free_atoms=True,
                        inference_only=True,
                    )
                ]
            )
        self._energy_task_name = energy_name
        self._forces_task_name = forces_name
        self._efs_head_names = self._discover_omol_efs_heads(task_name)

        # Keep OMol normalizers / element references on this module so .to(device)
        # moves them; they are not registered under HydraModel.
        self.task_normalizers = nn.ModuleDict()
        self.task_element_references = nn.ModuleDict()
        for name, task in hydra_model.tasks.items():
            if task_name not in getattr(task, "datasets", []):
                continue
            self.task_normalizers[name] = task.normalizer
            if task.element_references is not None:
                self.task_element_references[name] = task.element_references

        hydra_model.backbone.otf_graph = True
        hydra_model.eval()
        self.uma.eval()
        if frozen:
            for param in self.uma.parameters():
                param.requires_grad = False

    def _discover_omol_efs_heads(self, task_name: str) -> list[str]:
        names: list[str] = []
        for head_name, head in self.hydra.output_heads.items():
            dataset_names = getattr(head, "dataset_names", None)
            dataset_map = getattr(head, "dataset_name_to_exp", None)
            if dataset_names is not None and task_name in dataset_names:
                names.append(head_name)
            elif dataset_map is not None and task_name in dataset_map:
                names.append(head_name)
        if not names:
            names = list(self.hydra.output_heads.keys())
        return names

    def __str__(self) -> str:
        return """
##############################################################
# Wrapped UMA-s-1p2 OMol energy/forces (frozen HydraModel)   #
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

    def _prepare_if_needed(self, data: AtomicData) -> None:
        if self._prepared:
            return
        self.hydra.prepare_for_inference(data, self.inference_settings)
        self.uma.requires_grad_(False)
        self._prepared = True

    def _process_omol_outputs(self, data: AtomicData, output: dict) -> Dict[str, Tensor]:
        """Denormalize and restore element references (MLIPPredictUnit._process_outputs)."""
        processed: dict[str, Tensor] = {}
        for task_name in (self._energy_task_name, self._forces_task_name):
            task = self.hydra.tasks[task_name]
            tensor = _unwrap_task_tensor(output, task_name, task.property)
            pred = task.normalizer.denorm(tensor)
            if task.element_references is not None:
                pred = task.element_references.undo_refs(data, pred)
            processed[task.property] = pred
        energy = processed["energy"]
        if energy.ndim > 1:
            energy = energy.reshape(energy.shape[0])
        return {"E": energy, "Fa": processed["forces"]}

    def predict_from_atomic_data(self, data: AtomicData) -> Dict[str, Tensor]:
        """OMol energy/forces from a fairchem ``AtomicData`` batch (eager FP32)."""
        self.uma.eval()
        self.hydra.eval()
        param = next(self.uma.parameters())
        dtype = self.inference_settings.base_precision_dtype
        data = data.to(param.device).clone()
        for key, val in data:
            if torch.is_tensor(val) and val.is_floating_point():
                data[key] = val.to(dtype)
        self._prepare_if_needed(data)
        self.hydra.on_predict_check(data)
        backbone = self.hydra.backbone
        _enable_energy_gradients(backbone, data)
        inference_context = (
            torch.no_grad() if backbone.regress_config.direct_forces else nullcontext()
        )
        with inference_context, self._tf32_context_manager(False):
            emb = backbone(data)
            out: dict = {}
            for head_name in self._efs_head_names:
                head_out = self.hydra.output_heads[head_name](data, emb)
                if self.hydra.pass_through_head_outputs or any(
                    isinstance(key, str) and ("energy" in key or "forces" in key)
                    for key in head_out
                ):
                    out.update(head_out)
                else:
                    out[head_name] = head_out
            return self._process_omol_outputs(data, out)

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        batch_seg: Tensor,
        Q: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        data = _fields_to_atomic_data(
            Ra,
            Za,
            batch_seg,
            task_name=self.task_name,
            charge=Q,
            spin=S,
            vacuum_pbc=False,
        )
        return self.predict_from_atomic_data(data)

