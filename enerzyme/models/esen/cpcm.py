"""DC-UMA C-PCM residual adapters on frozen UMA-s-1p2 OMol.

The vacuum HydraModel stays frozen. Residual energy is physical eV and
residual forces are ``ΔF = -∂ΔE/∂R`` (no per-atom energy scaling before the
gradient). ``ε == 1`` is gated to exact zeros.

Adapter modes:

* ``stage_a``: node ``l=0`` embedding plus dielectric encoding (default).
* ``x_only``: node embedding only (geometry/chemistry, no ε).
* ``zero``: exact zero residual (vacuum UMA only).
* ``stage_b``: Stage A local MLP plus charge-conserving full-pair GB
  (Stillinger ``f_ij``) on coordinates. Does not reuse UMA ``graph_dict``.
  ``ΔE = gate(ε)·(E_local + g(ε)·(E_self + E_pair))`` with
  ``g = 1 - 1/ε``. Logs the three contributions that enter ``ΔE``.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from ase.units import Bohr, Hartree
from torch import Tensor, nn
from torch.nn import functional as F

from ...data.datatype import MIN_DIELECTRIC, validate_dielectric_tensor
from ...models.layers.electrostatics import ChargeConservationLayer
from .core import (
    UMAOmolWrapper,
    _enable_energy_gradients,
    _fields_to_atomic_data,
)

ADAPTER_MODES = ("stage_a", "x_only", "zero", "stage_b")
MIN_BORN_RADIUS = 0.5
# Coulomb constant in eV·Å/e² so GB energies are eV when r, a are in Å.
COULOMB_KE_EV_A = float(Hartree * Bohr)
STAGE_B_LOG_KEYS = ("E_local", "E_self", "E_pair")


def encode_dielectric(eps: Tensor) -> Tensor:
    """Return ``[g(ε), log ε]`` with ``g = 1 - 1/ε``. Shape ``[..., 2]``."""
    eps = eps.to(dtype=torch.float32 if eps.dtype in (torch.float16, torch.bfloat16) else eps.dtype)
    g = 1.0 - 1.0 / eps
    return torch.stack((g, torch.log(eps)), dim=-1)


def energy_gradient_forces(
    energy: Tensor,
    pos: Tensor,
    *,
    create_graph: bool,
    retain_graph: bool = False,
) -> Tensor:
    """``ΔF = -∂E/∂R``. ``allow_unused`` covers an all-``ε=1`` batch."""
    (grad,) = torch.autograd.grad(
        energy.sum(),
        pos,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if grad is None:
        return torch.zeros_like(pos)
    return torch.neg(grad)


def stillinger_gb_energy(
    pos: Tensor,
    qa: Tensor,
    radii: Tensor,
    batch_seg: Tensor,
    *,
    ke: float = COULOMB_KE_EV_A,
) -> tuple[Tensor, Tensor]:
    """Per-graph Born self-energy and Stillinger pair GB, without ``g(ε)``.

    Positions and radii are Å; charges are elementary; energies are eV:

    ``E_GB = -½ ke Σ_{i,j} q_i q_j / f_ij``

    with ``f_ii = a_i`` and
    ``f_ij = √(r_ij² + a_i a_j exp(-r_ij² / (4 a_i a_j)))`` for ``i ≠ j``.
    Pairs from different graphs are masked (full pairwise on coordinates;
    UMA ``graph_dict`` is not used). ``N ≤ 60`` per molecule.
    """
    n_atoms = int(pos.shape[0])
    n_graphs = int(batch_seg.max().item()) + 1 if n_atoms else 0
    e_self = pos.new_zeros(n_graphs)
    e_pair = pos.new_zeros(n_graphs)
    if n_atoms == 0:
        return e_self, e_pair
    qa = qa.reshape(-1).to(dtype=pos.dtype)
    radii = radii.reshape(-1).to(dtype=pos.dtype).clamp_min(MIN_BORN_RADIUS)
    r2 = (pos.unsqueeze(1) - pos.unsqueeze(0)).pow(2).sum(dim=-1)
    same = batch_seg.unsqueeze(0) == batch_seg.unsqueeze(1)
    aiaj = (radii.unsqueeze(1) * radii.unsqueeze(0)).clamp_min(1.0e-12)
    f = torch.sqrt(r2 + aiaj * torch.exp(-r2 / (4.0 * aiaj)))
    qq_over_f = (qa.unsqueeze(1) * qa.unsqueeze(0)) / f.clamp_min(1.0e-12)
    qq_over_f = qq_over_f.masked_fill(~same, 0.0)
    diag = torch.diagonal(qq_over_f)
    e_self.index_add_(0, batch_seg, -0.5 * ke * diag)
    off = qq_over_f - torch.diag_embed(diag)
    e_pair.index_add_(0, batch_seg, -0.5 * ke * off.sum(dim=1))
    return e_self, e_pair


def load_stage_a_adapter_weights(adapter: nn.Module, path: str | Path) -> dict[str, Any]:
    """Load Stage A ``norm`` / ``mlp`` into a Stage B adapter (``strict=False``)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("adapter_state_dict", ckpt)
    if not isinstance(state, dict):
        raise TypeError(f"adapter checkpoint {path} has no state dict")
    incompatible = adapter.load_state_dict(state, strict=False)
    return {
        "path": str(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "loaded_keys": [k for k in state if k in adapter.state_dict()],
    }


class StageASolvationAdapter(nn.Module):
    """LayerNorm → MLP(256, 128, 1) on node ``l=0`` features, optionally plus ε.

    Atomic energies are summed (not averaged) so residual forces stay
    consistent with the physical-eV graph energy. ``x_only=True`` drops the
    dielectric channels so the head can only memorize geometry/chemistry.
    """

    def __init__(
        self,
        sphere_channels: int = 128,
        hidden: Sequence[int] = (256, 128),
        x_only: bool = False,
    ) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        self.x_only = bool(x_only)
        self.sphere_channels = int(sphere_channels)
        in_dim = int(sphere_channels) if self.x_only else int(sphere_channels) + 2
        self.norm = nn.LayerNorm(in_dim)
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers.append(nn.Linear(dim, width))
            layers.append(nn.SiLU())
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)

    def graph_energy(self, node_l0: Tensor, batch_seg: Tensor, dielectric: Tensor) -> Tensor:
        """Ungated graph energy from the local MLP (Stage A / Stage B ``E_local``)."""
        dielectric = validate_dielectric_tensor(dielectric).to(
            device=node_l0.device, dtype=node_l0.dtype
        )
        n_graphs = int(dielectric.shape[0])
        if self.x_only:
            x = node_l0
        else:
            enc = encode_dielectric(dielectric)
            x = torch.cat((node_l0, enc[batch_seg]), dim=-1)
        ea = self.mlp(self.norm(x)).squeeze(-1)
        dE = node_l0.new_zeros(n_graphs)
        dE.index_add_(0, batch_seg, ea)
        return dE

    def forward(self, node_l0: Tensor, batch_seg: Tensor, dielectric: Tensor) -> Tensor:
        dielectric = validate_dielectric_tensor(dielectric).to(
            device=node_l0.device, dtype=node_l0.dtype
        )
        dE = self.graph_energy(node_l0, batch_seg, dielectric)
        gate = (dielectric > MIN_DIELECTRIC).to(dtype=dE.dtype)
        return dE * gate


class StageBSolvationAdapter(nn.Module):
    """Stage A local MLP plus charge-conserving full-pair generalized Born.

    ``norm`` / ``mlp`` keep Stage A parameter names so a Stage A ``adapter.pt``
    warm-starts with ``strict=False``. Charge and radius heads read the same
    ``l=0`` embedding. Charge conservation is
    ``q_i = q̃_i + (Q - Σ q̃_j) / N``. Radii are ``a_i = 0.5 + softplus(ã_i)``.

    The GB Coulomb kernel is evaluated on coordinates (not UMA edges). The
    local MLP is kept as the Stage A residual; GB is added with the Born
    prefactor ``g(ε)``:

    ``ΔE = gate(ε) · (E_local + g(ε) · (E_self + E_pair))``.
    """

    def __init__(
        self,
        sphere_channels: int = 128,
        hidden: Sequence[int] = (256, 128),
        ke: float = COULOMB_KE_EV_A,
        min_radius: float = MIN_BORN_RADIUS,
    ) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden)
        self.sphere_channels = int(sphere_channels)
        self.x_only = False
        self.ke = float(ke)
        self.min_radius = float(min_radius)
        in_dim = int(sphere_channels) + 2
        self.norm = nn.LayerNorm(in_dim)
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers.append(nn.Linear(dim, width))
            layers.append(nn.SiLU())
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.charge_head = nn.Linear(int(sphere_channels), 1)
        self.radius_head = nn.Linear(int(sphere_channels), 1)
        self.charge_conservation = ChargeConservationLayer()
        nn.init.zeros_(self.charge_head.weight)
        nn.init.zeros_(self.charge_head.bias)
        nn.init.zeros_(self.radius_head.weight)
        nn.init.zeros_(self.radius_head.bias)

    def graph_energy(self, node_l0: Tensor, batch_seg: Tensor, dielectric: Tensor) -> Tensor:
        dielectric = validate_dielectric_tensor(dielectric).to(
            device=node_l0.device, dtype=node_l0.dtype
        )
        n_graphs = int(dielectric.shape[0])
        enc = encode_dielectric(dielectric)
        x = torch.cat((node_l0, enc[batch_seg]), dim=-1)
        ea = self.mlp(self.norm(x)).squeeze(-1)
        dE = node_l0.new_zeros(n_graphs)
        dE.index_add_(0, batch_seg, ea)
        return dE

    def conserved_charges(
        self,
        node_l0: Tensor,
        batch_seg: Tensor,
        za: Tensor,
        total_charge: Tensor,
    ) -> Tensor:
        q_tilde = self.charge_head(node_l0).squeeze(-1)
        q_tot = total_charge.to(device=q_tilde.device, dtype=q_tilde.dtype)
        return self.charge_conservation.get_output(
            Za=za,
            Qa=q_tilde,
            Q=q_tot,
            batch_seg=batch_seg,
        )["Qa"]

    def born_radii(self, node_l0: Tensor) -> Tensor:
        return self.min_radius + F.softplus(self.radius_head(node_l0).squeeze(-1))

    def forward(
        self,
        node_l0: Tensor,
        batch_seg: Tensor,
        dielectric: Tensor,
        pos: Tensor,
        za: Tensor,
        total_charge: Tensor,
    ) -> dict[str, Tensor]:
        dielectric = validate_dielectric_tensor(dielectric).to(
            device=node_l0.device, dtype=node_l0.dtype
        )
        e_local = self.graph_energy(node_l0, batch_seg, dielectric)
        qa = self.conserved_charges(node_l0, batch_seg, za, total_charge)
        radii = self.born_radii(node_l0)
        e_self_raw, e_pair_raw = stillinger_gb_energy(
            pos, qa, radii, batch_seg, ke=self.ke
        )
        g = encode_dielectric(dielectric)[..., 0]
        gate = (dielectric > MIN_DIELECTRIC).to(dtype=e_local.dtype)
        e_local_g = e_local * gate
        e_self = e_self_raw * g * gate
        e_pair = e_pair_raw * g * gate
        dE = e_local_g + e_self + e_pair
        return {
            "dE": dE,
            "E_local": e_local_g,
            "E_self": e_self,
            "E_pair": e_pair,
            "Qa": qa,
        }


class DCUMACPCMWrapper(UMAOmolWrapper):
    """Frozen UMA OMol vacuum PES plus a trainable C-PCM residual adapter.

    Training (``self.training``): one backbone forward, skip the frozen vacuum
    head, return residual ``E=dE`` and ``Fa=dFa``.

    Eval / inference: the same backbone embedding is sent to the vacuum EFS
    head and the adapter; totals are ``E = E_vac + ΔE`` and ``Fa = F_vac + ΔF``.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        task_name: str = "omol",
        frozen: bool = True,
        use_ema: bool = True,
        inference_settings: str = "batch",
        add_untrained_stress: bool = True,
        strict: bool = True,
        adapter_hidden: Sequence[int] | None = None,
        sphere_channels: int | None = None,
        adapter_mode: str = "stage_a",
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            task_name=task_name,
            frozen=frozen,
            use_ema=use_ema,
            inference_settings=inference_settings,
            add_untrained_stress=add_untrained_stress,
            strict=strict,
        )
        mode = str(adapter_mode or "stage_a").strip().lower().replace("-", "_")
        if mode not in ADAPTER_MODES:
            raise ValueError(f"adapter_mode must be one of {ADAPTER_MODES}, got {adapter_mode!r}")
        self.adapter_mode = mode
        self._input_fields = {"Ra", "Za", "batch_seg", "Q", "S", "dielectric"}
        self._output_fields = {"E", "Fa", "dE", "dFa"}
        if mode == "stage_b":
            self._output_fields = self._output_fields | {"E_local", "E_self", "E_pair", "Qa"}
        self._relevant_fields = self._input_fields | self._output_fields
        self._name_mapping = {name: name for name in self._relevant_fields}
        self.adapted_relevant_fields = False
        channels = int(sphere_channels or self.dim_feature_out)
        hidden = (256, 128) if adapter_hidden is None else tuple(adapter_hidden)
        if mode == "zero":
            self.adapter = nn.Module()
        elif mode == "stage_b":
            self.adapter = StageBSolvationAdapter(
                sphere_channels=channels,
                hidden=hidden,
            )
        else:
            self.adapter = StageASolvationAdapter(
                sphere_channels=channels,
                hidden=hidden,
                x_only=mode == "x_only",
            )

    def adapter_parameters(self):
        return [param for param in self.adapter.parameters() if param.requires_grad]

    def __str__(self) -> str:
        return f"""
#####################################################################
# DC-UMA C-PCM frozen UMA-s-1p2 OMol + adapter_mode={self.adapter_mode}
#####################################################################
"""

    def train(self, mode: bool = True):
        super().train(mode)
        self.uma.eval()
        self.hydra.eval()
        self.adapter.train(mode)
        return self

    def _prepare_atomic_data(self, data):
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
        _enable_energy_gradients(self.hydra.backbone, data)
        return data

    def _vacuum_from_embedding(self, data, emb: dict) -> Dict[str, Tensor]:
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

    def _adapter_residual(
        self, data, node_l0: Tensor, dielectric: Tensor
    ) -> dict[str, Tensor]:
        if self.adapter_mode == "stage_b":
            return self.adapter(
                node_l0,
                data.batch,
                dielectric,
                data.pos,
                data.atomic_numbers,
                data.charge,
            )
        dE = self.adapter(node_l0, data.batch, dielectric)
        return {"dE": dE}

    def predict_cpcm_from_atomic_data(
        self, data, dielectric: Tensor
    ) -> Dict[str, Tensor]:
        dielectric = validate_dielectric_tensor(dielectric)
        data = self._prepare_atomic_data(data)
        dielectric = dielectric.to(device=data.pos.device, dtype=data.pos.dtype)
        n_graphs = int(data.natoms.shape[0])
        if dielectric.numel() == 1 and n_graphs > 1:
            dielectric = dielectric.expand(n_graphs)
        if dielectric.numel() != n_graphs:
            raise ValueError(
                f"dielectric length {dielectric.numel()} does not match "
                f"number of graphs {n_graphs}"
            )
        backbone = self.hydra.backbone
        inference_context = (
            torch.no_grad() if backbone.regress_config.direct_forces else nullcontext()
        )
        with inference_context, self._tf32_context_manager(False):
            if self.adapter_mode == "zero":
                n_atoms = int(data.pos.shape[0])
                dE = data.pos.new_zeros(n_graphs)
                dFa = data.pos.new_zeros(n_atoms, 3)
                need_vacuum = not self.training
                if need_vacuum:
                    emb = backbone(data)
                    vacuum = self._vacuum_from_embedding(data, emb)
                    return {
                        "E": vacuum["E"] + dE,
                        "Fa": vacuum["Fa"] + dFa,
                        "dE": dE,
                        "dFa": dFa,
                    }
                return {"E": dE, "Fa": dFa, "dE": dE, "dFa": dFa}
            emb = backbone(data)
            node_l0 = emb["node_embedding"][:, 0, :]
            adapted = self._adapter_residual(data, node_l0, dielectric)
            dE = adapted["dE"]
            extras = {k: adapted[k] for k in adapted if k != "dE"}
            need_vacuum = not self.training
            dFa = energy_gradient_forces(
                dE,
                data.pos,
                create_graph=self.training,
                # Keep ΔE for residual loss (train) and the vacuum head (eval).
                retain_graph=True,
            )
            if need_vacuum:
                vacuum = self._vacuum_from_embedding(data, emb)
                return {
                    "E": vacuum["E"] + dE,
                    "Fa": vacuum["Fa"] + dFa,
                    "dE": dE,
                    "dFa": dFa,
                    **extras,
                }
            return {"E": dE, "Fa": dFa, "dE": dE, "dFa": dFa, **extras}

    def get_output(
        self,
        Ra: Tensor,
        Za: Tensor,
        batch_seg: Tensor,
        dielectric: Optional[Tensor] = None,
        Q: Optional[Tensor] = None,
        S: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        dielectric = validate_dielectric_tensor(dielectric)
        data = _fields_to_atomic_data(
            Ra,
            Za,
            batch_seg,
            task_name=self.task_name,
            charge=Q,
            spin=S,
            vacuum_pbc=False,
        )
        return self.predict_cpcm_from_atomic_data(data, dielectric)
