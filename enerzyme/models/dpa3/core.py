"""DPA3 Core: repflow message passing on a line graph of order 2.

Zhang, Peng, Cai et al., arXiv:2506.01686. The Core maps an atom embedding
and the short-range graph to an invariant atom latent. Species embeddings,
the elemental energy bias, dataset codes, and the energy head stay outside.

The released DeePMD-kit descriptor is this order-2 stack (atoms, bonds,
angles). Order-3 dihedrals appear only in one scaling-law experiment and
are not part of this Core. DeePMD-kit is LGPL-3.0 and is not imported;
its checkpoints do not load.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..mp.nn.activation import resolve_scalar_activation
from ..mp.basis.envelope import exponential_switch, polynomial_transition
from ..layers import BaseAtomEmbedding, BaseFFCore, RangeSeparationLayer
from .geometry import AnglePairs, angle_pairs
from .interaction import RepFlowBlock


PUBLISHED_CORE_PARAMS = {
    "n_dim": 128,
    "e_dim": 64,
    "a_dim": 32,
    "nlayers": 6,
    "e_rcut": 6.0,
    "e_rcut_smth": 5.0,
    "e_sel": 120,
    "a_rcut": 4.0,
    "a_rcut_smth": 3.5,
    "a_sel": 20,
    "axis_neuron": 4,
    "update_style": "res_residual",
    "update_residual": 0.1,
    "update_residual_init": "const",
    "fix_stat_std": 0.3,
    "smooth_edge_update": True,
    "activation": "silu",
}


DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 32,
    "max_Za": 94,
    "cutoff_sr": 6.0,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "RandomAtomEmbedding"},
    {
        "name": "Core",
        "params": {
            "e_dim": 16,
            "a_dim": 16,
            "nlayers": 2,
            "e_sel": 16,
            "a_sel": 8,
            "axis_neuron": 4,
        },
    },
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "dense",
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]


def _positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


class DPA3Core(BaseFFCore):
    """Invariant repflow stack. ``atom_feature`` is ``(N, n_dim)``."""

    def __init__(
        self,
        dim_embedding: int,
        n_dim: Optional[int] = None,
        e_dim: int = 32,
        a_dim: int = 32,
        nlayers: int = 2,
        e_rcut: float = 6.0,
        e_rcut_smth: float = 5.0,
        e_sel: int = 32,
        a_rcut: float = 4.0,
        a_rcut_smth: float = 3.5,
        a_sel: int = 16,
        axis_neuron: int = 4,
        n_multi_edge_message: int = 1,
        update_angle: bool = True,
        update_style: str = "res_residual",
        update_residual: float = 0.1,
        update_residual_init: str = "const",
        a_compress_rate: int = 0,
        a_compress_e_rate: int = 1,
        a_compress_use_split: bool = False,
        sequential_update: bool = False,
        smooth_edge_update: bool = True,
        edge_init_use_dist: bool = False,
        switch: str = "polynomial",
        switch_decay: float = 20.0,
        dynamic_normalize: bool = False,
        sel_reduce_factor: float = 10.0,
        fix_stat_std: float = 0.3,
        env_protection: float = 0.0,
        activation: str = "silu",
    ) -> None:
        n_dim = dim_embedding if n_dim is None else _positive("n_dim", n_dim)
        if int(n_dim) != int(dim_embedding):
            raise ValueError(
                f"n_dim ({n_dim}) must equal dim_embedding ({dim_embedding}); "
                "the atom embedding is the initial node state"
            )
        e_dim = _positive("e_dim", e_dim)
        a_dim = _positive("a_dim", a_dim)
        nlayers = _positive("nlayers", nlayers)
        e_sel = _positive("e_sel", e_sel)
        a_sel = _positive("a_sel", a_sel)
        axis_neuron = _positive("axis_neuron", axis_neuron)
        n_multi_edge_message = _positive("n_multi_edge_message", n_multi_edge_message)
        if axis_neuron > n_dim or axis_neuron > e_dim:
            raise ValueError(
                f"axis_neuron ({axis_neuron}) must be <= n_dim ({n_dim}) and e_dim ({e_dim})"
            )
        if float(e_rcut) <= 0.0 or float(a_rcut) <= 0.0:
            raise ValueError("e_rcut and a_rcut must be positive")
        if float(a_rcut) > float(e_rcut) + 1e-8:
            raise ValueError(f"a_rcut ({a_rcut}) must be <= e_rcut ({e_rcut})")
        if not 0.0 <= float(e_rcut_smth) < float(e_rcut):
            raise ValueError(
                f"e_rcut_smth ({e_rcut_smth}) must lie in [0, e_rcut) with e_rcut={e_rcut}"
            )
        if not 0.0 <= float(a_rcut_smth) < float(a_rcut):
            raise ValueError(
                f"a_rcut_smth ({a_rcut_smth}) must lie in [0, a_rcut) with a_rcut={a_rcut}"
            )
        if update_style not in ("res_avg", "res_incr", "res_residual"):
            raise ValueError(f"unknown update_style {update_style!r}")
        if update_residual_init not in ("const", "norm"):
            raise ValueError(
                f"update_residual_init must be 'const' or 'norm', got {update_residual_init!r}"
            )
        if sequential_update and update_style != "res_residual":
            raise ValueError("sequential_update only supports update_style='res_residual'")
        if sequential_update and not update_angle:
            raise ValueError("sequential_update requires update_angle=True")
        if int(a_compress_rate) < 0:
            raise ValueError(f"a_compress_rate must be >= 0, got {a_compress_rate}")
        if int(a_compress_e_rate) < 1:
            raise ValueError(f"a_compress_e_rate must be >= 1, got {a_compress_e_rate}")
        if int(a_compress_rate) > 0:
            if a_dim % int(a_compress_rate) != 0:
                raise ValueError(
                    f"a_dim ({a_dim}) must be divisible by a_compress_rate ({a_compress_rate})"
                )
            if (a_dim * int(a_compress_e_rate)) % (2 * int(a_compress_rate)) != 0:
                raise ValueError(
                    "a_dim * a_compress_e_rate must be divisible by 2 * a_compress_rate"
                )
            n_compress = a_dim // int(a_compress_rate)
            e_compress = a_dim // (2 * int(a_compress_rate)) * int(a_compress_e_rate)
            if a_compress_use_split and (n_dim < n_compress or e_dim < e_compress):
                raise ValueError(
                    "split compression needs n_dim >= "
                    f"{n_compress} and e_dim >= {e_compress}"
                )
        if float(sel_reduce_factor) <= 0.0:
            raise ValueError(f"sel_reduce_factor must be positive, got {sel_reduce_factor}")
        if float(fix_stat_std) < 0.0:
            raise ValueError(f"fix_stat_std must be >= 0, got {fix_stat_std}")
        if float(env_protection) < 0.0:
            raise ValueError(f"env_protection must be >= 0, got {env_protection}")
        if switch not in ("polynomial", "exponential"):
            raise ValueError(f"switch must be 'polynomial' or 'exponential', got {switch!r}")
        if switch == "exponential" and (float(e_rcut_smth) <= 0.0 or float(a_rcut_smth) <= 0.0):
            raise ValueError("exponential switch needs a positive smoothing distance")
        if float(switch_decay) <= 0.0:
            raise ValueError(f"switch_decay must be positive, got {switch_decay}")

        super().__init__(
            input_fields={"atom_embedding", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"},
            output_fields={"atom_feature"},
        )
        self.dim_embedding = int(dim_embedding)
        self.n_dim = int(n_dim)
        self.e_dim = e_dim
        self.a_dim = a_dim
        self.nlayers = nlayers
        self.e_rcut = float(e_rcut)
        self.e_rcut_smth = float(e_rcut_smth)
        self.e_sel = e_sel
        self.a_rcut = float(a_rcut)
        self.a_rcut_smth = float(a_rcut_smth)
        self.a_sel = a_sel
        self.axis_neuron = axis_neuron
        self.update_angle = bool(update_angle)
        self.edge_init_use_dist = bool(edge_init_use_dist)
        self.switch = switch
        self.switch_decay = float(switch_decay)
        self.env_protection = float(env_protection)
        self.stat_std = float(fix_stat_std) if float(fix_stat_std) > 0.0 else 1.0
        self.dim_feature_out = self.n_dim
        self.feature_irreps = f"{self.n_dim}x0e"
        self.act = resolve_scalar_activation(activation)

        if dynamic_normalize:
            neighbor_scale = self.e_sel / float(sel_reduce_factor)
            angular_scale = self.a_sel / float(sel_reduce_factor)
        else:
            neighbor_scale = float(self.e_sel)
            angular_scale = float(self.a_sel)

        self.edge_embed = nn.Linear(1, self.e_dim)
        self.angle_embed = nn.Linear(1, self.a_dim, bias=False) if self.update_angle else None
        block_kwargs = dict(
            n_dim=self.n_dim,
            e_dim=self.e_dim,
            a_dim=self.a_dim,
            axis_neuron=self.axis_neuron,
            n_multi_edge_message=n_multi_edge_message,
            update_angle=self.update_angle,
            update_style=update_style,
            update_residual=float(update_residual),
            update_residual_init=update_residual_init,
            activation=self.act,
            a_compress_rate=int(a_compress_rate),
            a_compress_e_rate=int(a_compress_e_rate),
            a_compress_use_split=bool(a_compress_use_split),
            sequential_update=bool(sequential_update),
            smooth_edge_update=bool(smooth_edge_update),
            neighbor_scale=neighbor_scale,
            angular_scale=angular_scale,
        )
        self.blocks = nn.ModuleList([RepFlowBlock(**block_kwargs) for _ in range(self.nlayers)])

    def __str__(self) -> str:
        return f"""
#########################################################################
# DPA3 (arXiv:2506.01686) — repflow on a line graph of order 2         #
# Node {self.n_dim}, edge {self.e_dim}, angle {self.a_dim}, layers {self.nlayers} #
# Atom embedding, dataset code, and energy head stay outside the Core. #
#########################################################################
"""

    def build(self, built_layers: List[nn.Module]) -> None:
        self.adopt_radius_stack(built_layers)
        for layer in built_layers:
            if isinstance(layer, BaseAtomEmbedding) and int(layer.dim_embedding) != self.n_dim:
                raise ValueError(
                    f"atom embedding width {layer.dim_embedding} must equal DPA3 n_dim {self.n_dim}"
                )
            if isinstance(layer, RangeSeparationLayer):
                if float(layer.cutoff_sr) + 1e-5 < self.e_rcut:
                    raise ValueError(
                        f"RangeSeparation cutoff_sr ({layer.cutoff_sr}) must be "
                        f">= DPA3 e_rcut ({self.e_rcut})"
                    )

    def _switch(self, distance: Tensor, cutoff: float, cuton: float) -> Tensor:
        if self.switch == "polynomial":
            return polynomial_transition(distance, cutoff, cuton)
        return exponential_switch(distance, cutoff, cuton, self.switch_decay)

    def get_output(
        self,
        atom_embedding: Tensor,
        Dij_sr: Tensor,
        vij_sr: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
    ) -> Dict[str, Tensor]:
        num_nodes = int(atom_embedding.shape[0])
        if atom_embedding.ndim != 2 or atom_embedding.shape[-1] != self.n_dim:
            raise ValueError(
                f"atom_embedding must be (N, {self.n_dim}), got {tuple(atom_embedding.shape)}"
            )
        self._check_edges(Dij_sr, vij_sr, idx_i_sr, idx_j_sr, num_nodes)

        keep = Dij_sr < self.e_rcut
        dist = Dij_sr[keep]
        vij = vij_sr[keep]
        idx_i = idx_i_sr[keep]
        idx_j = idx_j_sr[keep]
        edge_switch = self._switch(dist, self.e_rcut, self.e_rcut_smth)
        angle_switch = self._switch(dist, self.a_rcut, self.a_rcut_smth)
        denom = (dist + self.env_protection).clamp(min=1e-12)
        radial = (edge_switch / denom / self.stat_std).unsqueeze(-1)
        h = edge_switch.unsqueeze(-1) * vij / denom.square().unsqueeze(-1) / self.stat_std
        if self.edge_init_use_dist:
            edge = self.edge_embed(dist.unsqueeze(-1))
        else:
            edge = self.act(self.edge_embed(radial))

        if self.update_angle:
            pairs = angle_pairs(idx_i, dist, num_nodes, self.a_rcut)
            angle = self._embed_angles(pairs, vij, dist)
        else:
            pairs = AnglePairs(idx_i.new_empty(0), idx_i.new_empty(0), idx_i.new_empty(0))
            angle = edge.new_zeros(0, self.a_dim)
        angular_edge = dist < self.a_rcut
        node = self.act(atom_embedding)
        for block in self.blocks:
            node, edge, angle = block(
                node, edge, angle, h, edge_switch, angle_switch,
                idx_i, idx_j, pairs.edge_ij, pairs.edge_ik, pairs.center, angular_edge,
            )
        return {"atom_feature": node}

    def _embed_angles(self, pairs: AnglePairs, vij: Tensor, dist: Tensor) -> Tensor:
        assert self.angle_embed is not None
        if pairs.edge_ij.numel() == 0:
            cosine = dist.new_zeros(0, 1)
        else:
            left = vij[pairs.edge_ij]
            right = vij[pairs.edge_ik]
            span = (dist[pairs.edge_ij] * dist[pairs.edge_ik]).clamp(min=1e-12)
            cosine = (left * right).sum(dim=-1) / span
            cosine = (cosine.clamp(-1.0, 1.0) * (1.0 - 1e-6)).unsqueeze(-1)
            cosine = cosine / math.sqrt(math.pi)
        return self.angle_embed(cosine)

    def _check_edges(
        self,
        dist: Tensor,
        vij: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        num_nodes: int,
    ) -> None:
        aligned = (
            dist.ndim == 1
            and vij.ndim == 2
            and vij.shape[-1] == 3
            and idx_i.ndim == 1
            and idx_j.ndim == 1
            and dist.shape[0] == vij.shape[0] == idx_i.shape[0] == idx_j.shape[0]
        )
        if not aligned:
            raise ValueError(
                "Dij_sr, vij_sr, idx_i_sr, and idx_j_sr must describe the same edges "
                f"(got {tuple(dist.shape)}, {tuple(vij.shape)}, {tuple(idx_i.shape)}, {tuple(idx_j.shape)})"
            )
        if idx_i.numel() == 0:
            return
        if int(idx_i.min()) < 0 or int(idx_j.min()) < 0:
            raise ValueError("edge indices must be non-negative")
        if int(idx_i.max()) >= num_nodes or int(idx_j.max()) >= num_nodes:
            raise ValueError("edge index is outside the atom embedding")
