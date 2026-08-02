"""TECE uvSO2 interaction: ECE + Radial Rotary Attention.

Contains the edge-frame message rejector (:class:`UvSO2TensorProduct`) and the
layer wrapper (:class:`UvSO2Interaction`). Reuses TACE edge helpers / Enerzyme
``e3nn_nn`` for the node residual and gated readout path.

Adapted from https://github.com/xvzemin/tace (MIT), Xu et al. arXiv:2607.10664.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor, nn
from torch.nn import Linear
from torch_scatter import scatter_sum

from ..activation import get_scaled_activation
from ..blocks.radial_mlp import RadialMLP
from ..e3nn_nn import IrrepsLinear, get_gated_nonlinear, get_resnet_layer, to_possible_tp_irreps
from ..so3 import (
    ComplexProductBasis,
    GraphSoftmax,
    LayoutTransform,
    SO2Gate,
    so2_expand_index,
    uvSO2Linear,
)

_SCATTER_NORMS = ("avg_num_neighbors", "density", "no_cutoff_density")


class UvSO2TensorProduct(nn.Module):
    """Edge-frame SO(2) message with optional ECE and RRA."""

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel: int,
        num_head: int,
        use_temperature: bool,
        edge_ace_hidden: int,
        edge_wise_hidden: int,
        num_radial_basis: int,
        so2_linear_type: str,
        gate_m0: bool,
        use_so2_edge_ace: bool,
        use_graph_softmax: bool,
        reshape_in: LayoutTransform,
        reshape_out: LayoutTransform,
        scalar_act: torch.nn.Module,
        tensor_act: torch.nn.Module,
        use_radial_phase: bool,
    ) -> None:
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel = num_channel
        self.num_head = num_head
        self.edge_ace_hidden = edge_ace_hidden
        self.edge_wise_hidden = edge_wise_hidden or self.num_channel
        self.num_channel_per_head = self.edge_wise_hidden // self.num_head
        if self.edge_wise_hidden % self.num_head != 0:
            raise ValueError(
                f"edge_wise_hidden ({self.edge_wise_hidden}) must be divisible by "
                f"num_head ({self.num_head})"
            )
        self.so2_linear_type = so2_linear_type
        self.use_temperature = use_temperature
        self.use_graph_softmax = use_graph_softmax
        self.use_so2_edge_ace = use_so2_edge_ace
        self.reshape_in = reshape_in
        self.reshape_out = reshape_out
        self.use_radial_phase = use_radial_phase

        self.num_components, expand_index = so2_expand_index(self.mmax, self.lmax)
        self.weight_numel = self.num_components * self.num_channel * 2
        self.register_buffer("expand_index", expand_index, persistent=False)

        start_m = 0 if gate_m0 else 1
        if self.use_so2_edge_ace:
            self.num_gates = sum(lmax + 1 for _ in range(start_m, mmax + 1))
            num_components_out = [self.num_gates + lmax + 1] + [
                lmax + 1 for _ in range(1, mmax + 1)
            ]
            num_components_in = [lmax + 1] + [lmax + 1 for _ in range(1, mmax + 1)]
            self.split_list = [
                self.num_gates,
                (lmax + 1) + sum((lmax + 1) * 2 for _ in range(1, mmax + 1)),
            ]
        else:
            self.num_gates = sum(lmax + 1 - m for m in range(start_m, mmax + 1))
            num_components_out = [self.num_gates + lmax + 1] + [
                lmax + 1 - m for m in range(1, mmax + 1)
            ]
            num_components_in = [lmax + 1] + [
                lmax + 1 - m for m in range(1, mmax + 1)
            ]
            self.split_list = [
                self.num_gates,
                (lmax + 1) + sum((lmax + 1 - m) * 2 for m in range(1, mmax + 1)),
            ]

        hidden = (
            self.edge_ace_hidden if self.use_so2_edge_ace else self.edge_wise_hidden
        )
        self.linear_up = uvSO2Linear(
            mmax,
            lmax,
            self.num_channel * 2,
            hidden,
            num_components_out=num_components_out,
            weight_type=self.so2_linear_type,
        )
        self.nonlinearity = SO2Gate(
            mmax,
            lmax,
            hidden,
            channel_wise=self.use_so2_edge_ace,
            gate_m0=gate_m0,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        if self.use_so2_edge_ace:
            self.linear_glu = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel * 2,
                hidden,
                num_components_out=[lmax + 1]
                + [lmax + 1 for _ in range(1, mmax + 1)],
                weight_type=self.so2_linear_type,
            )
            self.ece = ComplexProductBasis(
                mmax, lmax, self.edge_ace_hidden, m1m2=">="
            )
            self.linear_coefs = uvSO2Linear(
                0,
                lmax,
                self.num_channel * 2,
                1,
                num_components_out=[self.ece.weight_numel],
                weight_type=self.so2_linear_type,
            )
        self.linear_down = uvSO2Linear(
            mmax,
            lmax,
            hidden,
            self.edge_wise_hidden,
            num_components_in=num_components_in,
            weight_type=self.so2_linear_type,
        )
        if self.use_graph_softmax:
            self.query_proj = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel,
                self.edge_wise_hidden,
                weight_type=self.so2_linear_type,
            )
            self.key_proj = uvSO2Linear(
                mmax,
                lmax,
                self.num_channel,
                self.edge_wise_hidden,
                weight_type=self.so2_linear_type,
            )
            if self.use_radial_phase:
                self.radial_proj = Linear(num_radial_basis, 2 * self.num_head)
            else:
                self.radial_proj = Linear(num_radial_basis, self.num_head)
            torch.nn.init.zeros_(self.radial_proj.weight)
            torch.nn.init.zeros_(self.radial_proj.bias)
            self.attention_scale = 1.0 / math.sqrt(
                self.num_channel_per_head * self.split_list[1]
            )
            self.graph_softmax = GraphSoftmax()
            if self.use_temperature:
                self.temperature_min = 0.25
                self.temperature_max = 4.0
                initial_temperature = 1.0
                initial_temperature_logit = math.log(
                    (initial_temperature - self.temperature_min)
                    / (self.temperature_max - initial_temperature)
                )
                self.temperature_logit = torch.nn.Parameter(
                    torch.full((self.num_head,), initial_temperature_logit)
                )

    def _complex_qk_attention(
        self, query: torch.Tensor, key: torch.Tensor, edge_feats: torch.Tensor
    ) -> torch.Tensor:
        B = query.size(0)
        H = self.num_head
        C = self.num_channel_per_head

        if self.use_radial_phase:
            radial_proj = self.radial_proj(edge_feats)
            radial_bias = radial_proj[:, :H]
            radial_phase = math.pi * torch.tanh(radial_proj[:, H:])
        else:
            radial_bias = self.radial_proj(edge_feats)

        n = self.lmax + 1
        query_m0 = query[:, :n].view(B, n, H, C)
        key_m0 = key[:, :n].view(B, n, H, C)
        score = (query_m0 * key_m0).sum(dim=(1, 3))

        if self.use_radial_phase:
            offset = n
            for m in range(1, self.mmax + 1):
                n_m = self.lmax + 1 - m
                query_m = query[:, offset : offset + 2 * n_m].view(B, 2, n_m, H, C)
                key_m = key[:, offset : offset + 2 * n_m].view(B, 2, n_m, H, C)
                offset += 2 * n_m
                phase = (m * radial_phase).view(B, 1, H, 1)
                cos_phase = torch.cos(phase)
                sin_phase = torch.sin(phase)
                key_real = cos_phase * key_m[:, 0] - sin_phase * key_m[:, 1]
                key_imag = sin_phase * key_m[:, 0] + cos_phase * key_m[:, 1]
                score = score + (
                    query_m[:, 0] * key_real + query_m[:, 1] * key_imag
                ).sum(dim=(1, 3))
        else:
            offset = n
            for m in range(1, self.mmax + 1):
                n_m = self.lmax + 1 - m
                query_m = query[:, offset : offset + 2 * n_m].view(B, 2, n_m, H, C)
                key_m = key[:, offset : offset + 2 * n_m].view(B, 2, n_m, H, C)
                offset += 2 * n_m
                score = score + (
                    query_m[:, 0] * key_m[:, 0] + query_m[:, 1] * key_m[:, 1]
                ).sum(dim=(1, 3))

        if self.use_temperature:
            temperature = self.temperature_min + (
                self.temperature_max - self.temperature_min
            ) * torch.sigmoid(self.temperature_logit)
            return score * self.attention_scale * temperature + radial_bias
        return score * self.attention_scale + radial_bias

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor],
        wigner: torch.Tensor,
        wigner_inv: torch.Tensor,
        radial_basis: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = w.size(0)
        x = self.reshape_in(x)
        m_ij = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
        m_ij = torch.bmm(wigner, m_ij)

        if self.use_graph_softmax:
            key = self.key_proj(m_ij[:, :, : self.num_channel])
            query = self.query_proj(m_ij[:, :, self.num_channel :])
            real_alpha = self._complex_qk_attention(query, key, radial_basis)

        w = w.view(num_edges, self.num_components, self.num_channel * 2)
        w = torch.index_select(w, dim=1, index=self.expand_index)
        m_ij = w * m_ij

        if self.use_so2_edge_ace:
            coefs = self.nonlinearity.scalar_act(
                self.linear_coefs(m_ij).squeeze(-1)
            )
            m_ij_2 = self.linear_glu(m_ij)
            m_ij = self.linear_up(m_ij)
            gate = m_ij.narrow(1, 0, self.split_list[0])
            m_ij = m_ij.narrow(1, self.split_list[0], self.split_list[1])
            m_ij = m_ij + self.nonlinearity(m_ij, gate) + self.ece(
                m_ij, m_ij_2, coefs
            )
        else:
            m_ij = self.linear_up(m_ij)
            gate = m_ij.narrow(1, 0, self.split_list[0])
            m_ij = m_ij.narrow(1, self.split_list[0], self.split_list[1])
            m_ij = self.nonlinearity(m_ij, gate)

        m_ij = self.linear_down(m_ij)

        if self.use_graph_softmax:
            real_alpha = self.graph_softmax(
                real_alpha,
                edge_index[1],
                num_nodes=num_nodes,
                exp_rescale=cutoff,
            )
            if cutoff is not None:
                real_alpha = real_alpha * cutoff
            real_alpha = real_alpha.view(num_edges, 1, self.num_head, 1)
            m_ij = m_ij.view(
                num_edges, m_ij.size(1), self.num_head, self.num_channel_per_head
            )
            m_ij = real_alpha * m_ij
            m_ij = m_ij.view(num_edges, m_ij.size(1), self.edge_wise_hidden)
        elif cutoff is not None:
            m_ij = m_ij * cutoff.unsqueeze(-1)

        m_ij = torch.bmm(wigner_inv, m_ij)
        return self.reshape_out.inverse(
            scatter_sum(m_ij, edge_index[1], dim=0, dim_size=num_nodes)
        )


class UvSO2Interaction(nn.Module):
    """SO(2) edge interaction with ECE + optional Radial Rotary Attention."""

    def __init__(
        self,
        *,
        layer: int,
        num_layers: int,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        mmax: int,
        irreps_in: o3.Irreps,
        edge_feats_channel: int,
        num_radial_basis: int,
        avg_num_neighbors: float,
        radial_mlp: list,
        radial_bias: bool = False,
        scatter_norm: str = "avg_num_neighbors",
        nonlinear: Optional[str] = "sigmoid_gate",
        edge_nonlinear: str = "so2_sigmoid_gate",
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        parity: bool = False,
        bias: bool = True,
        target_irreps: Optional[o3.Irreps] = None,
        correlation: int = 2,
        use_so2_edge_ace: bool = True,
        use_graph_softmax: bool = True,
        use_radial_phase: bool = True,
        use_temperature: bool = True,
        num_head: int = 1,
        edge_ace_hidden: Optional[int] = None,
        edge_wise_hidden: Optional[int] = None,
        so2_linear_type: str = "w1",
        gate_m0: bool = False,
        scalar_act: Optional[str] = None,
        tensor_act: Optional[str] = None,
    ) -> None:
        super().__init__()
        if parity:
            raise ValueError("UvSO2Interaction does not support parity=True (O(3))")
        if Lmax != lmax:
            raise ValueError(
                f"UvSO2Interaction requires Lmax == lmax; got Lmax={Lmax}, lmax={lmax}"
            )
        irreps_in = o3.Irreps(irreps_in)
        if irreps_in.lmax <= 0:
            raise ValueError(
                "UvSO2Interaction requires irreps_in.lmax > 0 "
                "(seed equivariant features before the first SO2 layer)"
            )
        if scatter_norm not in _SCATTER_NORMS:
            raise ValueError(
                f"Unknown scatter_norm={scatter_norm!r}; expected one of {_SCATTER_NORMS}"
            )
        if edge_nonlinear not in ("so2_sigmoid_gate", "so2_silu_gate"):
            raise ValueError(
                f"Unknown edge_nonlinear={edge_nonlinear!r}; "
                "expected so2_sigmoid_gate or so2_silu_gate"
            )

        self.layer = layer
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.num_channel = num_channel
        self.Lmax = Lmax
        self.lmax = lmax
        self.mmax = mmax
        self.correlation = int(correlation)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.register_buffer(
            "_avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )
        self.resnet_type = resnet_type
        self.resnet_linear_type = resnet_linear_type
        self.use_graph_softmax = use_graph_softmax
        # Attention replaces neighbor-count normalization.
        self.scatter_norm = None if use_graph_softmax else scatter_norm

        self.irreps_in = irreps_in
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1)
        if target_irreps is None:
            target_irreps = o3.Irreps(
                [(1, (l, (-1) ** l)) for l in range(Lmax + 1)]
            )
        self.target_irreps = o3.Irreps(target_irreps)

        tp_base = to_possible_tp_irreps(self.irreps_in, self.irreps_sh, False, lmax=lmax)
        self.irreps_out = (tp_base * num_channel).regroup()

        last = layer == num_layers - 1
        if last:
            self.irreps_sc = (self.target_irreps * num_channel).regroup()
        elif self.correlation == 1:
            self.irreps_sc = o3.Irreps(
                [(num_channel, ir) for _, ir in self.irreps_out if ir.l <= Lmax]
            )
        else:
            sc_base = to_possible_tp_irreps(
                self.irreps_out, self.irreps_out, False, lmax=Lmax
            )
            self.irreps_sc = (sc_base * num_channel).regroup()

        edge_act = edge_nonlinear.split("_")[1]
        s_act = get_scaled_activation(scalar_act or edge_act)
        t_act = get_scaled_activation(tensor_act or edge_act)
        self.edge_ace_hidden = edge_ace_hidden or num_channel
        self.edge_wise_hidden = edge_wise_hidden or num_channel

        self.linear_up = IrrepsLinear(self.irreps_in, self.irreps_in, bias=bias)
        self.rejector = UvSO2TensorProduct(
            mmax=self.mmax,
            lmax=self.lmax,
            num_channel=self.num_channel,
            num_radial_basis=num_radial_basis,
            num_head=num_head,
            use_temperature=use_temperature,
            edge_ace_hidden=self.edge_ace_hidden,
            edge_wise_hidden=self.edge_wise_hidden,
            so2_linear_type=so2_linear_type,
            gate_m0=gate_m0,
            use_so2_edge_ace=use_so2_edge_ace,
            use_graph_softmax=use_graph_softmax,
            reshape_in=LayoutTransform(self.irreps_in),
            reshape_out=LayoutTransform(
                o3.Irreps(
                    [(self.edge_wise_hidden, ir) for _, ir in self.irreps_out]
                )
            ),
            scalar_act=s_act,
            tensor_act=t_act,
            use_radial_phase=use_radial_phase,
        )

        mid = o3.Irreps([(self.edge_wise_hidden, ir) for _, ir in self.irreps_out])
        self.nonlinearity, self.linear_nonlinearity, linear_down_out = get_gated_nonlinear(
            nonlinear, mid, self.irreps_out, bias=bias
        )
        self.linear_down = IrrepsLinear(
            mid.simplify(), linear_down_out, bias=bias
        )

        self.edge_info = RadialMLP(
            [edge_feats_channel] + list(radial_mlp) + [self.rejector.weight_numel],
            use_layer_norm=False,
            use_offset=False,
            bias=radial_bias,
        )

        if self.scatter_norm in ("density", "no_cutoff_density"):
            self.edge_density = RadialMLP(
                [edge_feats_channel, 64, 1],
                use_layer_norm=False,
                use_offset=False,
                bias=radial_bias,
            )
            self.alpha = nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.apply_density_cutoff = self.scatter_norm != "no_cutoff_density"

        if (use_first_resnet or layer > 0) and resnet_type == "BB":
            self.resnetBB = get_resnet_layer(
                self.irreps_in,
                self.irreps_sc,
                bias=bias,
                num_elements=num_elements,
                resnet_type=resnet_linear_type,
            )

    def forward(
        self,
        node_feats: Tensor,
        node_attrs: Tensor,
        edge_feats: Tensor,
        edge_index: Tensor,
        wigner: Tensor,
        wigner_inv: Tensor,
        radial_basis: Tensor,
        cutoff: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        sc = None
        if hasattr(self, "resnetBB"):
            if self.resnet_linear_type == "aware":
                sc = self.resnetBB(node_feats, node_attrs)
            else:
                sc = self.resnetBB(node_feats)

        node_feats = self.linear_up(node_feats)
        conv_weights = self.edge_info(edge_feats)
        # Cutoff is applied inside rejector (attention path) or after (non-attn).
        m_i = self.linear_down(
            self.rejector(
                node_feats,
                conv_weights,
                edge_index,
                cutoff,
                wigner,
                wigner_inv,
                radial_basis,
            )
        )

        if hasattr(self, "edge_density"):
            density = torch.tanh(self.edge_density(edge_feats) ** 2)
            if cutoff is not None and self.apply_density_cutoff:
                density = density * cutoff
            density = scatter_sum(
                density, edge_index[1], dim=0, dim_size=node_attrs.size(0)
            )
            density = density * self.beta + self.alpha
            density = density.masked_fill(density == 0, 1e-9)
            m_i = m_i / density
        elif self.scatter_norm == "avg_num_neighbors":
            m_i = m_i / self._avg_num_neighbors

        m_i = self.linear_nonlinearity(self.nonlinearity(m_i))
        return m_i, sc
