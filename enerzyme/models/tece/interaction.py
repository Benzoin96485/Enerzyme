"""TECE uvSO2 interaction wrapping ECE + RRA.

Reuses TACE edge embed/update registries and Enerzyme e3nn_nn helpers.
Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..blocks.radial_mlp import RadialMLP
from ..e3nn_nn import IrrepsLinear, get_gated_nonlinear, get_resnet_layer, to_possible_tp_irreps
from ..activation import get_scaled_activation
from ..so3 import LayoutTransform
from .fused import UvSO2TensorProduct

_SCATTER_NORMS = ("avg_num_neighbors", "density", "no_cutoff_density")


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
