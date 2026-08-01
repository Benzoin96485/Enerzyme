"""CgtpInteraction — spherical TACE message-passing block.

Adapted from https://github.com/xvzemin/tace (MIT).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..linear import IrrepsLinear, ScalarMLP
from ..paths import to_possible_tp_irreps
from ..residual import get_resnet_layer
from ..tensor_product import O3ScatterTensorProduct
from .nonlinear import get_nonlinear_layer


class CgtpInteraction(nn.Module):
    def __init__(
        self,
        *,
        layer: int,
        num_layers: int,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        irreps_in: o3.Irreps,
        edge_feats_channel: int,
        avg_num_neighbors: float,
        radial_mlp: list,
        radial_bias: bool = False,
        scatter_norm: str = "avg_num_neighbors",
        nonlinear: Optional[str] = "sigmoid_gate",
        l1l2: Optional[str] = None,
        resnet_type: str = "BB",
        resnet_linear_type: str = "aware",
        use_first_resnet: bool = False,
        parity: bool = False,
        bias: bool = True,
        target_irreps: Optional[o3.Irreps] = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.num_channel = num_channel
        self.Lmax = Lmax
        self.lmax = lmax
        self.scatter_norm = scatter_norm
        self.resnet_type = resnet_type
        self.resnet_linear_type = resnet_linear_type
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.register_buffer(
            "_avg_num_neighbors",
            torch.tensor(avg_num_neighbors, dtype=torch.get_default_dtype()),
        )

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=lmax, p=-1)
        if target_irreps is None:
            target_irreps = o3.Irreps.spherical_harmonics(Lmax, p=-1 if parity else 1)
            if not parity:
                target_irreps = o3.Irreps(
                    [(1, (l, (-1) ** l)) for l in range(Lmax + 1)]
                )

        # Message irreps: all possible node⊗Y products truncated to lmax (or Lmax for corr=1)
        tp_base = to_possible_tp_irreps(self.irreps_in, self.irreps_sh, parity, lmax=lmax)
        self.irreps_out = (tp_base * num_channel).regroup()

        last = layer == num_layers - 1
        if last:
            self.irreps_sc = (o3.Irreps(target_irreps) * num_channel).regroup()
        else:
            sc_base = to_possible_tp_irreps(self.irreps_in, self.irreps_sh, parity, lmax=Lmax)
            self.irreps_sc = (sc_base * num_channel).regroup()

        self.linear_up = IrrepsLinear(self.irreps_in, self.irreps_in, bias=bias)
        self.rejector = O3ScatterTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, l1l2=l1l2
        )

        mid = o3.Irreps([(num_channel, ir) for _, ir in self.irreps_out])
        self.nonlinearity, self.linear_nonlinearity, linear_down_out = get_nonlinear_layer(
            nonlinear, mid, self.irreps_out, bias=bias
        )
        self.linear_down = IrrepsLinear(
            self.rejector.irreps_out.simplify(), linear_down_out, bias=bias
        )

        self.edge_info = ScalarMLP(
            [edge_feats_channel] + list(radial_mlp) + [self.rejector.weight_numel],
            bias=radial_bias,
            act="silu",
            layer_norm=edge_feats_channel != edge_feats_channel,  # False
        )

        if scatter_norm in ("density", "no_cutoff_density"):
            self.edge_density = ScalarMLP(
                [edge_feats_channel, 64, 1], bias=radial_bias, act="silu"
            )
            self.alpha = nn.Parameter(torch.tensor(self.avg_num_neighbors))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.apply_density_cutoff = scatter_norm != "no_cutoff_density"

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
        edge_attrs: Tensor,
        edge_index: Tensor,
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
        if cutoff is not None:
            conv_weights = conv_weights * cutoff

        m_i = self.linear_down(
            self.rejector(node_feats, edge_attrs, conv_weights, edge_index)
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
