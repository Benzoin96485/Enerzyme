"""TACE spherical interaction stack: edge embed/update, CgtpInteraction, CgtpACE.

Adapted from https://github.com/xvzemin/tace (MIT). Shared e3nn helpers live in
``enerzyme.models.e3nn_nn``; radial MLP in ``enerzyme.models.blocks.radial_mlp``.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import torch
from e3nn import o3
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..blocks.radial_mlp import RadialMLP
from ..e3nn_nn import (
    Activation,
    ElementIrrepsLinear,
    IrrepsLinear,
    O3ScatterTensorProduct,
    UUUTensorProduct,
    get_gated_nonlinear,
    get_resnet_layer,
    to_possible_tp_irreps,
)


# ---------------------------------------------------------------------------
# Edge embedding / update (TACE-specific)
# ---------------------------------------------------------------------------


class IdentityEdgeEmbedding(nn.Module):
    def __init__(
        self, num_radial_basis: int, num_channel: int, num_elements: int, bias: bool = False
    ):
        super().__init__()
        self.out_dim = num_radial_basis

    def forward(self, node_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor) -> Tensor:
        return edge_feats


class NonLinearEdgeEmbedding(nn.Module):
    def __init__(
        self, num_radial_basis: int, num_channel: int, num_elements: int, bias: bool = False
    ):
        super().__init__()
        self.out_dim = num_channel
        self.radial_proj = IrrepsLinear(
            f"{num_radial_basis}x0e", f"{num_channel}x0e", bias=bias
        )
        self.act = Activation(self.radial_proj.irreps_out, [torch.nn.SiLU()])

    def forward(self, node_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor) -> Tensor:
        return self.act(self.radial_proj(edge_feats))


class IdentityEdgeUpdate(nn.Module):
    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ):
        super().__init__()
        self.out_dim = edge_embedding_channel

    def forward(self, node_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor) -> Tensor:
        return edge_feats


class Element2EdgeUpdate(nn.Module):
    """Concatenate edge feats with target/source element embeddings."""

    def __init__(
        self,
        num_elements: int,
        num_channel: int,
        edge_embedding_channel: int,
        bias: bool = False,
    ):
        super().__init__()
        self.out_dim = edge_embedding_channel + num_channel * 2
        self.source_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        self.target_embedding = IrrepsLinear(
            f"{num_elements}x0e", f"{num_channel}x0e", bias=bias
        )
        with torch.no_grad():
            self.source_embedding.weight.uniform_(-0.001, 0.001)
            self.target_embedding.weight.uniform_(-0.001, 0.001)

    def forward(self, node_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor) -> Tensor:
        tgt = self.target_embedding(node_attrs[edge_index[1]])
        src = self.source_embedding(node_attrs[edge_index[0]])
        return torch.cat([edge_feats, tgt, src], dim=-1)


EDGE_EMBEDDING = {
    "identity": IdentityEdgeEmbedding,
    "nonlinear": NonLinearEdgeEmbedding,
}

EDGE_UPDATE = {
    "identity": IdentityEdgeUpdate,
    "element2": Element2EdgeUpdate,
}


# ---------------------------------------------------------------------------
# CgtpInteraction
# ---------------------------------------------------------------------------


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
            if not parity:
                target_irreps = o3.Irreps(
                    [(1, (l, (-1) ** l)) for l in range(Lmax + 1)]
                )
            else:
                target_irreps = o3.Irreps.spherical_harmonics(Lmax, p=-1)

        tp_base = to_possible_tp_irreps(self.irreps_in, self.irreps_sh, parity, lmax=lmax)
        self.irreps_out = (tp_base * num_channel).regroup()

        last = layer == num_layers - 1
        if last:
            self.irreps_sc = (o3.Irreps(target_irreps) * num_channel).regroup()
        else:
            sc_base = to_possible_tp_irreps(
                self.irreps_in, self.irreps_sh, parity, lmax=Lmax
            )
            self.irreps_sc = (sc_base * num_channel).regroup()

        self.linear_up = IrrepsLinear(self.irreps_in, self.irreps_in, bias=bias)
        self.rejector = O3ScatterTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_out, l1l2=l1l2
        )

        mid = o3.Irreps([(num_channel, ir) for _, ir in self.irreps_out])
        self.nonlinearity, self.linear_nonlinearity, linear_down_out = get_gated_nonlinear(
            nonlinear, mid, self.irreps_out, bias=bias
        )
        self.linear_down = IrrepsLinear(
            self.rejector.irreps_out.simplify(), linear_down_out, bias=bias
        )

        self.edge_info = RadialMLP(
            [edge_feats_channel] + list(radial_mlp) + [self.rejector.weight_numel],
            use_layer_norm=False,
            use_offset=False,
            bias=radial_bias,
        )

        if scatter_norm in ("density", "no_cutoff_density"):
            self.edge_density = RadialMLP(
                [edge_feats_channel, 64, 1],
                use_layer_norm=False,
                use_offset=False,
                bias=radial_bias,
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


# ---------------------------------------------------------------------------
# CgtpACE
# ---------------------------------------------------------------------------


class CgtpACE(nn.Module):
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
        correlation: int,
        target_irreps: o3.Irreps,
        l1l2: Optional[str] = None,
        parity: bool = False,
        bias: bool = True,
        agnostic: bool = False,
    ) -> None:
        super().__init__()
        if parity and correlation > 2:
            raise ValueError(
                "CgtpACE with parity=True currently requires correlation < 3."
            )
        if correlation >= 3:
            warnings.warn(
                "CgtpACE correlation>=3 is expensive without fused backends.",
                stacklevel=2,
            )

        self.layer = layer
        self.correlation = correlation
        self.num_channel = num_channel
        self.num_elements = num_elements
        self.agnostic = agnostic
        self.last_layer = layer == num_layers - 1
        self.scale = 1.0 / math.sqrt(2.0)

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_hidden = o3.Irreps([(num_channel, ir) for _, ir in self.irreps_in])
        target_irreps = o3.Irreps(target_irreps)

        self.irreps_tp_out_list: List[o3.Irreps] = []
        for nu in range(2, correlation + 1):
            if nu == correlation:
                if self.last_layer:
                    self.irreps_tp_out_list.append(
                        (target_irreps * num_channel).regroup()
                    )
                else:
                    base = to_possible_tp_irreps(
                        self.irreps_hidden, self.irreps_hidden, parity, Lmax
                    )
                    self.irreps_tp_out_list.append((base * num_channel).regroup())
            else:
                self.irreps_tp_out_list.append(
                    to_possible_tp_irreps(
                        self.irreps_hidden, self.irreps_hidden, parity, lmax
                    )
                )

        if correlation == 1:
            coefs_base = o3.Irreps(
                [(num_channel, ir) for _, ir in self.irreps_in if ir.l <= Lmax]
            )
        else:
            coefs_base = (
                to_possible_tp_irreps(self.irreps_in, self.irreps_in, parity, Lmax)
                * num_channel
            ).regroup()
        if self.last_layer:
            self.irreps_coefs_out = (target_irreps * num_channel).regroup()
        else:
            self.irreps_coefs_out = coefs_base

        self.irreps_out = o3.Irreps(
            [(num_channel, ir) for _, ir in self.irreps_coefs_out]
        )

        coefs_cls = IrrepsLinear if agnostic else ElementIrrepsLinear
        coefs_kwargs = {"bias": bias}
        if not agnostic:
            coefs_kwargs["num_elements"] = num_elements

        self.coefs = nn.ModuleList()
        self.coefs.append(
            coefs_cls(
                o3.Irreps([(num_channel, ir) for _, ir in self.irreps_hidden]).simplify(),
                self.irreps_coefs_out,
                **coefs_kwargs,
            )
        )

        self.aces = nn.ModuleList()
        product_in1 = self.irreps_hidden
        for nu in range(2, correlation + 1):
            ace = UUUTensorProduct(
                irreps_in1=product_in1,
                irreps_in2=self.irreps_hidden,
                irreps_out=self.irreps_tp_out_list[nu - 2],
                l1l2=l1l2,
                identical_inputs=(nu == 2),
                warning=(nu == 2 and correlation >= 3),
            )
            self.aces.append(ace)
            self.coefs.append(
                coefs_cls(
                    o3.Irreps(
                        [(num_channel, ir) for _, ir in ace.irreps_out]
                    ).simplify(),
                    self.irreps_coefs_out,
                    **coefs_kwargs,
                )
            )
            product_in1 = ace.irreps_out

        if self.irreps_in.simplify() != self.irreps_hidden.simplify():
            self.linear_up = IrrepsLinear(
                self.irreps_in, self.irreps_hidden, bias=bias
            )
        else:
            self.linear_up = nn.Identity()

        self.linear = IrrepsLinear(
            o3.Irreps([(num_channel, ir) for _, ir in self.irreps_coefs_out]),
            self.irreps_out,
            bias=bias,
        )

    def forward(
        self,
        node_feats: Tensor,
        node_attrs: Tensor,
        sc: Optional[Tensor] = None,
    ) -> Tensor:
        node_feats = self.linear_up(node_feats)
        if self.agnostic:
            outs = self.coefs[0](node_feats)
        else:
            outs = self.coefs[0](node_feats, node_attrs)

        corr = {1: node_feats}
        for nu in range(2, self.correlation + 1):
            corr[nu] = self.aces[nu - 2](corr[nu - 1], node_feats)
            if self.agnostic:
                outs = outs + self.coefs[nu - 1](corr[nu])
            else:
                outs = outs + self.coefs[nu - 1](corr[nu], node_attrs)

        outs = self.linear(outs)
        if sc is not None:
            outs = outs + sc
        return outs
