"""GemNet dense maps, residuals, and directional interaction blocks.

An original implementation of the published block structure. Variance scales
stay at 1; the reference code fits them on a sample of activations and then
freezes them. SiLU is scaled by ``1 / 0.6``, and every residual and skip is
scaled by ``1/√2`` (three-way skips by ``1/√3``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, init

from ..functional import segment_sum_coo


_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_3 = 1.0 / math.sqrt(3.0)
_SILU_SCALE = 1.0 / 0.6


def he_orthogonal_(tensor: Tensor) -> Tensor:
    """He-scaled orthogonal map. Last axis is the output axis of a 3-D kernel."""
    if tensor.ndim not in (2, 3):
        raise ValueError(f"he_orthogonal_ expects a 2-D or 3-D tensor, got {tensor.ndim}")
    with torch.no_grad():
        init.orthogonal_(tensor)
        if tensor.ndim == 3:
            reduce_dims = (0, 1)
            fan_in = int(tensor.shape[0] * tensor.shape[1])
        else:
            reduce_dims = (1,)
            fan_in = int(tensor.shape[1])
        if fan_in < 2:
            tensor.mul_(fan_in ** -0.5)
            return tensor
        var, mean = torch.var_mean(tensor, dim=reduce_dims, unbiased=True, keepdim=True)
        tensor.sub_(mean).div_((var + 1e-6).sqrt())
        tensor.mul_(fan_in ** -0.5)
    return tensor


class GemNetSiLU(Module):
    """``silu(x) / 0.6``, the GemNet activation. Distinct from e3nn ``ScaledSiLU``."""

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x) * _SILU_SCALE


class GemNetDense(Module):
    """Bias-free linear map with optional GemNet SiLU and He-orthogonal init."""

    def __init__(self, in_features: int, out_features: int, activation: bool) -> None:
        super().__init__()
        self.linear = Linear(in_features, out_features, bias=False)
        he_orthogonal_(self.linear.weight)
        self.activation = GemNetSiLU() if activation else None

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        if self.activation is not None:
            y = self.activation(y)
        return y


class GemNetResidual(Module):
    """``(x + MLP(x)) / √2`` with bias-free activated dense layers."""

    def __init__(self, units: int, n_layers: int = 2) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.layers = ModuleList(
            [GemNetDense(units, units, activation=True) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return (x + y) * _INV_SQRT_2


class VarianceScale(Module):
    """Frozen multiplier, default 1. Included so a fitted scale can be loaded."""

    def __init__(self, init_value: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(init_value)))

    def forward(self, y: Tensor) -> Tensor:
        return y * self.scale


class BasisDownProjection(Module):
    """Contract the radial Bessel axis: ``(E, L, N) → (E, emb, L)``."""

    def __init__(self, num_angular: int, num_radial: int, emb_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_angular, num_radial, emb_size))
        he_orthogonal_(self.weight)

    def forward(self, radial: Tensor) -> Tensor:
        return torch.einsum("eln,lni->eil", radial, self.weight)


class BilinearAggregate(Module):
    """Sum ``Y · m`` over neighbor edges, then a bilinear map with the radial factor.

    ``radial_w`` has shape ``(E, interm, n_sph)`` and lives on the destination
    edge. ``weight`` has shape ``(emb_in, interm, units_out)``.
    """

    def __init__(self, emb_in: int, emb_interm: int, units_out: int) -> None:
        super().__init__()
        self.emb_in = emb_in
        self.units_out = units_out
        self.weight = torch.nn.Parameter(torch.empty(emb_in, emb_interm, units_out))
        he_orthogonal_(self.weight)

    def forward(
        self,
        msg: Tensor,
        idx_src: Tensor,
        idx_dst: Tensor,
        angular: Tensor,
        radial_w: Tensor,
        num_edges: int,
    ) -> Tensor:
        if num_edges == 0:
            return msg.new_zeros(0, self.units_out)
        if idx_dst.numel() == 0:
            return msg.new_zeros(num_edges, self.units_out)
        gathered = msg[idx_src]
        contrib = angular.unsqueeze(-1) * gathered.unsqueeze(1)
        flat = contrib.reshape(contrib.shape[0], -1)
        summed = segment_sum_coo(flat, idx_dst, dim_size=num_edges)
        sum_k = summed.view(num_edges, angular.shape[-1], self.emb_in)
        mixed = torch.einsum("eil,els->eis", radial_w, sum_k)
        return torch.einsum("eis,sio->eo", mixed, self.weight)


class EdgeEmbedding(Module):
    """``Dense([h_receiver, h_sender, edge_attr])`` with GemNet SiLU."""

    def __init__(self, atom_dim: int, attr_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dense = GemNetDense(2 * atom_dim + attr_dim, out_dim, activation=True)

    def forward(self, h: Tensor, attr: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:
        if idx_i.numel() == 0:
            return attr.new_zeros(0, self.dense.linear.out_features)
        return self.dense(torch.cat([h[idx_i], h[idx_j], attr], dim=-1))


class AtomUpdate(Module):
    """Scatter RBF-gated edge states onto the receiver and mix them."""

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_rbf: int,
        n_hidden: int,
    ) -> None:
        super().__init__()
        self.emb_edge = emb_edge
        self.emb_atom = emb_atom
        self.dense_rbf = GemNetDense(emb_rbf, emb_edge, activation=False)
        self.scale = VarianceScale()
        self.dense = GemNetDense(emb_edge, emb_atom, activation=True)
        self.residuals = ModuleList(
            [GemNetResidual(emb_atom) for _ in range(n_hidden)]
        )

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        num_nodes: int,
    ) -> Tensor:
        if messages.numel() == 0:
            scattered = messages.new_zeros(num_nodes, self.emb_edge)
        else:
            scattered = segment_sum_coo(
                messages * self.dense_rbf(rbf), idx_i, dim_size=num_nodes
            )
        hidden = self.dense(self.scale(scattered))
        for layer in self.residuals:
            hidden = layer(hidden)
        return hidden


class _EdgeUpdate(Module):
    """Residual edge update, atom update, and the atom-to-edge feedback."""

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_rbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        merge_scale: float,
    ) -> None:
        super().__init__()
        self.merge_scale = float(merge_scale)
        self.dense_ca = GemNetDense(emb_edge, emb_edge, activation=True)
        self.layers_before_skip = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_before_skip)]
        )
        self.layers_after_skip = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_after_skip)]
        )
        self.atom_update = AtomUpdate(emb_atom, emb_edge, emb_rbf, num_atom)
        self.concat = EdgeEmbedding(emb_atom, emb_edge, emb_edge)
        self.residual_m = ModuleList(
            [GemNetResidual(emb_edge) for _ in range(num_concat)]
        )

    def update(
        self,
        h: Tensor,
        m: Tensor,
        interacted: Tensor,
        rbf_h: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = (self.dense_ca(m) + interacted) * self.merge_scale
        for layer in self.layers_before_skip:
            x = layer(x)
        m = (m + x) * _INV_SQRT_2
        for layer in self.layers_after_skip:
            m = layer(m)
        h = (h + self.atom_update(m, rbf_h, idx_i, h.shape[0])) * _INV_SQRT_2
        m2 = self.concat(h, m, idx_i, idx_j)
        for layer in self.residual_m:
            m2 = layer(m2)
        m = (m + m2) * _INV_SQRT_2
        return h, m


class TripletInteraction(Module):
    """One-hop directional update. Messages on ``b→a`` are filtered by ``d_ba``."""

    def __init__(
        self,
        emb_edge: int,
        emb_trip: int,
        emb_bilinear: int,
        emb_rbf: int,
        emb_cbf: int,
    ) -> None:
        super().__init__()
        self.dense_ba = GemNetDense(emb_edge, emb_edge, activation=True)
        self.mlp_rbf = GemNetDense(emb_rbf, emb_edge, activation=False)
        self.scale_rbf = VarianceScale()
        self.down = GemNetDense(emb_edge, emb_trip, activation=True)
        self.bilinear = BilinearAggregate(emb_trip, emb_cbf, emb_bilinear)
        self.scale_cbf = VarianceScale()
        self.up_ca = GemNetDense(emb_bilinear, emb_edge, activation=True)
        self.up_ac = GemNetDense(emb_bilinear, emb_edge, activation=True)

    def forward(
        self,
        m: Tensor,
        rbf: Tensor,
        radial_w: Tensor,
        angular: Tensor,
        idx_edge: Tensor,
        idx_other: Tensor,
        swap: Tensor,
    ) -> Tensor:
        x = self.scale_rbf(self.dense_ba(m) * self.mlp_rbf(rbf))
        x = self.down(x)
        aggregated = self.bilinear(
            x, idx_other, idx_edge, angular, radial_w, m.shape[0]
        )
        aggregated = self.scale_cbf(aggregated)
        direct = self.up_ca(aggregated)
        swapped = self.up_ac(aggregated)[swap] if swap.numel() else direct
        return (direct + swapped) * _INV_SQRT_2


class QuadrupletInteraction(Module):
    """Two-hop update along ``d → b − a ← c``, summed onto ``c→a``."""

    def __init__(
        self,
        emb_edge: int,
        emb_quad: int,
        emb_bilinear: int,
        emb_rbf: int,
        emb_cbf: int,
        emb_sbf: int,
    ) -> None:
        super().__init__()
        self.dense_db = GemNetDense(emb_edge, emb_edge, activation=True)
        self.mlp_rbf = GemNetDense(emb_rbf, emb_edge, activation=False)
        self.scale_rbf = VarianceScale()
        self.mlp_cbf = GemNetDense(emb_cbf, emb_quad, activation=False)
        self.scale_cbf = VarianceScale()
        self.down = GemNetDense(emb_edge, emb_quad, activation=True)
        self.bilinear = BilinearAggregate(emb_quad, emb_sbf, emb_bilinear)
        self.scale_sbf = VarianceScale()
        self.up_ca = GemNetDense(emb_bilinear, emb_edge, activation=True)
        self.up_ac = GemNetDense(emb_bilinear, emb_edge, activation=True)

    def forward(
        self,
        m: Tensor,
        rbf: Tensor,
        cbf: Tensor,
        radial_w: Tensor,
        angular: Tensor,
        edge_db: Tensor,
        edge_ca: Tensor,
        swap: Tensor,
    ) -> Tensor:
        x = self.scale_rbf(self.dense_db(m) * self.mlp_rbf(rbf))
        x = self.down(x)
        if edge_ca.numel() == 0:
            aggregated = m.new_zeros(m.shape[0], self.bilinear.units_out)
        else:
            filtered = self.scale_cbf(x[edge_db] * self.mlp_cbf(cbf))
            aggregated = self.bilinear(
                filtered,
                torch.arange(filtered.shape[0], device=filtered.device),
                edge_ca,
                angular,
                radial_w,
                m.shape[0],
            )
        aggregated = self.scale_sbf(aggregated)
        direct = self.up_ca(aggregated)
        swapped = self.up_ac(aggregated)[swap] if swap.numel() else direct
        return (direct + swapped) * _INV_SQRT_2


class GemNetInteraction(Module):
    """One GemNet block. Quadruplet weights exist only when ``triplets_only`` is false."""

    def __init__(
        self,
        emb_atom: int,
        emb_edge: int,
        emb_trip: int,
        emb_quad: int,
        emb_rbf: int,
        emb_cbf: int,
        emb_sbf: int,
        emb_bil_trip: int,
        emb_bil_quad: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        triplets_only: bool,
    ) -> None:
        super().__init__()
        self.triplets_only = triplets_only
        self.trip = TripletInteraction(
            emb_edge, emb_trip, emb_bil_trip, emb_rbf, emb_cbf
        )
        self.quad: Optional[QuadrupletInteraction]
        if triplets_only:
            self.quad = None
            merge_scale = _INV_SQRT_2
        else:
            self.quad = QuadrupletInteraction(
                emb_edge, emb_quad, emb_bil_quad, emb_rbf, emb_cbf, emb_sbf
            )
            merge_scale = _INV_SQRT_3
        self.edges = _EdgeUpdate(
            emb_atom,
            emb_edge,
            emb_rbf,
            num_before_skip,
            num_after_skip,
            num_concat,
            num_atom,
            merge_scale,
        )

    def forward(
        self,
        h: Tensor,
        m: Tensor,
        rbf3: Tensor,
        rbf_h: Tensor,
        radial_w3: Tensor,
        angular3: Tensor,
        idx_edge: Tensor,
        idx_other: Tensor,
        swap: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rbf4: Optional[Tensor] = None,
        cbf4: Optional[Tensor] = None,
        radial_w4: Optional[Tensor] = None,
        angular4: Optional[Tensor] = None,
        edge_db: Optional[Tensor] = None,
        edge_ca: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        interacted = self.trip(
            m, rbf3, radial_w3, angular3, idx_edge, idx_other, swap
        )
        if self.quad is not None:
            assert (
                rbf4 is not None
                and cbf4 is not None
                and radial_w4 is not None
                and angular4 is not None
                and edge_db is not None
                and edge_ca is not None
            )
            interacted = interacted + self.quad(
                m, rbf4, cbf4, radial_w4, angular4, edge_db, edge_ca, swap
            )
        return self.edges.update(h, m, interacted, rbf_h, idx_i, idx_j)
