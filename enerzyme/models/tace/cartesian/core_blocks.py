"""Cartesian TACE path helpers (ICT / Cartesian-3j).

Uses vendored ``enerzyme.models.tace.cartnn`` (from tace v0.1.0, MIT).
"""

from __future__ import annotations

import math
import string
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch_scatter import scatter_sum

from ..cartnn import ICTD, CartesianHarmonics
from ...blocks.radial_mlp import RadialMLP
from ..interaction import EDGE_UPDATE


def add_dict_to_left(t1: Dict[int, Tensor], t2: Dict[int, Tensor]) -> Dict[int, Tensor]:
    for k, v in t2.items():
        if k in t1:
            t1[k] = t1[k] + v
        else:
            t1[k] = v
    return t1


def dict_to_scalar(feats: Dict[int, Tensor]) -> Tensor:
    if 0 not in feats:
        raise KeyError("Cartesian features missing l=0")
    t = feats[0]
    return t.reshape(t.shape[0], -1)


class RankLinear(nn.Module):
    def __init__(self, in_c: int, out_c: int, l: int, bias: bool = False):
        super().__init__()
        self.l = l
        self.alpha = 1.0 / math.sqrt(max(in_c, 1))
        self.weight = nn.Parameter(torch.empty(out_c, in_c))
        torch.nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
        if bias and l == 0:
            self.bias = nn.Parameter(torch.zeros(out_c))
        else:
            self.register_parameter("bias", None)
        letters = [c for c in string.ascii_letters[3:] if c not in ("C", "z")]
        idx = "".join(letters[:l])
        self.expr = f"Cc,bc{idx}->bC{idx}"

    def forward(self, t: Tensor) -> Tensor:
        out = torch.einsum(self.expr, self.weight * self.alpha, t)
        if self.bias is not None:
            out = out + self.bias
        return out


class ElementRankLinear(nn.Module):
    def __init__(self, in_c: int, out_c: int, l: int, num_elements: int, bias: bool = False):
        super().__init__()
        self.l = l
        self.alpha = 1.0 / math.sqrt(max(in_c, 1))
        self.weight = nn.Parameter(torch.empty(num_elements, out_c, in_c))
        torch.nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
        if bias and l == 0:
            self.bias = nn.Parameter(torch.zeros(num_elements, out_c))
        else:
            self.register_parameter("bias", None)
        letters = [c for c in string.ascii_letters[3:] if c not in ("C", "z")]
        idx = "".join(letters[:l])
        self.expr = f"bz,zCc,bc{idx}->bC{idx}"

    def forward(self, t: Tensor, attrs: Tensor) -> Tensor:
        out = torch.einsum(self.expr, attrs, self.weight * self.alpha, t)
        if self.bias is not None:
            out = out + torch.einsum("bz,zC->bC", attrs, self.bias)
        return out


class SelfInteraction(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        ls: List[int],
        bias: bool = False,
        element_aware: bool = False,
        num_elements: int = 0,
    ):
        super().__init__()
        self.ls = list(ls)
        self.element_aware = element_aware
        mods = {}
        for l in self.ls:
            if element_aware:
                mods[str(l)] = ElementRankLinear(in_c, out_c, l, num_elements, bias=bias)
            else:
                mods[str(l)] = RankLinear(in_c, out_c, l, bias=bias)
        self.layers = nn.ModuleDict(mods)

    def forward(self, feats: Dict[int, Tensor], attrs: Optional[Tensor] = None) -> Dict[int, Tensor]:
        out: Dict[int, Tensor] = {}
        for l in self.ls:
            if l not in feats:
                continue
            layer = self.layers[str(l)]
            out[l] = layer(feats[l], attrs) if self.element_aware else layer(feats[l])
        return out


class DictSkipIdentity(nn.Module):
    """Cartesian analogue of ``SkipIdentity``: pass matching ranks, zero-pad missing ``l``."""

    def __init__(self, ls_out: List[int], num_channel: int):
        super().__init__()
        self.ls_out = list(ls_out)
        self.num_channel = num_channel
        self.element_aware = False

    def forward(
        self, feats: Dict[int, Tensor], attrs: Optional[Tensor] = None
    ) -> Dict[int, Tensor]:
        del attrs
        ref = next(iter(feats.values()))
        n, dtype, device = ref.shape[0], ref.dtype, ref.device
        out: Dict[int, Tensor] = {}
        for l in self.ls_out:
            if l in feats:
                out[l] = feats[l]
            else:
                shape = (n, self.num_channel) + ((3,) * l if l > 0 else ())
                out[l] = torch.zeros(shape, dtype=dtype, device=device)
        return out


def _satisfy(l1: int, l2: int, restriction: Optional[str]) -> bool:
    if restriction is None:
        return True
    return {
        "<": l1 < l2,
        "<=": l1 <= l2,
        ">": l1 > l2,
        ">=": l1 >= l2,
        "==": l1 == l2,
        "!=": l1 != l2,
    }[restriction]


def _generate_combs(lmax_in: int, lmax_out: int, l1l2: Optional[str] = None):
    combs = []
    for l1 in range(lmax_in + 1):
        for l2 in range(lmax_out + 1):
            for l3 in range(abs(l1 - l2), min(lmax_out, l1 + l2) + 1, 2):
                if _satisfy(l1, l2, l1l2):
                    k = (l1 + l2 - l3) // 2
                    combs.append((l1, l2, l3, k))
    return combs


class InterEinsum(nn.Module):
    """Channel-coupled reducible Cartesian TP + index contraction."""

    def __init__(self, comb: Tuple[int, int, int, int]):
        super().__init__()
        l1, l2, l3, k = comb
        self.comb = comb
        letters = list(string.ascii_lowercase)
        # avoid b,c reserved for batch/channel in the left pattern — use from 'd'
        free = [c for c in letters if c not in ("b", "c")]
        in1_idx = free[:l1]
        in2_idx = free[l1 : l1 + l2]
        for i in range(k):
            in2_idx[l2 - 1 - i] = in1_idx[l1 - 1 - i]
        out_idx = in1_idx[: l1 - k] + [in2_idx[i] for i in range(l2 - k)]
        assert len(out_idx) == l3
        self.expr = (
            f"bc{''.join(in1_idx)},bc{''.join(in2_idx)}->bc{''.join(out_idx)}"
        )
        self.normalizer = 1.0 / math.sqrt(3**k) if k > 0 else 1.0

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        return torch.einsum(self.expr, t1, t2) * self.normalizer


def _split_cartesian_harmonics(flat: Tensor, lmax: int) -> Dict[int, Tensor]:
    out: Dict[int, Tensor] = {}
    offset = 0
    B = flat.shape[0]
    for l in range(lmax + 1):
        dim = 3**l
        chunk = flat[:, offset : offset + dim]
        if l == 0:
            out[0] = chunk.reshape(B)  # [E]
        else:
            out[l] = chunk.reshape((B,) + (3,) * l)
        offset += dim
    return out


def _expand_edge_to_channels(t: Tensor, l: int, num_channel: int) -> Tensor:
    """[E] or [E, 3^l] -> [E, C, ...]"""
    if l == 0:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return t.unsqueeze(1).expand(-1, num_channel, 1).squeeze(-1)  # [E, C]
    return t.unsqueeze(1).expand(-1, num_channel, *t.shape[1:])


class CartesianContraction(nn.Module):
    def __init__(
        self,
        num_channel: int,
        lmax_in: int,
        lmax_out: int,
        l1l2: Optional[str] = None,
    ):
        super().__init__()
        self.num_channel = num_channel
        self.combs = _generate_combs(lmax_in, lmax_out, l1l2)
        if not self.combs:
            raise ValueError(f"No Cartesian TP paths for lmax_in={lmax_in}, lmax_out={lmax_out}")
        self.tcs = nn.ModuleList([InterEinsum(c) for c in self.combs])
        self.weight_numel = num_channel * len(self.combs)
        self.ws_slices = [
            slice(i * num_channel, (i + 1) * num_channel) for i in range(len(self.combs))
        ]
        counts: Dict[int, int] = defaultdict(int)
        for _, _, l3, _ in self.combs:
            counts[l3] += 1
        self.linear_downs = nn.ModuleDict(
            {str(l3): RankLinear(num_channel * cnt, num_channel, l3) for l3, cnt in counts.items()}
        )
        for l in range(lmax_out + 1):
            DS = ICTD(l, l)[1]
            self.register_buffer(f"D_{l}", DS[0].to(torch.get_default_dtype()))

    def D(self, l: int) -> Tensor:
        return getattr(self, f"D_{l}")

    def forward(
        self,
        node_feats: Dict[int, Tensor],
        edge_attrs: Dict[int, Tensor],
        weights: Tensor,
        edge_index: Tensor,
    ) -> Dict[int, Tensor]:
        sender, receiver = edge_index[0], edge_index[1]
        n_nodes = next(iter(node_feats.values())).shape[0]
        buckets: Dict[int, List[Tensor]] = defaultdict(list)

        for tc, comb, sl in zip(self.tcs, self.combs, self.ws_slices):
            l1, l2, l3, _ = comb
            if l1 not in node_feats or l2 not in edge_attrs:
                continue
            t1 = node_feats[l1][sender]
            t2 = _expand_edge_to_channels(edge_attrs[l2], l2, self.num_channel)
            w = weights[:, sl]
            if l1 == 0:
                t1w = t1 * w
            else:
                t1w = t1 * w.reshape(-1, self.num_channel, *([1] * l1))
            buckets[l3].append(tc(t1w, t2))

        out: Dict[int, Tensor] = {}
        for l3, parts in buckets.items():
            cat = torch.cat(parts, dim=1)
            flat = cat.reshape(cat.shape[0], -1)
            agg = scatter_sum(flat, receiver, dim=0, dim_size=n_nodes)
            rest = cat.shape[2:]
            agg = agg.reshape((n_nodes, cat.shape[1]) + rest)
            reduced = self.linear_downs[str(l3)](agg)
            if l3 == 0:
                out[l3] = reduced
            else:
                B, C = reduced.shape[:2]
                t = reduced.reshape(B, C, -1) @ self.D(l3).to(dtype=reduced.dtype)
                out[l3] = t.reshape((B, C) + (3,) * l3)
        return out


class CartesianProduct(nn.Module):
    def __init__(
        self,
        num_channel: int,
        lmax_in: int,
        ls_out: List[int],
        correlation: int,
        num_elements: int,
        bias: bool = True,
    ):
        super().__init__()
        self.correlation = correlation
        self.ls_out = list(ls_out)
        self.num_channel = num_channel
        for r in range(lmax_in + 1):
            DS = ICTD(r, r)[1]
            self.register_buffer(f"D_{r}", DS[0].to(torch.get_default_dtype()))

        self.coefs = nn.ModuleList(
            [
                SelfInteraction(
                    num_channel,
                    num_channel,
                    ls=self.ls_out,
                    bias=bias,
                    element_aware=True,
                    num_elements=num_elements,
                )
                for _ in range(correlation)
            ]
        )
        self.pair_ctrs = nn.ModuleList()
        self._pair_combs: List[List[Tuple[int, int, int, int]]] = []
        for _ in range(1, correlation):
            combs = [
                c
                for c in _generate_combs(lmax_in, max(self.ls_out), None)
                if c[2] in self.ls_out
            ]
            self._pair_combs.append(combs)
            self.pair_ctrs.append(nn.ModuleList([InterEinsum(c) for c in combs]))
        self.out_linear = SelfInteraction(
            num_channel, num_channel, ls=self.ls_out, bias=bias, element_aware=False
        )

    def D(self, l: int) -> Tensor:
        return getattr(self, f"D_{l}")

    def _contract_dict(self, left, right, ctrs, combs) -> Dict[int, Tensor]:
        buckets: Dict[int, List[Tensor]] = defaultdict(list)
        for tc, comb in zip(ctrs, combs):
            l1, l2, l3, _ = comb
            if l1 not in left or l2 not in right:
                continue
            buckets[l3].append(tc(left[l1], right[l2]))
        out: Dict[int, Tensor] = {}
        for l3, parts in buckets.items():
            t = parts[0]
            for p in parts[1:]:
                t = t + p
            if l3 == 0:
                out[l3] = t
            else:
                B, C = t.shape[:2]
                proj = t.reshape(B, C, -1) @ self.D(l3).to(dtype=t.dtype)
                out[l3] = proj.reshape((B, C) + (3,) * l3)
        return out

    def forward(
        self,
        node_feats: Dict[int, Tensor],
        node_attrs: Tensor,
        sc: Optional[Dict[int, Tensor]] = None,
    ) -> Dict[int, Tensor]:
        # Ensure all ls_out present for coef linears; missing -> zeros
        base = {}
        ref = next(iter(node_feats.values()))
        n = ref.shape[0]
        dtype, device = ref.dtype, ref.device
        for l in range(max(max(node_feats.keys()), max(self.ls_out)) + 1):
            if l in node_feats:
                base[l] = node_feats[l]
            elif l in self.ls_out or l <= max(node_feats.keys()):
                shape = (n, self.num_channel) + ((3,) * l if l > 0 else ())
                base[l] = torch.zeros(shape, dtype=dtype, device=device)

        outs = self.coefs[0]({l: base[l] for l in self.ls_out if l in base}, node_attrs)
        curr = base
        for nu in range(1, self.correlation):
            curr = self._contract_dict(
                curr, base, self.pair_ctrs[nu - 1], self._pair_combs[nu - 1]
            )
            term_in = {l: curr[l] for l in self.ls_out if l in curr}
            term = self.coefs[nu](term_in, node_attrs)
            outs = add_dict_to_left(outs, term)
        outs = self.out_linear(outs)
        if sc is not None:
            outs = add_dict_to_left(outs, {k: v for k, v in sc.items() if k in outs or k in self.ls_out})
            # ensure sc keys merged even if missing in outs
            for k, v in sc.items():
                if k in self.ls_out and k not in outs:
                    outs[k] = v
        return outs


_SCATTER_NORMS = ("avg_num_neighbors", "density", "no_cutoff_density")


def _broadcast_node_div(tensor: Tensor, density: Tensor) -> Tensor:
    """Divide per-node tensor features by ``density`` with rank-aware broadcast."""
    view = density.reshape(density.shape[0], *([1] * (tensor.ndim - 1)))
    return tensor / view


class CartesianLayerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_elements: int,
        num_channel: int,
        Lmax: int,
        lmax: int,
        correlation: List[int],
        avg_num_neighbors: float,
        edge_embedding_channel: int,
        edge_update: str,
        scatter_norm: str,
        radial_mlp: List[int],
        radial_bias: bool,
        use_first_resnet: bool,
        resnet_type: str,
        resnet_linear_type: str,
        l1l2: Optional[str],
        bias: bool,
    ):
        super().__init__()
        if scatter_norm not in _SCATTER_NORMS:
            raise ValueError(
                f"Unknown scatter_norm={scatter_norm!r}; expected one of {_SCATTER_NORMS}"
            )
        self.num_layers = num_layers
        self.num_channel = num_channel
        self.lmax = lmax
        self.scatter_norm = scatter_norm
        self.resnet_type = resnet_type
        self.apply_density_cutoff = scatter_norm != "no_cutoff_density"
        self.register_buffer(
            "_avg", torch.tensor(float(avg_num_neighbors), dtype=torch.get_default_dtype())
        )

        self.harmonics = CartesianHarmonics(
            list(range(lmax + 1)),
            normalize=True,
            normalization="component",
            norm=True,
            traceless=True,
        )

        self.edge_updates = nn.ModuleList()
        self.radial_nets = nn.ModuleList()
        self.linear_ups = nn.ModuleList()
        self.contractions = nn.ModuleList()
        self.products = nn.ModuleList()
        # Optional per-layer residuals: plain list + add_module (avoid None in ModuleList).
        self.resnets: List[Optional[nn.Module]] = []
        self.edge_densities = nn.ModuleList()
        self.density_alphas = nn.ParameterList()
        self.density_betas = nn.ParameterList()
        use_density = scatter_norm in ("density", "no_cutoff_density")

        for layer in range(num_layers):
            eu = EDGE_UPDATE[edge_update](
                num_elements=num_elements,
                num_channel=num_channel,
                edge_embedding_channel=edge_embedding_channel,
                bias=False,
            )
            self.edge_updates.append(eu)
            lmax_in = 0 if layer == 0 else Lmax
            ls_in = list(range(lmax_in + 1))
            ls_out = list(range(Lmax + 1)) if layer < num_layers - 1 else [0]

            self.linear_ups.append(
                SelfInteraction(num_channel, num_channel, ls=ls_in, bias=bias)
            )
            ctr = CartesianContraction(num_channel, lmax_in, lmax, l1l2=l1l2)
            self.contractions.append(ctr)
            self.radial_nets.append(
                RadialMLP(
                    [eu.out_dim] + list(radial_mlp) + [ctr.weight_numel],
                    use_layer_norm=False,
                    use_offset=False,
                    bias=radial_bias,
                )
            )
            if use_density:
                self.edge_densities.append(
                    RadialMLP(
                        [eu.out_dim, 64, 1],
                        use_layer_norm=False,
                        use_offset=False,
                        bias=radial_bias,
                    )
                )
                self.density_alphas.append(
                    nn.Parameter(torch.tensor(float(avg_num_neighbors)))
                )
                self.density_betas.append(nn.Parameter(torch.tensor(0.0)))
            # Match spherical CgtpInteraction: only BB residuals are wired.
            if (use_first_resnet or layer > 0) and resnet_type == "BB":
                if resnet_linear_type == "identity":
                    resnet = DictSkipIdentity(ls_out, num_channel)
                else:
                    resnet = SelfInteraction(
                        num_channel,
                        num_channel,
                        ls=ls_out,
                        bias=bias,
                        element_aware=(resnet_linear_type == "aware"),
                        num_elements=num_elements,
                    )
                self.add_module(f"resnet_{layer}", resnet)
                self.resnets.append(resnet)
            else:
                self.resnets.append(None)
            self.products.append(
                CartesianProduct(
                    num_channel=num_channel,
                    lmax_in=lmax,
                    ls_out=ls_out,
                    correlation=correlation[layer],
                    num_elements=num_elements,
                    bias=bias,
                )
            )

    def forward(
        self,
        node_feats: Tensor,
        node_attrs: Tensor,
        edge_emb: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        cutoff: Optional[Tensor] = None,
    ) -> Tensor:
        feats: Dict[int, Tensor] = {0: node_feats}
        edge_attrs = _split_cartesian_harmonics(self.harmonics(edge_vec), self.lmax)

        for layer in range(self.num_layers):
            edge_feats = self.edge_updates[layer](node_attrs, edge_emb, edge_index)
            ws = self.radial_nets[layer](edge_feats)
            if cutoff is not None:
                ws = ws * cutoff
            res = self.resnets[layer]
            sc = None
            if res is not None:
                sc = res(feats, node_attrs) if res.element_aware else res(feats)

            up = self.linear_ups[layer](feats)
            msg = self.contractions[layer](up, edge_attrs, ws, edge_index)
            if self.scatter_norm in ("density", "no_cutoff_density"):
                density = torch.tanh(self.edge_densities[layer](edge_feats) ** 2)
                if cutoff is not None and self.apply_density_cutoff:
                    density = density * cutoff
                density = scatter_sum(
                    density, edge_index[1], dim=0, dim_size=node_attrs.size(0)
                )
                density = density * self.density_betas[layer] + self.density_alphas[layer]
                density = density.masked_fill(density == 0, 1e-9)
                msg = {l: _broadcast_node_div(t, density) for l, t in msg.items()}
            elif self.scatter_norm == "avg_num_neighbors":
                msg = {l: t / self._avg for l, t in msg.items()}
            feats = self.products[layer](msg, node_attrs, sc=sc)

        return dict_to_scalar(feats)
