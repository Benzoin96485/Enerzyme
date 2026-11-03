"""Cartesian gated equivariant blocks (Schütt et al., ICML 2021).

Shared by PaiNN's tensor heads and any later Core that emits channel-last
Cartesian vector features ``[N, 3, F]``. Scalar nonlinearities are exact SiLU
(``F.silu``). The Enerzyme ``swish`` activation defaults to ``beta=1.702``;
these helpers pin ``alpha=beta=1`` so the simple path is SiLU.

Adapted from SchNetPack 2.0.4 ``schnetpack.nn.equivariant`` and
``schnetpack.nn.blocks.build_gated_equivariant_mlp`` (MIT).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Sequential, SiLU

from ..blocks.mlp import DenseLayer


# Exact SiLU via BaseScaledTemperedActivation's parameter-free branch.
_SILU_PARAMS = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}


def cartesian_dense(
    n_in: int,
    n_out: int,
    *,
    bias: bool = True,
    activation: bool = False,
) -> DenseLayer:
    """Xavier dense map over the last dimension, optional exact SiLU."""
    return DenseLayer(
        dim_feature_in=n_in,
        dim_feature_out=n_out,
        activation_fn="swish" if activation else None,
        activation_params=dict(_SILU_PARAMS) if activation else {},
        initial_weight="xavier_uniform",
        initial_bias="zero",
        use_bias=bias,
    )


class GatedEquivariantBlock(Module):
    """One gated map ``(s, v) -> (s', v')`` with ``v' = gate(s, ||V v||) * W v``.

    ``s`` is ``[N, F_s]`` and ``v`` is ``[N, 3, F_v]``. Channel counts may
    change. The vector mix is bias-free, so a zero vector stays zero.
    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        sactivation: bool = False,
    ) -> None:
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = cartesian_dense(n_vin, 2 * n_vout, bias=False, activation=False)
        self.scalar_net = Sequential(
            cartesian_dense(n_sin + n_vout, n_hidden, activation=True),
            cartesian_dense(n_hidden, n_sout + n_vout, activation=False),
        )
        self.sactivation = SiLU() if sactivation else None

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        scalars, vectors = inputs
        mixed = self.mix_vectors(vectors)
        vectors_v, vectors_w = torch.split(mixed, self.n_vout, dim=-1)
        vectors_vn = torch.linalg.vector_norm(vectors_v, dim=-2)
        ctx = torch.cat([scalars, vectors_vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, gate = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = gate.unsqueeze(-2) * vectors_w
        if self.sactivation is not None:
            s_out = self.sactivation(s_out)
        return s_out, v_out


def _gated_widths(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]],
    n_layers: int,
) -> List[int]:
    """SchNetPack neuron schedule, including the output width."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if n_hidden is None:
        width = n_in
        widths: List[int] = []
        for _ in range(n_layers):
            widths.append(width)
            width = max(n_out, width // 2)
        widths.append(n_out)
        return widths
    if isinstance(n_hidden, int):
        hidden = [n_hidden] * (n_layers - 1)
    else:
        hidden = [int(h) for h in n_hidden]
    if len(hidden) != n_layers - 1:
        raise ValueError(
            f"n_hidden has length {len(hidden)}; expected n_layers - 1 = {n_layers - 1}"
        )
    return [n_in, *hidden, n_out]


def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
) -> Sequential:
    """Stack of gated blocks. Hidden scalar outputs use SiLU; the last does not.

    With ``n_hidden=None`` and ``n_layers=2`` the widths are
    ``[F, max(n_out, F // 2), n_out]``, matching SchNetPack.
    """
    widths = _gated_widths(n_in, n_out, n_hidden, n_layers)
    gating = widths[:-1]
    blocks = [
        GatedEquivariantBlock(
            n_sin=widths[i],
            n_vin=widths[i],
            n_sout=widths[i + 1],
            n_vout=widths[i + 1],
            n_hidden=gating[i],
            sactivation=i < n_layers - 1,
        )
        for i in range(n_layers)
    ]
    return Sequential(*blocks)
