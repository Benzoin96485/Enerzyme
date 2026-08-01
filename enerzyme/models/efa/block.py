"""EFA plug-in block with residual MLP (identity-at-init last layer)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.nn import Linear, Module, SiLU

from .attention import EuclideanFastAttention


class EFABlock(Module):
    """Drop-in nonlocal block: Dense/EFA path + skip + SiLU MLP.

    ``forward(features, positions, batch_seg) -> [N, dim_features]`` additive
    update (last Dense zero-initialized so the block behaves like identity at
    init when used as ``x + EFABlock(x, ...)`` or as a SpookyNet nonlocal delta).

    When ``as_delta=True`` (default), returns the update to **add** to ``features``.
    When ``as_delta=False``, returns refined features ``features + update``.
    """

    def __init__(
        self,
        dim_features: int,
        *,
        num_features_qk: Optional[int] = None,
        num_features_v: Optional[int] = None,
        lebedev_num: int = 146,
        max_frequency: Optional[float] = None,
        max_length: float = 10.0,
        post_mlp: bool = True,
        as_delta: bool = True,
        zero_init_last: bool = True,
    ) -> None:
        super().__init__()
        self.dim_features = int(dim_features)
        self.post_mlp = bool(post_mlp)
        self.as_delta = bool(as_delta)

        self.efa = EuclideanFastAttention(
            dim_features,
            num_features_qk=num_features_qk,
            num_features_v=num_features_v,
            lebedev_num=lebedev_num,
            max_frequency=max_frequency,
            max_length=max_length,
            parametrized=True,
        )
        out_v = self.efa.num_features_v
        self.project_in = Linear(out_v, self.dim_features, bias=True)
        self.act = SiLU()
        if self.post_mlp:
            self.mlp_1 = Linear(self.dim_features, self.dim_features, bias=True)
            self.mlp_2 = Linear(self.dim_features, self.dim_features, bias=True)
        else:
            self.mlp_1 = None
            self.mlp_2 = Linear(self.dim_features, self.dim_features, bias=True)

        if zero_init_last:
            torch.nn.init.zeros_(self.mlp_2.weight)
            if self.mlp_2.bias is not None:
                torch.nn.init.zeros_(self.mlp_2.bias)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        batch_seg: Tensor,
    ) -> Tensor:
        y = self.efa(features, positions, batch_seg)
        y = self.project_in(y) + features  # skip around EFA
        if self.post_mlp and self.mlp_1 is not None:
            y = self.mlp_2(self.act(self.mlp_1(y)))
        else:
            y = self.mlp_2(y)
        # y is residual update (~0 at init). Full features = features + y only
        # if project_in path already mixed; EnergyModel applies y_nl as additive
        # branch. Here ``y`` after MLP is the additive nonlocal contribution
        # relative to the skip-augmented path:
        #   skip: project_in(efa)+features, then MLP with zero last -> ~0
        # so returning ``y`` as delta matches SpookyNet nonlocal.
        if self.as_delta:
            return y
        return features + y


def parse_era_iterations(spec: Optional[object]) -> Optional[Sequence[int]]:
    """Parse ``era_use_in_iterations`` from list/tuple/str/None.

    ``None`` / ``""`` / ``[]`` → disable EFA. ``\"0 1\"`` / ``[0, 1]`` → those layers.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip()
        if not spec:
            return None
        return [int(x) for x in spec.replace(",", " ").split()]
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return None
        return [int(x) for x in spec]
    raise TypeError(f"Unsupported era_use_in_iterations type: {type(spec)}")
