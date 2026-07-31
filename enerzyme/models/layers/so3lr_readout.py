"""Partial-charge and Hirshfeld readout heads for SO3LR-style stacks."""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
from torch import Tensor
from torch.nn import Embedding, Identity, Linear, Module, Sequential

from . import BaseFFLayer


class PartialChargeReadout(BaseFFLayer):
    """Element-biased partial-charge head (So3krates-torch ``PartialChargesOutputHead``).

    Emits raw ``Qa``; pair with :class:`ChargeConservationLayer` for neutrality.
    """

    def __init__(
        self,
        dim_embedding: Optional[int] = None,
        built_layers: Optional[List[Module]] = None,
        regression_dim: Optional[int] = None,
        max_Za: int = 100,
        activation_fn: Optional[Callable[[], Module]] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if dim_embedding is None and built_layers:
            for layer in reversed(built_layers):
                if hasattr(layer, "dim_feature_out"):
                    dim_embedding = int(layer.dim_feature_out)
                    break
                if hasattr(layer, "dim_embedding"):
                    dim_embedding = int(layer.dim_embedding)
                    break
        if dim_embedding is None:
            raise TypeError("dim_embedding value should be provided")
        super().__init__(
            input_fields={"atom_feature", "Za"},
            output_fields={"Qa"},
        )
        self.atomic_embedding = Embedding(max_Za + 1, 1)
        act = Identity if activation_fn is None else activation_fn
        if regression_dim is not None:
            self.transform = Sequential(
                Linear(dim_embedding, regression_dim),
                act(),
                Linear(regression_dim, 1),
            )
        else:
            self.transform = Linear(dim_embedding, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.atomic_embedding.embedding_dim ** 0.5)
        torch.nn.init.normal_(self.atomic_embedding.weight, mean=0.0, std=std)
        modules = (
            self.transform
            if isinstance(self.transform, Sequential)
            else [self.transform]
        )
        for m in modules:
            if isinstance(m, Linear):
                std_m = 1.0 / (m.in_features ** 0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std_m)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_Qa(self, atom_feature: Tensor, Za: Tensor) -> Tensor:
        q_bias = self.atomic_embedding(Za.long()).squeeze(-1)
        x = self.transform(atom_feature).squeeze(-1)
        return x + q_bias


class HirshfeldReadout(BaseFFLayer):
    """Hirshfeld volume-ratio head (So3krates-torch ``HirshfeldOutputHead``).

    Outputs ``ha = |v_shift + (q ⊙ k) / √d|``.
    """

    def __init__(
        self,
        dim_embedding: Optional[int] = None,
        built_layers: Optional[List[Module]] = None,
        regression_dim: Optional[int] = None,
        max_Za: int = 100,
        activation_fn: Optional[Callable[[], Module]] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if dim_embedding is None and built_layers:
            for layer in reversed(built_layers):
                if hasattr(layer, "dim_feature_out"):
                    dim_embedding = int(layer.dim_feature_out)
                    break
                if hasattr(layer, "dim_embedding"):
                    dim_embedding = int(layer.dim_embedding)
                    break
        if dim_embedding is None:
            raise TypeError("dim_embedding value should be provided")
        if dim_embedding % 2 != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be even for HirshfeldReadout"
            )
        super().__init__(
            input_fields={"atom_feature", "Za"},
            output_fields={"ha"},
        )
        half = dim_embedding // 2
        self.v_shift_embedding = Embedding(max_Za + 1, 1)
        self.q_embedding = Embedding(max_Za + 1, half)
        act = Identity if activation_fn is None else activation_fn
        if regression_dim is not None:
            self.transform = Sequential(
                Linear(dim_embedding, regression_dim // 2),
                act(),
                Linear(regression_dim // 2, half),
            )
        else:
            self.transform = Linear(dim_embedding, half)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in (self.v_shift_embedding, self.q_embedding):
            std = 1.0 / (emb.embedding_dim ** 0.5)
            torch.nn.init.normal_(emb.weight, mean=0.0, std=std)
        modules = (
            self.transform
            if isinstance(self.transform, Sequential)
            else [self.transform]
        )
        for m in modules:
            if isinstance(m, Linear):
                std_m = 1.0 / (m.in_features ** 0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std_m)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_ha(self, atom_feature: Tensor, Za: Tensor) -> Tensor:
        v_shift = self.v_shift_embedding(Za.long()).squeeze(-1)
        q = self.q_embedding(Za.long())
        k = self.transform(atom_feature)
        qk = (q * k * (1.0 / math.sqrt(k.shape[-1]))).sum(dim=-1)
        return torch.abs(v_shift + qk)
