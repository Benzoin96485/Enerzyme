"""DimeNet embedding, residual, interaction, and atom-side RBF readout.

Directional message passing (Eq. 4 / Fig. 4) lives here. Property MLPs stay
in shared :class:`~enerzyme.models.layers.readout.HierachicalReadout`.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE
from ..blocks.mlp import DenseLayer
from ..functional import segment_sum_coo


_SWISH_PARAMS: ACTIVATION_PARAM_TYPE = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}


def _act_params(
    activation_params: Optional[ACTIVATION_PARAM_TYPE],
) -> ACTIVATION_PARAM_TYPE:
    return dict(_SWISH_PARAMS if not activation_params else activation_params)


class DimeNetResidualLayer(Module):
    """Official residual: ``x + Dense_act(Dense_act(x))`` (Glorot-orthogonal)."""

    def __init__(
        self,
        dim_embedding: int,
        activation_fn: ACTIVATION_KEY_TYPE = "swish",
        activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
    ) -> None:
        super().__init__()
        params = _act_params(activation_params)
        self.dense1 = DenseLayer(
            dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )
        self.dense2 = DenseLayer(
            dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dense2(self.dense1(x))


class DimeNetEmbeddingBlock(Module):
    """Initial directed messages from pre-core atom embeddings and RBF.

    Official concat is ``[h_i || h_j || Dense(rbf)]`` (receiver, sender), which
    differs from the paper's ``[h_j || h_i || e]`` typesetting.
    """

    def __init__(
        self,
        dim_embedding: int,
        num_rbf: int,
        activation_fn: ACTIVATION_KEY_TYPE = "swish",
        activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
    ) -> None:
        super().__init__()
        params = _act_params(activation_params)
        self.dense_rbf = DenseLayer(
            num_rbf,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )
        self.dense = DenseLayer(
            3 * dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )

    def forward(
        self,
        atom_embedding: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        rbf_h = self.dense_rbf(rbf)
        return self.dense(
            torch.cat(
                [atom_embedding[idx_i], atom_embedding[idx_j], rbf_h], dim=-1
            )
        )


class DimeNetInteractionBlock(Module):
    """Original DimeNet interaction (bilinear SBF, not DimeNet++ Hadamard)."""

    def __init__(
        self,
        dim_embedding: int,
        num_bilinear: int,
        num_sbf: int,
        num_rbf: int,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        activation_fn: ACTIVATION_KEY_TYPE = "swish",
        activation_params: Optional[ACTIVATION_PARAM_TYPE] = None,
    ) -> None:
        super().__init__()
        params = _act_params(activation_params)
        self.dense_rbf = DenseLayer(
            num_rbf,
            dim_embedding,
            use_bias=False,
            initial_weight="semi_orthogonal_glorot",
        )
        self.dense_sbf = DenseLayer(
            num_sbf,
            num_bilinear,
            use_bias=False,
            initial_weight="semi_orthogonal_glorot",
        )
        self.dense_ji = DenseLayer(
            dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )
        self.dense_kj = DenseLayer(
            dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )
        self.W = Parameter(torch.empty(dim_embedding, num_bilinear, dim_embedding))
        torch.nn.init.normal_(self.W, mean=0.0, std=2.0 / dim_embedding)
        self.layers_before_skip = ModuleList(
            [
                DimeNetResidualLayer(dim_embedding, activation_fn, params)
                for _ in range(num_before_skip)
            ]
        )
        self.final_before_skip = DenseLayer(
            dim_embedding,
            dim_embedding,
            activation_fn=activation_fn,
            activation_params=params,
            initial_weight="semi_orthogonal_glorot",
        )
        self.layers_after_skip = ModuleList(
            [
                DimeNetResidualLayer(dim_embedding, activation_fn, params)
                for _ in range(num_after_skip)
            ]
        )

    def forward(
        self,
        x: Tensor,
        rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> Tensor:
        rbf_h = self.dense_rbf(rbf)
        sbf_h = self.dense_sbf(sbf)
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x) * rbf_h
        if idx_kj.numel() == 0:
            x_kj_sum = x.new_zeros(x.shape)
        else:
            x_kj = torch.einsum("wj,wl,ijl->wi", sbf_h, x_kj[idx_kj], self.W)
            x_kj_sum = segment_sum_coo(x_kj, idx_ji, dim_size=x.shape[0])
        hidden = x_ji + x_kj_sum
        for layer in self.layers_before_skip:
            hidden = layer(hidden)
        hidden = self.final_before_skip(hidden) + x
        for layer in self.layers_after_skip:
            hidden = layer(hidden)
        return hidden


class DimeNetAtomReadout(Module):
    """RBF-gated scatter of messages onto atoms (output-block first half)."""

    def __init__(self, num_rbf: int, dim_embedding: int) -> None:
        super().__init__()
        self.dense_rbf = DenseLayer(
            num_rbf,
            dim_embedding,
            use_bias=False,
            initial_weight="semi_orthogonal_glorot",
        )

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        idx_i: Tensor,
        num_nodes: int,
    ) -> Tensor:
        gated = self.dense_rbf(rbf) * messages
        return segment_sum_coo(gated, idx_i, dim_size=num_nodes)
