"""Materials Graphormer pieces used to extend M3GNet.

Yang et al., arXiv:2405.04967, supplementary section S1.3. The structural
encoder adds a centrality vector to the species embedding, then a cutoff
graph attention whose logits include a Gaussian distance bias. The smooth
mask multiplies the attention weights after the softmax, as written in
eq. (4). Periodic images are not extra nodes: ``idx_j`` already addresses
the atom inside the cell, so key and value are shared with that image.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Dropout, GELU, LayerNorm, Linear, Module, Sequential

from ..mp.basis.envelope import polynomial_envelope
from ..mp.aggregate import SoftmaxAttention
from ..mp.message.attention import dot_product_attention_logits
from ..mp.ops import scatter_sum
from ..mp.ops import segment_softmax as receiver_softmax


def segment_softmax(logits: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """MatterSim receiver weights with its original denominator floor."""
    return receiver_softmax(logits, index, dim_size, clamp_min=1e-12)


class CentralityEncoding(Module):
    """``b'_i = Σ_j Linear(m_ij Φ_ij)`` added to the species embedding.

    The linear map sits inside the sum, so its bias is added once per
    neighbor. An atom with an empty neighborhood gets a zero update.
    """

    def __init__(self, num_basis: int, dim: int) -> None:
        super().__init__()
        self.linear = Linear(int(num_basis), int(dim), bias=True)

    def forward(
        self,
        basis: Tensor,
        dist: Tensor,
        idx_i: Tensor,
        n_atoms: int,
        cutoff: float,
    ) -> Tensor:
        mask = polynomial_envelope(dist, cutoff, 3)
        if idx_i.numel() == 0:
            return basis.new_zeros(n_atoms, self.linear.out_features)
        mapped = self.linear(basis * mask.unsqueeze(-1))
        return scatter_sum(mapped, idx_i, n_atoms)


class GraphormerBlock(Module):
    """One structural-encoder block: masked attention, then a pointwise FFN.

    Both branches are residual and post-normalized. ``basis`` is the Gaussian
    expansion of each edge. ``mask`` is the polynomial cutoff on that edge.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_basis: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError(
                f"dim {dim} must be divisible by num_heads {num_heads}"
            )
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.aggregate = SoftmaxAttention(clamp_min=1e-12)
        self.qkv = Linear(self.dim, 3 * self.dim, bias=True)
        self.distance_bias = Linear(int(num_basis), self.num_heads, bias=False)
        # No bias: a zero aggregate (every mask is zero, or there is no edge)
        # must stay zero. An output bias would shift every atom whenever the
        # neighbor list is non-empty.
        self.out = Linear(self.dim, self.dim, bias=False)
        self.norm_attn = LayerNorm(self.dim)
        self.ffn = Sequential(
            Linear(self.dim, int(ffn_dim)),
            GELU(),
            Linear(int(ffn_dim), self.dim),
        )
        self.norm_ffn = LayerNorm(self.dim)
        self.dropout = Dropout(float(dropout)) if float(dropout) > 0 else None

    def attend(
        self,
        node: Tensor,
        basis: Tensor,
        mask: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        """Attention update before the residual connection. ``(N, F)``."""
        n_atoms = int(node.shape[0])
        if idx_i.numel() == 0:
            return node.new_zeros(n_atoms, self.dim)
        qkv = self.qkv(node).view(n_atoms, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        logits = dot_product_attention_logits(query[idx_i], key[idx_j], scale=self.scale)
        logits = logits + self.distance_bias(basis)
        message = self.aggregate(value[idx_j], logits, idx_i, n_atoms, mask.unsqueeze(-1))
        return self.out(message.reshape(n_atoms, self.dim))

    def forward(
        self,
        node: Tensor,
        basis: Tensor,
        mask: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        update = self.attend(node, basis, mask, idx_i, idx_j)
        if self.dropout is not None:
            update = self.dropout(update)
        node = self.norm_attn(node + update)
        hidden = self.ffn(node)
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        return self.norm_ffn(node + hidden)


class GraphormerMessage(GraphormerBlock):
    def forward(self, state, ctx):
        node = super().forward(
            state.scalar, ctx.inputs["graphormer_rbf"], state.aux["attention_mask"], ctx.receiver, ctx.sender
        )
        return state._replace(scalar=node)
