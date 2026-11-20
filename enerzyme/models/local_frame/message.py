"""Equivariant updates for LEFTNet and AlphaNet.

Both models split a directionally gated message into a scalar residual and
a vector residual

``Δv = v_j ⊙ g2 / √3 + r̂_ij ⊗ g3``,

then scale by ``1/√F``. LEFTNet stops there. AlphaNet runs the same split
and, on the scalar channel only, appends a matrix-product phase that the
next layer uses as a rotary embedding.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .encoding import init_linears
from .geometry import reduce_nodes, scalarize


def apply_rope(features: Tensor, rope: Tensor | None) -> Tensor:
    """Multiply the complex view of ``features`` by a per-node phase."""
    if rope is None:
        return features
    half = features.shape[-1] // 2
    real, imag = features.split(half, dim=-1)
    rotated = torch.complex(real, imag) * rope
    return torch.cat((rotated.real, rotated.imag), dim=-1)


class DirectionalEmbedding(nn.Module):
    """Node, radial, and structure projections that gate each edge."""

    def __init__(self, hidden: int, num_rbf: int, layernorm: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden) if layernorm else nn.Identity()
        self.x_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 3),
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden * 3)
        self.dir_proj = nn.Sequential(
            nn.Linear(3 * hidden + num_rbf, hidden * 3),
            nn.SiLU(),
            nn.Linear(hidden * 3, hidden * 3),
        )
        self.hidden = hidden
        init_linears(self)

    def forward(self, features: Tensor, rbf: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.x_proj(self.norm(features))
        gated = self.rbf_proj(rbf) * self.dir_proj(weight)
        return hidden, gated


class CartesianMessage(nn.Module):
    """LEFTNet equivariant message. Both residuals are summed on the receiver."""

    def __init__(self, hidden: int, num_rbf: int) -> None:
        super().__init__()
        self.embed = DirectionalEmbedding(hidden, num_rbf, layernorm=False)
        self.hidden = hidden
        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden)

    def forward(
        self,
        features: Tensor,
        vectors: Tensor,
        rbf: Tensor,
        weight: Tensor,
        edge_unit: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden, gated = self.embed(features, rbf, weight)
        scalar, vec_gate, radial_gate = (hidden[idx_j] * gated).split(self.hidden, dim=-1)
        vec_gate = vec_gate * self.inv_sqrt_3
        message = vectors[idx_j] * vec_gate.unsqueeze(1)
        message = message + radial_gate.unsqueeze(1) * edge_unit.unsqueeze(-1)
        message = message * self.inv_sqrt_h
        n_atoms = features.shape[0]
        return (
            reduce_nodes(scalar, idx_i, n_atoms),
            reduce_nodes(message, idx_i, n_atoms),
        )


class AlphaNetMessage(nn.Module):
    """AlphaNet message: the Cartesian update plus an MPS phase (RoPE).

    The returned phase is ``exp(i θ)`` with ``θ`` in ``R^{F/2}``. The next
    layer multiplies it onto the scalar state. The scalar residual is the
    ungated half of the aggregated message, not that phase.
    """

    def __init__(
        self,
        hidden: int,
        num_rbf: int,
        hidden_chi: int,
        head: int,
        chi1: int,
        chi2: int,
        dropout: bool,
        norm_before: bool,
        norm_after: bool,
        reduce: str,
    ) -> None:
        super().__init__()
        self.embed = DirectionalEmbedding(hidden, num_rbf, layernorm=True)
        self.hidden = hidden
        self.hidden_chi = hidden_chi
        self.head = head
        self.chi1 = chi1
        self.chi2 = chi2
        self.reduce = reduce
        self.inv_sqrt_3 = 1.0 / math.sqrt(3.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden)
        self.inv_sqrt_kernel = 1.0 / math.sqrt(hidden // head)
        self.scale = nn.Linear(hidden, hidden_chi * 2)
        self.diagonal = nn.Sequential(
            nn.Linear(hidden * 3, hidden_chi // 2),
            nn.SiLU(),
            nn.Linear(hidden_chi // 2, chi2),
        )
        self.diachi1 = nn.Parameter(torch.randn(chi1))
        self.scale2 = nn.Linear(chi1, hidden // 2)
        width = hidden_chi // head
        self.kernel_real = nn.Parameter(torch.randn(head + 1, width, chi2))
        self.kernel_imag = nn.Parameter(torch.randn(head + 1, width, chi2))
        self.fc_mps = nn.Linear(chi1, chi1)
        self.dia = nn.Linear(chi1, chi1)
        self.dropout = nn.Dropout(p=0.5) if dropout else nn.Identity()
        self.activation = nn.SiLU()
        self.norm_before = nn.LayerNorm(chi1 + hidden) if norm_before else None
        self.norm_after = nn.LayerNorm(chi1) if norm_after else None
        init_linears(self)

    def _phase(self, scalar: Tensor, gated: Tensor) -> Tensor:
        real, imag = self.scale(scalar).split(self.hidden_chi, dim=-1)
        width = self.hidden_chi // self.head
        real = self.dropout(real.reshape(-1, self.head, width))
        imag = self.dropout(imag.reshape(-1, self.head, width))
        query = torch.complex(real, imag)
        ones = query.new_ones(query.shape[0], 1, width)
        kernel = torch.complex(self.kernel_real, self.kernel_imag) * self.inv_sqrt_kernel
        kernel = kernel.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
        conv = torch.einsum("ijl,ijlk->ik", torch.cat((ones, query), dim=1), kernel)
        diag_gate = self.activation(self.diagonal(gated))
        raw = diag_gate.unsqueeze(-1) * self.diachi1.view(1, 1, -1)
        raw = raw + torch.ones(
            raw.shape[0],
            self.chi2,
            self.chi1,
            device=raw.device,
            dtype=raw.dtype,
        )
        mixed = torch.einsum("ik,ikl->il", conv, self.dia(raw).to(query.dtype))
        return torch.angle(torch.complex(self.fc_mps(mixed.real), self.fc_mps(mixed.imag)))

    def forward(
        self,
        features: Tensor,
        vectors: Tensor,
        rbf: Tensor,
        weight: Tensor,
        edge_unit: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        rope: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = apply_rope(features, rope)
        hidden, gated = self.embed(features, rbf, weight)
        scalar, vec_gate, radial_gate = (hidden[idx_j] * gated).split(self.hidden, dim=-1)
        vec_gate = vec_gate * self.inv_sqrt_3
        message = vectors[idx_j] * vec_gate.unsqueeze(1)
        message = message + radial_gate.unsqueeze(1) * edge_unit.unsqueeze(-1)
        message = message * self.inv_sqrt_h
        phase = self._phase(scalar, gated)
        aggregated = reduce_nodes(
            torch.cat((phase, scalar), dim=-1), idx_i, features.shape[0], self.reduce
        )
        if self.norm_before is not None:
            aggregated = self.norm_before(aggregated)
        phase, residual = aggregated.split((self.chi1, self.hidden), dim=-1)
        if self.norm_after is not None:
            phase = self.norm_after(phase)
        phase = self.scale2(phase)
        rope_out = torch.complex(torch.cos(phase), torch.sin(phase))
        dvec = reduce_nodes(message, idx_i, features.shape[0])
        return rope_out, residual, dvec


class FrameTransition(nn.Module):
    """Node update after the edge messages.

    ``l2`` is the released LEFTNet reduction (Euclidean norm of one vector
    stream). The node-frame projection in that file is computed and then
    discarded. ``frame`` is the paper update: scalarize on the node frame,
    map the three axes with a bias-free linear, then tensorize.
    ``l1`` is the released AlphaNet reduction and does not use a node frame.
    """

    def __init__(self, hidden: int, scalarization: str) -> None:
        super().__init__()
        if scalarization not in {"l1", "l2", "frame"}:
            raise ValueError(
                f"scalarization must be 'l1', 'l2', or 'frame', got {scalarization!r}"
            )
        self.scalarization = scalarization
        self.hidden = hidden
        self.vec_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.mix = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden * 3),
        )
        self.frame_reduce = (
            nn.Linear(3, 1, bias=False) if scalarization == "frame" else None
        )
        self.inv_sqrt_2 = 1.0 / math.sqrt(2.0)
        self.inv_sqrt_h = 1.0 / math.sqrt(hidden)
        self.eps = 1e-10
        init_linears(self)

    def _scalar(self, vec1: Tensor, node_frame: Tensor | None) -> Tensor:
        if self.scalarization == "l2":
            squared = torch.sum(vec1 * vec1, dim=-2)
            return torch.sqrt(squared + self.eps)
        if self.scalarization == "l1":
            return vec1.abs().sum(dim=-2)
        if node_frame is None:
            raise ValueError("frame scalarization requires a node frame")
        projected = scalarize(vec1, node_frame)
        reduced = self.frame_reduce(projected.transpose(-1, -2))
        return reduced.squeeze(-1)

    def forward(
        self,
        features: Tensor,
        vectors: Tensor,
        node_frame: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected = self.vec_proj(vectors)
        vec1, vec2 = projected.split(self.hidden, dim=-1)
        scalar = self._scalar(vec1, node_frame)
        vec_dot = (vec1 * vec2).sum(dim=-2) * self.inv_sqrt_h
        mixed = self.mix(torch.cat((features, scalar), dim=-1))
        update_s, update_v, scale = mixed.split(self.hidden, dim=-1)
        delta_s = (update_s + update_v + vec_dot) * self.inv_sqrt_2
        delta_v = scale.unsqueeze(1) * vec2
        return delta_s, delta_v


def init_quantum(embedding: Tensor, kernel: Tensor) -> Tensor:
    """Map the species embedding to the complex matrix-product state."""
    mixed = embedding.matmul(kernel)
    real, imag = mixed.split(kernel.shape[-1] // 2, dim=-1)
    return torch.complex(real, imag)


def contract_quantum(
    scalar: Tensor,
    quantum: Tensor,
    kernel_real: Tensor,
    kernel_imag: Tensor,
    eps: float,
) -> Tensor:
    """One temporal contraction, then project each component onto the unit circle."""
    kernel = torch.complex(kernel_real, kernel_imag)
    updated = torch.einsum("ikl,bi,bl->bk", kernel, scalar.to(dtype=kernel.dtype), quantum)
    return updated / updated.abs().clamp_min(eps)
