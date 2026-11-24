"""Local-frame embedding and message blocks shared by LEFTNet and AlphaNet."""

import torch
from torch import nn

from ...basis.envelope import cosine_transition
from ...block import Block, BlockState
from ...core import Embed
from ...head import Head
from .encoding import LocalSubstructure, RadialFilter, aggregate_neighbors
from .geometry import (alphanet_edge_frame, leftnet_edge_frame, leftnet_node_frame,
                       require_matching_distance, require_same_graph, subtract_graph_mean)
from .message import contract_quantum, init_quantum


class FrameEmbed(Embed):
    input_fields = ("atom_embedding", "Ra", "batch_seg")

    def __init__(self, width, num_rbf, cutoff, scalarization, eps, quantum=False):
        super().__init__()
        self.width = width
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.scalarization = scalarization
        self.eps = eps
        self.quantum = quantum
        self.embed_norm = nn.LayerNorm(width, elementwise_affine=False) if quantum else None
        self.radial = RadialFilter(num_rbf, width)
        self.lse = LocalSubstructure(width, normalize=quantum, scale=quantum,
                                     endpoint="target_source" if quantum else "source_target")

    def forward(self, inputs, ctx):
        x, pos = inputs["atom_embedding"], inputs["Ra"]
        if x.shape[-1] != self.width:
            raise ValueError(f"atom_embedding width {x.shape[-1]} != dim_embedding {self.width}")
        if ctx.rbf.shape[-1] != self.num_rbf:
            raise ValueError(f"rbf width {ctx.rbf.shape[-1]} != num_rbf {self.num_rbf}")
        i, j = ctx.receiver, ctx.sender
        if i.shape != j.shape or i.ndim != 1:
            raise ValueError("idx_i_sr and idx_j_sr must be 1D and the same length")
        if ctx.vec.shape[0] != ctx.num_edges or ctx.dist.numel() != ctx.num_edges:
            raise ValueError("vij_sr, Dij_sr, and the neighbor index must have one row per edge")
        if pos.shape != (ctx.num_nodes, 3):
            raise ValueError("Ra must have shape [N, 3] with the same N as atom_embedding")
        require_same_graph(ctx.batch, i, j)
        require_matching_distance(ctx.vec, ctx.dist)
        window = (cosine_transition(ctx.dist.reshape(-1), self.cutoff) if ctx.cutoff_values is None
                  else ctx.cutoff_values.reshape(-1))
        species = self.embed_norm(x) if self.quantum else x
        radial = self.radial(ctx.rbf, window)
        scalar = aggregate_neighbors(species, radial, i, j)
        vector = scalar.new_zeros(ctx.num_nodes, 3, self.width)
        node_frame = weight = direction = None
        if self.quantum:
            ctx.cache["quantum_kernels"] = self.kernels_real, self.kernels_imag
            quantum = init_quantum(species, self.kernel1)
            if ctx.num_edges:
                frame, direction = alphanet_edge_frame(pos, ctx.vec, ctx.dist, i, j, self.eps)
        else:
            quantum = None
            if ctx.num_edges or self.scalarization == "frame":
                centered = subtract_graph_mean(pos, ctx.batch)
                if ctx.num_edges:
                    frame, direction = leftnet_edge_frame(centered, ctx.vec, i, j, self.eps)
                if self.scalarization == "frame":
                    node_frame = leftnet_node_frame(centered, i, j, self.eps)
        if ctx.num_edges:
            weight = self.lse(scalar, direction, frame, radial, ctx.rbf, window, i, j)
        ctx.cache["local_frames"] = weight, direction, node_frame
        return BlockState(scalar, vector, aux={"quantum": quantum, "rope": None} if self.quantum else None)


class FrameBlock(Block):
    def __init__(self, message, transition, quantum_layer=None, eps=1e-10):
        super().__init__()
        self.message = message
        self.transition = transition
        self.quantum_layer = quantum_layer
        self.eps = eps

    def forward(self, state, ctx):
        scalar, vector = state.scalar, state.equi
        weight, direction, node_frame = ctx.cache["local_frames"]
        aux = None
        rope = None
        if weight is not None:
            args = scalar, vector, ctx.rbf, weight, direction, ctx.receiver, ctx.sender
            if self.quantum_layer is None:
                ds, dv = self.message(*args)
            else:
                rope, ds, dv = self.message(*args, state.aux["rope"])
            scalar, vector = scalar + ds, vector + dv
        if self.quantum_layer is not None:
            real, imag = ctx.cache["quantum_kernels"]
            quantum = contract_quantum(scalar, state.aux["quantum"], real[self.quantum_layer],
                                       imag[self.quantum_layer], self.eps)
            aux = {"quantum": quantum, "rope": rope}
        ds, dv = self.transition(scalar, vector, node_frame)
        return BlockState(scalar + ds, vector + dv, aux=aux)


class QuantumHead(Head):
    output_fields = ("atom_feature", "atom_vector_feature", "quantum_feature")

    def __init__(self, width, chi):
        super().__init__()
        self.chi = chi
        self.quantum_mix = nn.Linear(2 * chi, width)
        nn.init.xavier_uniform_(self.quantum_mix.weight)
        nn.init.zeros_(self.quantum_mix.bias)

    def forward(self, state, ctx, taps, inputs):
        quantum = state.aux["quantum"]
        packed = torch.cat((quantum.real, quantum.imag), dim=-1)
        return {"atom_feature": state.scalar + self.quantum_mix(packed) / self.chi,
                "atom_vector_feature": state.equi, "quantum_feature": packed / self.chi}
