"""Spherical node states, edge rotations and SO(2) block adapters."""

import torch

from ..block import Block, BlockState
from ..core import Embed
from ..equivariant import CoefficientMapping, SO3_Embedding, SO3_Rotation, init_edge_rot_mat
from ..head import Head


def _frame(inputs, vec):
    frame = inputs.get("edge_rot_mat")
    return init_edge_rot_mat(vec) if frame is None else frame.to(vec)


def _seed(node, projection, lmax):
    scalar = projection(node)
    x = scalar.new_zeros(node.shape[0], (lmax + 1) ** 2, scalar.shape[-1])
    x[:, 0, :] = scalar
    return x


class SphereHead(Head):
    output_fields = ("atom_feature", "atom_sphere_feature")

    def __init__(self, norm=None, *, wrapped=False, focus=False):
        super().__init__()
        self.norm = norm
        self.wrapped = wrapped
        self.focus = focus

    def forward(self, state, ctx, taps, inputs):
        x = state.equi.embedding if self.wrapped else state.equi
        if self.norm is not None:
            x = self.norm(x)
        if self.focus:
            x = x.squeeze(2)
        return {"atom_feature": x[:, 0, :], "atom_sphere_feature": x}


class ChannelEmbed(Embed):
    """Scalar-to-sphere projection in the edge frames of eSCN or EquiformerV2."""

    def __init__(self, sphere_proj, lmax, mmax, degree=None, mapping=None):
        super().__init__()
        self.sphere_proj = sphere_proj
        self.lmax = lmax
        self.mmax = mmax
        self.degree = degree
        self.mapping = mapping if mapping is not None else CoefficientMapping([lmax], [mmax], torch.device("cpu"))

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        z = inputs["Za"].long()
        edge = torch.stack((ctx.sender, ctx.receiver))
        rotation = [SO3_Rotation(_frame(inputs, -ctx.vec), self.lmax, apply_rotate_inv_rescale=self.degree is not None)]
        mapping = self.mapping.to(node.device)
        x = SO3_Embedding(ctx.num_nodes, [self.lmax], self.sphere_proj.out_features, node.device, node.dtype)
        x.embedding[:, 0, :] = self.sphere_proj(node)
        batch = ctx.batch
        if self.degree is not None:
            x.embedding = x.embedding + self.degree(z, ctx.rbf, edge, rotation=rotation).embedding
            if batch is None:
                batch = z.new_zeros(ctx.num_nodes)
        return BlockState(equi=x, edge=edge, aux={"rotation": rotation, "mapping": mapping, "Za": z, "batch": batch})


class ChannelBlock(Block):
    def __init__(self, module, residual):
        super().__init__()
        self.module = module
        self.residual = residual

    def forward(self, state, ctx):
        x = self.module(state.equi, state.aux["Za"], ctx.rbf, state.edge, state.aux["rotation"], state.aux["mapping"])
        if self.residual:
            x.embedding = state.equi.embedding + x.embedding
        return state._replace(equi=x)


def channel_attention(layer, state, ctx):
    x = layer(state.equi, state.aux["Za"], ctx.rbf, state.edge, batch=state.aux["batch"],
              rotation=state.aux["rotation"])
    return state._replace(equi=x)


class FusedChannelEmbed(Embed):
    def __init__(self, projection, lmax, rotation, degree, envelope):
        super().__init__()
        self.projection = projection
        self.lmax = lmax
        self.rotation = rotation
        self.degree = degree
        self.envelope = envelope

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        z = inputs["Za"].long()
        batch = ctx.batch if ctx.batch is not None else z.new_zeros(ctx.num_nodes)
        edge = torch.stack((ctx.sender, ctx.receiver))
        rotation = self.rotation.for_edges(_frame(inputs, -ctx.vec))
        window = inputs.get("edge_envelope_weight")
        if window is None and self.envelope is not None:
            window = self.envelope(ctx.edge_length())
        x = _seed(node, self.projection, self.lmax)
        x = x + self.degree(z, ctx.rbf, edge, window, rotation=rotation)
        return BlockState(equi=x, edge=edge, aux={"source_z": z[edge[0]], "target_z": z[edge[1]],
                                                  "window": window, "batch": batch, "rotation": rotation})


def fused_attention(layer, state, ctx):
    aux = state.aux
    x = layer(state.equi, aux["source_z"], aux["target_z"], ctx.rbf, state.edge,
              aux["window"], aux["batch"], rotation=aux["rotation"])
    return state._replace(equi=x)


class SmoothChannelEmbed(Embed):
    def __init__(self, projection, source, target, degree, lmax, mapping):
        super().__init__()
        self.projection = projection
        self.source = source
        self.target = target
        self.degree = degree
        self.lmax = lmax
        self.mapping = mapping

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        self.mapping.to(node.device)
        z = inputs["Za"].to(device=node.device, dtype=torch.long)
        src, dst = ctx.sender.to(node.device), ctx.receiver.to(node.device)
        edge = torch.stack((src, dst))
        x = _seed(node, self.projection, self.lmax)
        rotation = radial = distance = None
        if ctx.num_edges:
            vec = ctx.vec.to(node.device)
            rotation = [SO3_Rotation(_frame(inputs, vec), self.lmax, apply_rotate_inv_rescale=False)]
            distance = torch.linalg.vector_norm(vec, dim=-1)
            radial = torch.cat((ctx.rbf.to(node.device), self.source(z[src]), self.target(z[dst])), dim=1)
            x = x + self.degree(radial, distance, edge, rotation, ctx.num_nodes)
        return BlockState(equi=x, edge=edge, aux={"radial": radial, "distance": distance, "rotation": rotation})


def smooth_convolution(layer, state, ctx):
    aux = state.aux
    x = layer(state.equi, aux["radial"], aux["distance"], state.edge, aux["rotation"])
    return state._replace(equi=x)
