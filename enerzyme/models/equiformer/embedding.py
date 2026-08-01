# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License)
# into Enerzyme with package-local imports.

import torch
from torch import Tensor
from torch_scatter import scatter
from e3nn import o3

from ..e3nn_nn import LinearRS
from ..blocks.radial_mlp import RadialProfile
from ..layers._base_layer import BaseFFLayer
from .interaction import DepthwiseTensorProduct, _RESCALE, _USE_BIAS

_MAX_ATOM_TYPE = 87  # Enerzyme default covers Z up to 86
_AVG_DEGREE = 15.57930850982666
_AVG_NUM_NODES = 18.03065905448718


class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE, bias=True):
        
        super().__init__()
        self.max_atom_type = max_atom_type
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(o3.Irreps('{}x0e'.format(self.max_atom_type)), 
            self.irreps_node_embedding, bias=bias)
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type ** 0.5)
        
        
    def forward(self, node_atom):
        '''
            `node_atom` is a LongTensor.
        '''
        node_atom_onehot = torch.nn.functional.one_hot(
            node_atom, self.max_atom_type
        ).to(dtype=self.atom_type_lin.tp.weight.dtype)
        node_attr = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        
        return node_embedding, node_attr, node_atom_onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    

class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)
        
    
    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0, 
            dim_size=node_features.shape[0])
        return node_features


class EquiformerNodeEmbedding(BaseFFLayer):
    """Map atomic numbers ``Za`` to Equiformer irreps node embeddings.

    Unlike scalar :class:`~enerzyme.models.layers.atom_embedding.NuclearEmbedding`,
    this produces an e3nn irreps feature used by :class:`EquiformerCore`.
    """

    def __init__(
        self,
        max_Za: int,
        irreps_node_embedding: str = "128x0e+64x1e+32x2e",
        bias: bool = True,
    ) -> None:
        super().__init__(input_fields={"Za"}, output_fields={"atom_embedding"})
        self.max_Za = max_Za
        self.irreps_node_embedding = irreps_node_embedding
        self.embed = NodeEmbeddingNetwork(
            irreps_node_embedding,
            max_atom_type=max_Za + 1,
            bias=bias,
        )

    def get_atom_embedding(self, Za: Tensor) -> Tensor:
        embedding, _, _ = self.embed(Za.long())
        return embedding

