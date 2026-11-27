"""Atom neighbourhoods and optional atom–fragment message passing for E2Former."""

import torch

from ..mp.block import Block, BlockState
from ..mp.core import Embed
from ..mp.message.so2 import SphereHead
from .cluster import FragmentTopology, pool_fragment_irreps, resolve_fragments, select_fragment_neighbors
from .graph import build_topk_neighborhood, center_positions_by_batch
from .wigner_otf import build_so2_wigner_frames


class NeighborhoodEmbed(Embed):
    def __init__(self, projection, degree, lmax, max_neighbors, use_frames, paths):
        super().__init__()
        self.projection = projection
        self.degree = degree
        self.lmax = lmax
        self.max_neighbors = max_neighbors
        self.use_frames = use_frames
        self.paths = paths

    def forward(self, inputs, ctx):
        node = inputs["atom_embedding"]
        z = inputs["Za"].long()
        batch = ctx.batch if ctx.batch is not None else z.new_zeros(ctx.num_nodes)
        graph = build_topk_neighborhood(inputs["Ra"].to(node.dtype), ctx.receiver, ctx.sender,
                                       ctx.vec, ctx.rbf, max_neighbors=self.max_neighbors, batch_seg=batch)
        x = node.new_zeros(ctx.num_nodes, (self.lmax + 1) ** 2, self.projection.out_features)
        x[:, 0, :] = self.projection(node)
        x = x + self.degree(z, graph["edge_vec"], graph["attn_mask"], graph["attn_weight"], graph["f_sparse_idx_node"])
        pos = graph["f_node_pos_wigner"]
        if self.use_frames:
            graph.update(build_so2_wigner_frames(pos, graph["f_exp_node_pos"], self.lmax,
                                                 l3_sequential=self.paths, training=self.training))
        return BlockState(equi=x, edge=graph["attn_weight"], aux={"graph": graph, "pos": pos, "batch": batch, "Za": z})


def neighborhood_attention(layer, state, ctx):
    graph = state.aux["graph"]
    x, weight = layer(node_pos=state.aux["pos"], node_irreps=state.equi, edge_dis=graph["edge_dis"],
                      edge_vec=graph["edge_vec"], attn_weight=state.edge, atomic_numbers=state.aux["Za"],
                      attn_mask=graph["attn_mask"], batched_data=graph)
    return BlockState(equi=x, edge=weight, aux=state.aux)


class FragmentGraph(Block):
    topology_fields = ("Ra", "batch_seg", "cluster_ids")

    def __init__(self, radial, cutoff, max_neighbors, mode, min_nodes, random_state, use_frames, lmax, paths):
        super().__init__()
        self.radial = radial
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.mode = mode
        self.min_nodes = min_nodes
        self.random_state = random_state
        self.use_frames = use_frames
        self.lmax = lmax
        self.paths = paths

    def _fragments(self, pos, batch, cluster_ids):
        ids, centers, cluster_batch, _ = resolve_fragments(
            pos, batch, fragment_mode=self.mode, cluster_ids=cluster_ids,
            min_nodes_per_group=self.min_nodes, random_state=self.random_state,
        )
        neighbors = select_fragment_neighbors(pos, centers, ids, batch, cluster_batch,
                                              self.cutoff, self.max_neighbors)
        return ids, centers, neighbors

    def prepare_force_topology(self, Ra, batch_seg, cluster_ids):
        if Ra is None:
            return None
        if batch_seg is None:
            batch_seg = Ra.new_zeros(len(Ra), dtype=torch.long)
        pos = center_positions_by_batch(Ra, batch_seg)
        ids, centers, neighbors = self._fragments(pos, batch_seg, cluster_ids)
        return FragmentTopology(ids, torch.bincount(ids, minlength=len(centers)), neighbors)

    def forward(self, state, ctx):
        pos, batch = state.aux["pos"], state.aux["batch"]
        topology = ctx.cache.get("lsr_fragments")
        if topology is None:
            ids, centers, neighbors = self._fragments(pos, batch, ctx.inputs.get("cluster_ids"))
            graph = neighbors.geometry(pos, centers)
        else:
            ids = topology.ids
            centers, graph = topology.geometry(pos)
        weights = self.radial(graph["edge_dis"]).masked_fill(graph["attn_mask"], 0.0)
        if self.use_frames and centers.shape[0]:
            graph.update(build_so2_wigner_frames(pos, centers, self.lmax,
                                                 l3_sequential=self.paths, training=self.training))
        aux = {"graph": graph, "pos": pos, "centers": centers, "ids": ids, "Za": state.aux["Za"]}
        return BlockState(scalar=state.equi, equi=state.equi, edge=weights, aux=aux)


def fragment_attention(layer, state, ctx):
    aux = state.aux
    if aux["centers"].shape[0] == 0:
        return state
    graph = aux["graph"]
    clusters = pool_fragment_irreps(state.equi, aux["ids"], aux["centers"].shape[0])
    x, weight = layer(node_pos=aux["pos"], node_irreps=state.equi, edge_dis=graph["edge_dis"],
                      edge_vec=graph["edge_vec"], attn_weight=state.edge, atomic_numbers=aux["Za"],
                      attn_mask=graph["attn_mask"], batched_data=graph,
                      cluster_pos=aux["centers"], cluster_irreps=clusters)
    return BlockState(scalar=state.scalar, equi=x, edge=weight, aux=aux)


class FragmentHead(SphereHead):
    def __init__(self, norm, norm_short, norm_long, fuse, has_long_blocks):
        super().__init__(norm)
        self.norm_short = norm_short
        self.norm_long = norm_long
        self.fuse = fuse
        self.has_long_blocks = has_long_blocks

    def forward(self, state, ctx, taps, inputs):
        if self.has_long_blocks and state.aux["centers"].shape[0]:
            x = self.fuse(torch.cat((self.norm_short(state.scalar), self.norm_long(state.equi)), dim=-1))
        else:
            x = self.norm(state.scalar)
        return {"atom_feature": x[:, 0, :], "atom_sphere_feature": x}
