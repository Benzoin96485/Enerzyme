"""Helpers for Equiformer numerical parity against vendored upstream nets."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from e3nn import o3
from numpy.testing import assert_allclose
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "equiformer_upstream"
MOL_PATH = Path(__file__).resolve().parent / "fixtures" / "equiformer_parity_mol.npz"


def _dense_radius_graph(
    x: Tensor,
    r: float,
    batch: Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 1000,
    **kwargs,
) -> Tensor:
    """Pure-PyTorch radius graph for small parity fixtures (no torch-cluster)."""
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    src_list = []
    dst_list = []
    for b in batch.unique().tolist():
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        pos = x.index_select(0, idx)
        dist = torch.cdist(pos, pos)
        ii, jj = torch.where(dist <= r)
        if not loop:
            keep = ii != jj
            ii, jj = ii[keep], jj[keep]
        # Cap neighbors per destination roughly like torch_cluster
        if ii.numel() > 0 and max_num_neighbors is not None:
            # Keep all for small N; only truncate if pathological
            if ii.numel() > max_num_neighbors * idx.numel():
                # sort by distance and keep max_num_neighbors per dst
                d = dist[ii, jj]
                order = torch.argsort(d)
                ii, jj = ii[order], jj[order]
                kept_src, kept_dst = [], []
                counts = {}
                for s, t in zip(ii.tolist(), jj.tolist()):
                    counts[t] = counts.get(t, 0) + 1
                    if counts[t] <= max_num_neighbors:
                        kept_src.append(s)
                        kept_dst.append(t)
                ii = torch.tensor(kept_src, device=x.device, dtype=torch.long)
                jj = torch.tensor(kept_dst, device=x.device, dtype=torch.long)
        src_list.append(idx[ii])
        dst_list.append(idx[jj])
    if not src_list:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)


def _ensure_radius_graph_import() -> None:
    """Provide torch_cluster.radius_graph if the optional package is missing."""
    try:
        from torch_cluster import radius_graph  # noqa: F401

        return
    except ImportError:
        pass
    mod = types.ModuleType("torch_cluster")
    mod.radius_graph = _dense_radius_graph
    sys.modules["torch_cluster"] = mod


_ensure_radius_graph_import()

# Allow `import nets...` from the vendored fixture.
if str(FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARITY_HPARAMS = {
    "max_Za": 16,
    "num_rbf": 16,
    "r_max": 5.0,
    "num_layers": 2,
    "num_heads": 2,
    "fc_neurons": [64, 64],
    "irreps_node_embedding": "32x0e+16x1e+8x2e",
    "irreps_feature": "64x0e",
    "irreps_sh": "1x0e+1x1e+1x2e",
    "irreps_head": "16x0e+8x1o+4x2e",
    "irreps_mlp_mid": "32x0e+16x1e+8x2e",
    "irreps_node_attr": "1x0e",
    "nonlinear_message": True,
    "use_attn_head": False,
    "avg_degree": 15.57930850982666,
    "avg_num_nodes": 18.03065905448718,
}


def make_parity_hparams(**overrides: Any) -> Dict[str, Any]:
    hp = dict(PARITY_HPARAMS)
    hp.update(overrides)
    return hp


def assert_close(
    a: Tensor,
    b: Tensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    err_msg: str = "",
) -> None:
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}; {err_msg}"
    assert_allclose(
        a.detach().cpu().numpy(),
        b.detach().cpu().numpy(),
        atol=atol,
        rtol=rtol,
        err_msg=err_msg,
    )


def load_parity_molecule(dtype: torch.dtype = torch.float64) -> Dict[str, Tensor]:
    data = np.load(MOL_PATH)
    za = torch.as_tensor(data["Za"], dtype=torch.long)
    pos = torch.as_tensor(data["pos"], dtype=dtype)
    r_max = float(data["r_max"])
    return {"Za": za, "pos": pos, "r_max": r_max}


def build_radius_graph(
    pos: Tensor, batch: Tensor, r_max: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Return edge_src, edge_dst, edge_vec using the same convention as upstream."""
    _ensure_radius_graph_import()
    from torch_cluster import radius_graph

    edge_index = radius_graph(pos, r=r_max, batch=batch, max_num_neighbors=1000)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
    return edge_src, edge_dst, edge_vec


def to_enerzyme_edges(
    edge_src: Tensor, edge_dst: Tensor, edge_vec: Tensor
) -> Dict[str, Tensor]:
    """Map official (src,dst,vec) to Enerzyme idx_i/idx_j/vij.

    Official: vec = pos[src]-pos[dst], scatter onto dst.
    Enerzyme Core: edge_src=idx_j, edge_dst=idx_i, vij=Rj-Ri.
    """
    return {
        "idx_i_sr": edge_dst,
        "idx_j_sr": edge_src,
        "vij_sr": edge_vec,
        "Dij_sr": edge_vec.norm(dim=1).clamp_min(1e-12),
    }


def build_official_md17(hp: Dict[str, Any], dtype: torch.dtype = torch.float64):
    from nets.graph_attention_transformer import NodeEmbeddingNetwork
    from nets.graph_attention_transformer_md17 import GraphAttentionTransformerMD17

    model = GraphAttentionTransformerMD17(
        irreps_in="64x0e",
        irreps_node_embedding=hp["irreps_node_embedding"],
        num_layers=hp["num_layers"],
        irreps_node_attr=hp["irreps_node_attr"],
        irreps_sh=hp["irreps_sh"],
        max_radius=hp["r_max"],
        number_of_basis=hp["num_rbf"],
        basis_type="exp",
        fc_neurons=list(hp["fc_neurons"]),
        irreps_feature=hp["irreps_feature"],
        irreps_head=hp["irreps_head"],
        num_heads=hp["num_heads"],
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=hp["nonlinear_message"],
        irreps_mlp_mid=hp["irreps_mlp_mid"],
        use_attn_head=hp["use_attn_head"],
        norm_layer="layer",
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
    )
    # Match Enerzyme one-hot width (atomic numbers up to max_Za).
    max_atom_type = int(hp["max_Za"]) + 1
    model.atom_embed = NodeEmbeddingNetwork(
        model.irreps_node_embedding, max_atom_type=max_atom_type
    )
    return model.to(dtype=dtype).eval()


def build_enerzyme_parts(
    hp: Dict[str, Any],
    *,
    output_mode: str = "direct",
    dtype: torch.dtype = torch.float64,
):
    from enerzyme.models.equiformer.core import EquiformerCore
    from enerzyme.models.equiformer.node_embedding_layer import EquiformerNodeEmbedding
    from enerzyme.models.layers.rbf import ExpNormalSmearing

    embed = EquiformerNodeEmbedding(
        max_Za=hp["max_Za"],
        irreps_node_embedding=hp["irreps_node_embedding"],
    )
    rbf = ExpNormalSmearing(
        num_rbf=hp["num_rbf"],
        cutoff_sr=hp["r_max"],
        cuton=0.0,
        trainable=False,
    )
    core = EquiformerCore(
        num_rbf=hp["num_rbf"],
        irreps_node_embedding=hp["irreps_node_embedding"],
        irreps_feature=hp["irreps_feature"],
        irreps_node_attr=hp["irreps_node_attr"],
        irreps_sh=hp["irreps_sh"],
        irreps_head=hp["irreps_head"],
        irreps_mlp_mid=hp["irreps_mlp_mid"],
        num_layers=hp["num_layers"],
        num_heads=hp["num_heads"],
        fc_neurons=list(hp["fc_neurons"]),
        nonlinear_message=hp["nonlinear_message"],
        use_attn_head=hp["use_attn_head"],
        alpha_drop=0.0,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        avg_degree=hp["avg_degree"],
        avg_num_nodes=hp["avg_num_nodes"],
        output_mode=output_mode,
    )
    embed = embed.to(dtype=dtype).eval()
    rbf = rbf.to(dtype=dtype).eval()
    core = core.to(dtype=dtype).eval()
    return embed, rbf, core


def _load_matched(dst: torch.nn.Module, src: torch.nn.Module, label: str) -> None:
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    missing = [k for k in dst_sd if k not in src_sd]
    unexpected = [k for k in src_sd if k not in dst_sd]
    if missing or unexpected:
        raise KeyError(
            f"weight map mismatch for {label}: missing={missing[:10]} "
            f"unexpected={unexpected[:10]}"
        )
    # Shape check
    for k in dst_sd:
        if dst_sd[k].shape != src_sd[k].shape:
            raise RuntimeError(
                f"{label}.{k} shape {dst_sd[k].shape} != official {src_sd[k].shape}"
            )
    dst.load_state_dict(src_sd)


def copy_official_weights_to_enerzyme(
    official: torch.nn.Module,
    embed: torch.nn.Module,
    rbf: torch.nn.Module,
    core: torch.nn.Module,
) -> None:
    """Copy aligned modules from upstream MD17 model into Enerzyme pieces."""
    _load_matched(embed.embed, official.atom_embed, "atom_embed")

    # ExpNormal: official may expose means/betas as buffers on `rbf`.
    rbf_sd = {
        k: v
        for k, v in official.rbf.state_dict().items()
        if k in ("means", "betas")
    }
    if set(rbf_sd) != {"means", "betas"}:
        raise KeyError(f"official rbf keys unexpected: {list(official.rbf.state_dict())}")
    rbf.load_state_dict(rbf_sd, strict=False)
    # Ensure buffers match even if strict=False left others
    with torch.no_grad():
        rbf.means.copy_(official.rbf.means)
        rbf.betas.copy_(official.rbf.betas)

    _load_matched(core.edge_deg_embed, official.edge_deg_embed, "edge_deg_embed")
    _load_matched(core.blocks, official.blocks, "blocks")
    _load_matched(core.norm, official.norm, "norm")
    if hasattr(core, "head") and core.head is not None:
        _load_matched(core.head, official.head, "head")


@torch.no_grad()
def official_node_features_after_norm(
    official: torch.nn.Module,
    node_atom: Tensor,
    pos: Tensor,
    batch: Tensor,
    edge_src: Tensor,
    edge_dst: Tensor,
    edge_vec: Tensor,
) -> Tensor:
    """Mirror upstream forward through final LayerNorm (no energy head)."""
    from e3nn import o3

    edge_sh = o3.spherical_harmonics(
        l=official.irreps_edge_attr,
        x=edge_vec,
        normalize=True,
        normalization="component",
    )
    atom_embedding, _, _ = official.atom_embed(node_atom)
    edge_length = edge_vec.norm(dim=1)
    edge_length_embedding = official.rbf(edge_length)
    edge_degree_embedding = official.edge_deg_embed(
        atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
    )
    node_features = atom_embedding + edge_degree_embedding
    node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
    for blk in official.blocks:
        node_features = blk(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_length_embedding,
            batch=batch,
        )
    node_features = official.norm(node_features, batch=batch)
    return node_features


def official_atomic_energy(
    official: torch.nn.Module,
    node_features: Tensor,
    batch: Tensor,
    *,
    edge_src: Tensor,
    edge_dst: Tensor,
    edge_sh: Tensor,
    edge_scalars: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return (Ea_per_atom, E_graph) using the same head + ScaledScatter as upstream."""
    node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
    if official.use_attn_head:
        ea = official.head(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_scalars,
            batch=batch,
        )
    else:
        ea = official.head(node_features)
    ea = ea.view(-1)
    e_graph = official.scale_scatter(ea.unsqueeze(-1), batch, dim=0).view(-1)
    return ea, e_graph


def enerzyme_atomic_energy(
    core: torch.nn.Module,
    node_features: Tensor,
    batch: Tensor,
    *,
    edge_src: Tensor,
    edge_dst: Tensor,
    edge_sh: Tensor,
    edge_scalars: Tensor,
    avg_num_nodes: float,
) -> Tuple[Tensor, Tensor]:
    """Match upstream head then ScaledScatter reduction for graph energy."""
    from torch_scatter import scatter

    node_attr = torch.ones_like(node_features.narrow(1, 0, 1))
    if core.use_attn_head:
        ea = core.head(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_scalars,
            batch=batch,
        )
    else:
        ea = core.head(node_features)
    ea = ea.view(-1)
    e_sum = scatter(ea, batch, dim=0, reduce="sum")
    e_graph = e_sum / (avg_num_nodes ** 0.5)
    return ea, e_graph


def enerzyme_feature_atom_feature(
    core: torch.nn.Module,
    embed: torch.nn.Module,
    rbf: torch.nn.Module,
    za: Tensor,
    batch: Tensor,
    ez: Dict[str, Tensor],
) -> Tensor:
    """Feature-mode Core ``get_output`` → full-irreps ``atom_feature``."""
    assert getattr(core, "output_mode", None) == "feature", (
        "enerzyme_feature_atom_feature expects output_mode='feature'"
    )
    atom_emb = embed.get_atom_embedding(za)
    rbf_vals = rbf.get_rbf(ez["Dij_sr"])
    out = core.get_output(
        vij_sr=ez["vij_sr"],
        idx_i_sr=ez["idx_i_sr"],
        idx_j_sr=ez["idx_j_sr"],
        rbf=rbf_vals,
        atom_embedding=atom_emb,
        batch_seg=batch,
    )
    return out["atom_feature"]


def scaled_scatter_energy(ea: Tensor, batch: Tensor, avg_num_nodes: float) -> Tensor:
    """Upstream ScaledScatter: sum / sqrt(N_avg)."""
    from torch_scatter import scatter

    e_sum = scatter(ea.view(-1), batch, dim=0, reduce="sum")
    return e_sum / (avg_num_nodes ** 0.5)


def build_linear_rs_energy_head(
    num_0e: int,
    *,
    flavor: str = "enerzyme",
    dtype: torch.dtype = torch.float64,
) -> torch.nn.Module:
    """Official-style scalar energy MLP: LinearRS → SiLU → LinearRS → 1x0e.

    ``flavor='official'`` uses vendored upstream modules; ``enerzyme`` uses the
    ported LinearRS / Activation. State dicts are compatible across flavors.
    """
    ir = o3.Irreps(f"{num_0e}x0e")
    ir_out = o3.Irreps("1x0e")
    if flavor == "official":
        from nets.fast_activation import Activation
        from nets.graph_attention_transformer_md17 import _RESCALE
        from nets.tensor_product_rescale import LinearRS
    elif flavor == "enerzyme":
        from enerzyme.models.equiformer.attention import _RESCALE
        from enerzyme.models.equiformer.fast_activation import Activation
        from enerzyme.models.equiformer.tensor_product import LinearRS
    else:
        raise ValueError(f"unknown flavor {flavor!r}")
    head = torch.nn.Sequential(
        LinearRS(ir, ir, rescale=_RESCALE),
        Activation(ir, acts=[torch.nn.SiLU()]),
        LinearRS(ir, ir_out, rescale=_RESCALE),
    )
    return head.to(dtype=dtype).eval()
