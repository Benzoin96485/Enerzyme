"""
kNN + radius graph preprocessing and angular tensors (fairchem AllScAIP layout).

Ports logic from fairchem ``data_preprocess.py`` / ``escaip/utils/graph_utils.py`` (MIT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from e3nn.o3._spherical_harmonics import _spherical_harmonics
from torch import Tensor

from .radius_utils import envelope_fn, safe_normalize
from .radius_graph import biknn_radius_graph_from_batch
from .smearing import GaussianSmearing, LinearSigmoidSmearing, SigmoidSmearing, SiLUSmearing
from .types import GraphAttentionData


def _broadcast_scatter_index(index: Tensor, ref: Tensor, dim: int) -> Tensor:
    dim = ref.dim() + dim if dim < 0 else dim
    size = ((1,) * dim) + (-1,) + ((1,) * (ref.dim() - dim - 1))
    return index.view(size).expand_as(ref).to(torch.long)


def compilable_scatter(
    src: Tensor,
    index: Tensor,
    dim_size: int,
    dim: int = 0,
    reduce: str = "sum",
) -> Tensor:
    """Scatter-reduce matching fairchem ``compilable_scatter`` (sum along ``dim``)."""
    dim = src.dim() + dim if dim < 0 else dim
    size = src.size()[:dim] + (dim_size,) + src.size()[dim + 1 :]
    if reduce not in ("sum", "add"):
        raise ValueError(f"reduce={reduce} not supported")
    idx = _broadcast_scatter_index(index.long(), src, dim)
    return src.new_zeros(size).scatter_add_(dim, idx, src)


def get_edge_distance_expansion(
    distance_function: str,
    max_radius: float,
    edge_distance_expansion_size: int,
    edge_distance: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    expansion_cls = {
        "gaussian": GaussianSmearing,
        "sigmoid": SigmoidSmearing,
        "linear_sigmoid": LinearSigmoidSmearing,
        "silu": SiLUSmearing,
    }[distance_function]
    if distance_function == "silu":
        edge_distance_expansion_func = expansion_cls(
            0.0, max_radius, edge_distance_expansion_size, basis_width_scalar=2.0
        ).to(device=device, dtype=dtype)
    else:
        edge_distance_expansion_func = expansion_cls(
            0.0, max_radius, edge_distance_expansion_size, basis_width_scalar=2.0
        ).to(device=device, dtype=dtype)
    flat = edge_distance.flatten()
    out = edge_distance_expansion_func(flat)
    return out


def get_frequency_vectors(
    hidden_size: int,
    atten_num_heads: int,
    freequency_list: Sequence[int],
    edge_direction: Tensor,
) -> Tensor:
    head_dim = hidden_size // atten_num_heads
    sum_repeats = sum(freequency_list)
    assert sum_repeats == head_dim, (
        f"Sum of freequency_list ({sum_repeats}) must equal head_dim ({head_dim})."
    )
    lmax = len(freequency_list) - 1
    repeat_dims = list(freequency_list)
    edge_direction = edge_direction.to(torch.float32)
    harmonics = _spherical_harmonics(
        lmax, edge_direction[..., 0], edge_direction[..., 1], edge_direction[..., 2]
    )
    components: List[Tensor] = []
    curr_idx = 0
    for _l in range(lmax + 1):
        sh_dim = 2 * _l + 1
        curr_irrep = harmonics[:, :, curr_idx : curr_idx + sh_dim] / math.sqrt(sh_dim)
        rep_count = repeat_dims[_l]
        if rep_count > 0:
            component = curr_irrep.unsqueeze(2).expand(-1, -1, rep_count, -1)
            component = component.reshape(component.shape[0], component.shape[1], -1)
            components.append(component)
        curr_idx += sh_dim
    if components:
        return torch.cat(components, dim=-1)
    return torch.zeros(
        (edge_direction.shape[0], edge_direction.shape[1], 0),
        device=edge_direction.device,
    )


def get_node_direction_expansion_neighbor(
    direction_vec: Tensor,
    neighbor_mask: Tensor,
    lmax: int,
) -> Tensor:
    """BOO-style node scalars per angular momentum block, shape ``(N, lmax + 1)``."""
    neighbor_mask = neighbor_mask.float().unsqueeze(-1)
    edge_sh = _spherical_harmonics(
        lmax=lmax,
        x=direction_vec[:, :, 0],
        y=direction_vec[:, :, 1],
        z=direction_vec[:, :, 2],
    )
    sh_index = torch.arange(lmax + 1, device=edge_sh.device)
    sh_index = torch.repeat_interleave(sh_index, 2 * sh_index + 1)
    edge_sh = edge_sh / torch.clamp(torch.sqrt(2 * sh_index + 1), min=1e-6).unsqueeze(0).unsqueeze(0)
    masked_sh = edge_sh * neighbor_mask
    neighbor_count = neighbor_mask.sum(dim=1)
    neighbor_count = torch.clamp(neighbor_count, min=1)
    node_boo = masked_sh.sum(dim=1) / neighbor_count
    node_boo_squared = node_boo**2
    node_boo = compilable_scatter(
        node_boo_squared, sh_index, dim_size=lmax + 1, dim=-1, reduce="sum"
    )
    return torch.clamp(node_boo, min=1e-6).sqrt()


def get_node_attention_mask(
    node_batch: Tensor,
    dist_pairwise: Tensor | None,
    n_freq: int = 32,
    r_min: float = 0.25,
    r_max: float = 30.0,
    use_sincx_mask: bool = True,
    single_system_no_padding: bool = False,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    N_pad = node_batch.size(0)
    if single_system_no_padding:
        if not use_sincx_mask:
            return None, None, None
        base_mask = None
        valid_mask = None
    else:
        same_graph = node_batch.unsqueeze(1) == node_batch.unsqueeze(0)
        real = node_batch.unsqueeze(0) != -1
        real2 = node_batch.unsqueeze(1) != -1
        valid_mask = same_graph & real & real2
        base_mask = torch.zeros((N_pad, N_pad), dtype=torch.float32, device=node_batch.device)
        neg_inf = torch.finfo(base_mask.dtype).min
        base_mask = base_mask.masked_fill(~valid_mask, neg_inf)
        base_mask = base_mask.unsqueeze(0)
    if not use_sincx_mask:
        return None, base_mask, valid_mask
    if dist_pairwise is None:
        raise ValueError("dist_pairwise required when use_sincx_mask is True")
    omega_min = math.pi / (4.0 * r_max)
    omega_max = math.pi / r_min
    omega = torch.logspace(
        math.log10(omega_min),
        math.log10(omega_max),
        n_freq,
        device=node_batch.device,
        dtype=torch.float32,
    )
    x = dist_pairwise.unsqueeze(-1) * omega.view(1, 1, -1)
    sincx = torch.empty_like(x)
    small = x.abs() < 1e-4
    x_small = x[small]
    x2 = x_small * x_small
    sincx[small] = 1 - x2 / 6 + (x2 * x2) / 120
    sincx[~small] = torch.sin(x[~small]) / x[~small]
    return sincx, base_mask, valid_mask


def pad_batch(
    max_atoms: int,
    max_batch_size: int,
    atomic_numbers: Tensor,
    charge: Tensor | None,
    spin: Tensor | None,
    edge_direction: Tensor,
    edge_distance: Tensor,
    neighbor_index: Tensor,
    src_mask: Tensor,
    dst_mask: Tensor,
    src_index: Tensor,
    dst_index: Tensor,
    dist_pairwise: Tensor | None,
    node_batch: Tensor,
    num_graphs: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor | None,
    Tensor,
]:
    device = atomic_numbers.device
    _, num_nodes, _ = neighbor_index.shape
    pad_size = max_atoms - num_nodes
    if pad_size < 0:
        raise ValueError("max_atoms smaller than num_nodes")
    if max_batch_size < num_graphs:
        raise ValueError("max_batch_size smaller than num_graphs")
    atomic_numbers = F.pad(atomic_numbers, (0, pad_size), value=0)
    edge_direction = F.pad(edge_direction, (0, 0, 0, 0, 0, pad_size), value=0)
    edge_distance = F.pad(edge_distance, (0, 0, 0, pad_size), value=0)
    neighbor_index = F.pad(neighbor_index, (0, 0, 0, pad_size), value=-1)
    node_batch = F.pad(node_batch, (0, pad_size), value=-1)
    src_mask = F.pad(src_mask, (0, 0, 0, pad_size), value=float("-inf"))
    dst_mask = F.pad(dst_mask, (0, 0, 0, pad_size), value=float("-inf"))
    src_index = F.pad(src_index, (0, 0, 0, pad_size), value=-1)
    dst_index = F.pad(dst_index, (0, 0, 0, pad_size), value=-1)
    if dist_pairwise is not None:
        dist_pairwise = F.pad(dist_pairwise, (0, pad_size, 0, pad_size), value=0)
    if charge is not None:
        charge = F.pad(charge, (0, max_batch_size - num_graphs), value=0)
    else:
        charge = torch.zeros(max_batch_size, dtype=torch.float, device=device)
    if spin is not None:
        spin = F.pad(spin, (0, max_batch_size - num_graphs), value=0)
    else:
        spin = torch.zeros(max_batch_size, dtype=torch.float, device=device)
    return (
        atomic_numbers,
        charge,
        spin,
        edge_direction,
        edge_distance,
        neighbor_index,
        src_mask,
        dst_mask,
        src_index,
        dst_index,
        dist_pairwise,
        node_batch,
    )


@dataclass
class AllScAIPGraphHParams:
    """Hyperparameters for graph preprocessing (mirrors fairchem config groups)."""

    hidden_size: int = 128
    max_radius: float = 5.0
    knn_k: int = 32
    knn_soft: bool = False
    knn_sigmoid_scale: float = 0.2
    knn_lse_scale: float = 0.1
    knn_use_low_mem: bool = False
    knn_pad_size: int | None = None
    use_envelope: bool = True
    distance_function: str = "gaussian"
    edge_distance_expansion_size: int = 8
    edge_direction_expansion_size: int = 3
    node_direction_expansion_size: int = 3
    atten_num_heads: int = 4
    freequency_list: List[int] | None = None
    use_freq_mask: bool = True
    use_node_path: bool = True
    use_sincx_mask: bool = True
    attn_num_freq: int = 32
    single_system_no_padding: bool = False
    use_padding: bool = False
    max_atoms: int | None = None
    max_batch_size: int | None = None
    preprocess_on_cpu: bool = False
    pbc: bool = False


def _default_freequency_list(head_dim: int, lmax: int) -> List[int]:
    nlev = lmax + 1
    base = head_dim // nlev
    out = [base] * nlev
    out[-1] += head_dim - sum(out)
    return out


def build_graph_attention_data(
    Ra: Tensor,
    Za: Tensor,
    batch_seg: Tensor,
    hparams: AllScAIPGraphHParams | None = None,
    charge: Tensor | None = None,
    spin: Tensor | None = None,
    cell: Tensor | None = None,
) -> GraphAttentionData:
    """
    Build ``GraphAttentionData`` from Enerzyme fields (positions, species, batch).

    Tensor layout matches fairchem ``data_preprocess_radius_graph`` output.
    """
    hp = hparams or AllScAIPGraphHParams()
    original_device = Ra.device
    if hp.preprocess_on_cpu and original_device.type == "cuda":
        Ra = Ra.cpu()
        Za = Za.cpu()
        batch_seg = batch_seg.cpu()
        if charge is not None:
            charge = charge.cpu()
        if spin is not None:
            spin = spin.cpu()
        if cell is not None:
            cell = cell.cpu()
        work_device = torch.device("cpu")
    else:
        work_device = original_device

    Ra = Ra.to(device=work_device, dtype=torch.float32)
    Za = Za.to(device=work_device)
    batch_seg = batch_seg.to(device=work_device)
    num_graphs = int(batch_seg.max().item()) + 1
    num_nodes = Za.shape[0]

    head_dim = hp.hidden_size // hp.atten_num_heads
    lmax_edge = hp.edge_direction_expansion_size - 1
    lmax_node = hp.node_direction_expansion_size - 1
    freq_list = hp.freequency_list
    if freq_list is None:
        freq_list = _default_freequency_list(head_dim, lmax_edge)

    need_dist_pairwise = hp.use_node_path and hp.use_sincx_mask

    (
        dist_pairwise,
        disp,
        src_env,
        dst_env,
        src_index,
        dst_index,
        neighbor_index,
    ) = biknn_radius_graph_from_batch(
        Ra,
        batch_seg,
        hp.max_radius,
        hp.knn_k,
        knn_soft=hp.knn_soft,
        knn_sigmoid_scale=hp.knn_sigmoid_scale,
        knn_lse_scale=hp.knn_lse_scale,
        knn_use_low_mem=hp.knn_use_low_mem,
        knn_pad_size=hp.knn_pad_size,
        max_atoms=hp.max_atoms if hp.use_padding else None,
        compute_dist_pairwise=need_dist_pairwise,
        cell=cell,
        pbc=hp.pbc,
    )

    edge_direction = safe_normalize(disp, dim=-1)
    edge_distance = torch.linalg.vector_norm(disp, dim=-1)
    src_mask = src_env
    dst_mask = dst_env
    if hp.use_envelope:
        src_mask = envelope_fn(src_env, True)
        dst_mask = envelope_fn(dst_env, True)

    atomic_numbers = Za.long()
    node_batch = batch_seg

    if hp.use_padding:
        if hp.max_atoms is None or hp.max_batch_size is None:
            raise ValueError("use_padding requires max_atoms and max_batch_size")
        (
            atomic_numbers,
            charge_t,
            spin_t,
            edge_direction,
            edge_distance,
            neighbor_index,
            src_mask,
            dst_mask,
            src_index,
            dst_index,
            dist_pairwise,
            node_batch,
        ) = pad_batch(
            hp.max_atoms,
            hp.max_batch_size,
            atomic_numbers,
            charge,
            spin,
            edge_direction,
            edge_distance,
            neighbor_index,
            src_mask,
            dst_mask,
            src_index,
            dst_index,
            dist_pairwise,
            node_batch,
            num_graphs,
        )
        max_num_nodes = hp.max_atoms
        max_batch_size = hp.max_batch_size
    else:
        if charge is None:
            charge_t = torch.zeros(num_graphs, dtype=Ra.dtype, device=work_device)
        else:
            charge_t = charge.to(device=work_device, dtype=Ra.dtype)
            if charge_t.dim() == 0:
                charge_t = charge_t.expand(num_graphs)
        if spin is None:
            spin_t = torch.zeros(num_graphs, dtype=Ra.dtype, device=work_device)
        else:
            spin_t = spin.to(device=work_device, dtype=Ra.dtype)
            if spin_t.dim() == 0:
                spin_t = spin_t.expand(num_graphs)
        max_num_nodes = num_nodes
        max_batch_size = num_graphs

    edge_distance_expansion = get_edge_distance_expansion(
        hp.distance_function,
        hp.max_radius,
        hp.edge_distance_expansion_size,
        edge_distance,
        work_device,
        Ra.dtype,
    ).view(edge_distance.shape[0], edge_distance.shape[1], hp.edge_distance_expansion_size)

    edge_direction_expansion = _spherical_harmonics(
        lmax=lmax_edge,
        x=edge_direction[:, :, 0],
        y=edge_direction[:, :, 1],
        z=edge_direction[:, :, 2],
    )

    node_direction_expansion = get_node_direction_expansion_neighbor(
        direction_vec=edge_direction,
        neighbor_mask=src_mask != float("-inf"),
        lmax=lmax_node,
    )

    if hp.use_freq_mask:
        frequency_vectors = get_frequency_vectors(
            hp.hidden_size,
            hp.atten_num_heads,
            freq_list,
            edge_direction,
        )
    else:
        frequency_vectors = None

    if hp.use_node_path:
        sincx, base_mask, valid_mask = get_node_attention_mask(
            node_batch,
            dist_pairwise,
            hp.attn_num_freq,
            use_sincx_mask=hp.use_sincx_mask,
            single_system_no_padding=hp.single_system_no_padding,
        )
    else:
        sincx = None
        base_mask = None
        valid_mask = None

    node_batch = node_batch.masked_fill(node_batch == -1, 0)

    node_indices = torch.arange(max_num_nodes, device=work_device)
    node_padding_mask = (node_indices < num_nodes).to(torch.float32)

    x = GraphAttentionData(
        atomic_numbers=atomic_numbers,
        charge=charge_t,
        spin=spin_t,
        node_direction_expansion=node_direction_expansion,
        edge_distance_expansion=edge_distance_expansion,
        edge_direction_expansion=edge_direction_expansion,
        edge_direction=edge_direction,
        src_neighbor_attn_mask=src_mask,
        dst_neighbor_attn_mask=dst_mask,
        src_index=src_index,
        dst_index=dst_index,
        frequency_vectors=frequency_vectors,
        node_base_attn_mask=base_mask,
        node_sincx_matrix=sincx,
        node_valid_mask=valid_mask,
        neighbor_index=neighbor_index,
        node_batch=node_batch,
        node_padding_mask=node_padding_mask,
        max_batch_size=max_batch_size,
        num_graphs=num_graphs,
        max_num_nodes=max_num_nodes,
        num_nodes=num_nodes,
    )
    if hp.preprocess_on_cpu and original_device.type == "cuda":
        x = x.to(original_device)
    return x
