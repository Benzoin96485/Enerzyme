from typing import Iterable, Tuple, Dict, Any, List, Union, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
import numpy as np
from ..data import is_atomic, is_int, is_idx, requires_grad, is_target, is_target_uq, get_tensor_rank, is_grad
from ..data.neighbor_list import full_neighbor_list
from ..utils import logger


def _generator_is_enabled(generator_config: Optional[Dict[str, Any]]) -> bool:
    if generator_config is None:
        return False
    if isinstance(generator_config, dict) and generator_config.get("enabled") is False:
        return False
    return True


def _inject_generator_flow_dict(
    batch_features: Dict[str, Tensor],
    batch_targets: Dict[str, Tensor],
    generator_config: Dict[str, Any],
    dtype: torch.dtype,
    generator_training: bool = True,
) -> None:
    """Add ``flow_t`` (per graph), ``Q_flow_a``, ``S_flow_a`` for conditional flow matching.

    When ``generator_training`` is False (eval / predict collate), set ``flow_t=0`` and
    ``Q_flow_a``/``S_flow_a`` to the init values so the ODE can start at t=0 without random
    interpolation (targets are still required for metrics).
    """
    cfg = generator_config
    init_q = cfg.get("init_q_key", "Q_init_a")
    init_s = cfg.get("init_s_key", "S_init_a")
    tq = cfg.get("target_q_key", "Qa")
    ts = cfg.get("target_s_key", "Sa")
    out_q = cfg.get("out_q_key", "Q_flow_a")
    out_s = cfg.get("out_s_key", "S_flow_a")
    t_key = cfg.get("t_key", "flow_t")

    if not batch_targets:
        logger.warning("Generator: no targets in batch; skip flow fields.")
        return
    missing = []
    for key, store, label in (
        (init_q, batch_features, "features"),
        (init_s, batch_features, "features"),
        (tq, batch_targets, "targets"),
        (ts, batch_targets, "targets"),
    ):
        if key not in store:
            missing.append(f"{label}.{key}")
    if missing:
        logger.warning("Generator: skip flow fields; missing: %s", ", ".join(missing))
        return

    batch_seg = batch_features["batch_seg"]
    num_graphs = int(batch_seg.max().item()) + 1
    q_init = batch_features[init_q].to(dtype=dtype)
    s_init = batch_features[init_s].to(dtype=dtype)
    q_tgt = batch_targets[tq].to(dtype=dtype)
    s_tgt = batch_targets[ts].to(dtype=dtype)
    if generator_training:
        t_g = torch.rand(num_graphs, dtype=dtype, device=batch_seg.device)
        t_atom = t_g[batch_seg]
        batch_features[t_key] = t_g
        batch_features[out_q] = (1.0 - t_atom) * q_init + t_atom * q_tgt
        batch_features[out_s] = (1.0 - t_atom) * s_init + t_atom * s_tgt
    else:
        batch_features[t_key] = torch.zeros(num_graphs, dtype=dtype, device=batch_seg.device)
        batch_features[out_q] = q_init.clone()
        batch_features[out_s] = s_init.clone()
    # Expose init tensors on targets so losses (e.g. CFMLoss) see (output, net_target) only.
    batch_targets[init_q] = batch_features[init_q]
    batch_targets[init_s] = batch_features[init_s]


def _inject_generator_flow_pyg(
    batch_features: Batch,
    batch_targets: Batch,
    generator_config: Dict[str, Any],
    dtype: torch.dtype,
    generator_training: bool = True,
) -> None:
    cfg = generator_config
    init_q = cfg.get("init_q_key", "Q_init_a")
    init_s = cfg.get("init_s_key", "S_init_a")
    tq = cfg.get("target_q_key", "Qa")
    ts = cfg.get("target_s_key", "Sa")
    out_q = cfg.get("out_q_key", "Q_flow_a")
    out_s = cfg.get("out_s_key", "S_flow_a")
    t_key = cfg.get("t_key", "flow_t")

    missing = []
    for key, store, label in (
        (init_q, batch_features, "features"),
        (init_s, batch_features, "features"),
        (tq, batch_targets, "targets"),
        (ts, batch_targets, "targets"),
    ):
        if key not in store:
            missing.append(f"{label}.{key}")
    if missing:
        logger.warning("Generator (pyg): skip flow fields; missing: %s", ", ".join(missing))
        return

    b = batch_features.batch
    num_graphs = int(b.max().item()) + 1
    q_init = batch_features[init_q].to(dtype=dtype)
    s_init = batch_features[init_s].to(dtype=dtype)
    q_tgt = batch_targets[tq].to(dtype=dtype)
    s_tgt = batch_targets[ts].to(dtype=dtype)
    if generator_training:
        t_g = torch.rand(num_graphs, dtype=dtype, device=b.device)
        t_atom = t_g[b]
        batch_features[t_key] = t_g
        batch_features[out_q] = (1.0 - t_atom) * q_init + t_atom * q_tgt
        batch_features[out_s] = (1.0 - t_atom) * s_init + t_atom * s_tgt
    else:
        batch_features[t_key] = torch.zeros(num_graphs, dtype=dtype, device=b.device)
        batch_features[out_q] = q_init.clone()
        batch_features[out_s] = s_init.clone()
    batch_targets[init_q] = batch_features[init_q]
    batch_targets[init_s] = batch_features[init_s]


class _StructureData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("idx_i", "idx_j"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def _batch_geometry(features):
    """Complete optional per-structure geometry when periodic and open samples share a batch."""
    has_cell = any("cell" in f for f in features)
    has_pbc = any("pbc" in f for f in features)
    if not (has_cell or has_pbc):
        return features
    result = []
    for f in features:
        item = dict(f)
        if has_pbc:
            flags = np.asarray(f.get("pbc", False), dtype=bool)
            if flags.ndim == 0:
                flags = np.full(3, flags, dtype=bool)
            if flags.shape != (3,):
                raise ValueError("pbc must be a scalar or three flags, one per lattice vector")
            item["pbc"] = flags
        if has_cell and "cell" not in f:
            if np.any(item.get("pbc", False)):
                raise ValueError("Periodic samples require a cell when batching crystal geometry")
            item["cell"] = np.zeros((3, 3))
        result.append(item)
    return result


def _batch_neighbor_lists(features, otf_graph):
    with_offsets = any("offsets" in f for f in features)
    lists = []
    for f in features:
        periodic = "pbc" in f and np.any(f["pbc"])
        if "offsets" in f and "idx_i" not in f:
            raise ValueError("Cartesian offsets require a precomputed neighbor list")
        if ("idx_i" in f) != ("idx_j" in f):
            raise ValueError("A precomputed neighbor list requires both idx_i and idx_j")
        if "idx_i" in f:
            n_pair = int(f.get("N_pair", len(f["idx_i"])))
            if n_pair < 0 or n_pair > min(len(f["idx_i"]), len(f["idx_j"])):
                raise ValueError("N_pair exceeds the available neighbor indices")
            idx_i, idx_j = np.asarray(f["idx_i"][:n_pair]), np.asarray(f["idx_j"][:n_pair])
            if "offsets" in f:
                shifts = np.asarray(f["offsets"][:n_pair])
                if shifts.shape != (n_pair, 3):
                    raise ValueError("offsets must have one Cartesian vector per neighbor pair")
            elif periodic:
                raise ValueError("Periodic precomputed edges require Cartesian offsets")
            else:
                shifts = None
        elif otf_graph:
            if periodic:
                raise ValueError("Periodic samples need a precomputed neighbor_list='cutoff' with Cartesian offsets")
            idx_i, idx_j = full_neighbor_list(f["N"])
            shifts = None
        else:
            lists.append(None)
            continue
        if with_offsets and shifts is None:
            shifts = np.zeros((len(idx_i), 3))
        lists.append((idx_i, idx_j, shifts))
    count = sum(pair is not None for pair in lists)
    if 0 < count < len(features):
        raise ValueError(f"Incomplete neighbor lists in batch: {count}/{len(features)} structures have edges.")
    return lists


def _decorate_pyg_batch_input(
    batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]],
    dtype: torch.dtype,
    device: torch.device,
    otf_graph: bool = True,
    generator_config: Optional[Dict[str, Any]] = None,
    generator_training: bool = True,
) -> Tuple[Batch, Batch]:
    if len(batch[0]) == 3:
        features, targets, data_keys = zip(*batch)
    else:
        features, targets = zip(*batch)
        data_keys = [None] * len(features)
    features = _batch_geometry(features)
    feature_list = []
    for feature, pair in zip(features, _batch_neighbor_lists(features, otf_graph)):
        data_dict = dict()
        for k, v in feature.items():
            if is_atomic(k):
                data_dict[k] = torch.tensor(
                    v[:feature["N"]],
                    dtype=torch.long if is_int(k) else dtype
                )
            elif not is_idx(k):
                data_dict[k] = torch.tensor(
                    v,
                    dtype=torch.long if is_int(k) else dtype,
                )
                # Graph-level tensors with rank > 0 (e.g. (3,3) cell) need an
                # explicit leading batch dim, or PyG's default dim-0 concat in
                # Batch.from_data_list flattens them instead of stacking them
                # (mirrors the same unsqueeze already done for targets below).
                if get_tensor_rank(k) > 0:
                    data_dict[k] = data_dict[k].unsqueeze(0)
        data_dict["N"] = feature["N"]
        if pair is None:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            idx_i, idx_j, shifts = pair
            data_dict["idx_i"] = torch.as_tensor(idx_i, dtype=torch.long)
            data_dict["idx_j"] = torch.as_tensor(idx_j, dtype=torch.long)
            edge_index = torch.stack((data_dict["idx_i"], data_dict["idx_j"]))
            if shifts is not None:
                data_dict["offsets"] = torch.as_tensor(shifts, dtype=dtype)
        feature_list.append(_StructureData(edge_index=edge_index, num_nodes=feature["N"], **data_dict))
    batch_features = Batch.from_data_list(feature_list)
    for k, v in batch_features.items():
        if requires_grad(k):
            v.requires_grad_(True)

    target_list = []
    for i, target in enumerate(targets):
        data_dict = dict()
        for k, v in (target or {}).items():
            if is_atomic(k):
                data_dict[k] = torch.tensor(
                    v[:features[i]["N"]],
                    dtype=torch.long if is_int(k) else dtype
                )
            elif k == "data_key":
                data_dict[k] = v
            else:
                data_dict[k] = torch.tensor(
                    v,
                    dtype=torch.long if is_int(k) else dtype,
                )
                if get_tensor_rank(k) > 0:
                    data_dict[k] = data_dict[k].unsqueeze(0)
            data_dict["N"] = features[i]["N"]
        data = Data(num_nodes=features[i]["N"], data_key=data_keys[i], **data_dict)
        target_list.append(data)
    batch_targets = Batch.from_data_list(target_list)
    for k, v in batch_targets.items():
        if requires_grad(k):
            v.requires_grad_(True)
    if _generator_is_enabled(generator_config):
        _inject_generator_flow_pyg(
            batch_features, batch_targets, generator_config, dtype, generator_training
        )
    return batch_features, batch_targets


def _decorate_batch_input(
    batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]],
    dtype: torch.dtype,
    device: torch.device,
    otf_graph: bool = True,
    generator_config: Optional[Dict[str, Any]] = None,
    generator_training: bool = True,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    if len(batch[0]) == 3:
        features, targets, data_keys = zip(*batch)
    else:
        features, targets = zip(*batch)
        data_keys = None
    features = _batch_geometry(features)
    batch_features = dict()
    batch_targets = dict()
    
    for k in features[0]:
        if is_atomic(k):
            batch_features[k] = torch.tensor(
                np.concatenate([feature[k][:feature["N"]] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
                requires_grad=requires_grad(k)
            )
        elif not is_idx(k):
            batch_features[k] = torch.tensor(
                np.array([feature[k] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
            )

    batch_idx_i = []
    batch_idx_j = []
    batch_offsets = []
    batch_seg = []
    count = 0
    for i, (feature, pair) in enumerate(zip(features, _batch_neighbor_lists(features, otf_graph))):
        if pair is not None:
            idx_i, idx_j, shifts = pair
            batch_idx_i.append(idx_i + count)
            batch_idx_j.append(idx_j + count)
            if shifts is not None:
                batch_offsets.append(shifts)
        batch_seg.append(np.full(feature["N"], i, dtype=int))
        count += feature["N"]
    batch_features["N"] = [feature["N"] for feature in features]
    batch_features["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long)
    if batch_idx_i:
        batch_features["idx_i"] = torch.tensor(np.concatenate(batch_idx_i), dtype=torch.long)
        batch_features["idx_j"] = torch.tensor(np.concatenate(batch_idx_j), dtype=torch.long)
        if batch_offsets:
            batch_features["offsets"] = torch.tensor(np.concatenate(batch_offsets), dtype=dtype)

    if targets[0] is not None:
        for k in targets[0]:
            if is_atomic(k): 
                batch_targets[k] = torch.tensor(
                    np.concatenate([target[k][:features[i]["N"]] for i, target in enumerate(targets)]), 
                    dtype=torch.long if is_int(k) else dtype
                )
            else:
                batch_targets[k] = torch.tensor(
                    np.array([target[k] for target in targets]), 
                    dtype=torch.long if is_int(k) else dtype,
                )
        batch_targets["data_key"] = data_keys
    if _generator_is_enabled(generator_config):
        _inject_generator_flow_dict(
            batch_features, batch_targets, generator_config, dtype, generator_training
        )
    return batch_features, batch_targets


def _decorate_batch_output(output: Dict[str, Any], features: Dict[str, Any], targets: Optional[Dict[str, Any]], non_target_features: List[str]=[]) -> Tuple[Dict[str, Union[np.ndarray, List]], Optional[Dict[str, Union[np.ndarray, List]]]]:
    y_pred = dict()
    y_truth = dict()
    for k, v in output.items():
        if is_target(k):
            if is_atomic(k):
                y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
            else:
                y_pred[k] = v.detach().cpu().numpy()
        elif is_target_uq(k):
            target = k[:-4]
            if is_atomic(target):
                y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
            else:
                y_pred[k] = v.detach().cpu().numpy()
        elif is_grad(k):
            y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
    for k in non_target_features:
        if k in y_pred:
            continue
        if len(output[k]) == len(features["Za"]):
            y_pred[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(output[k], features["N"])))
        elif len(output[k]) == len(features["N"]):
            y_pred[k] = output[k].detach().cpu().numpy()
        else:
            raise ValueError(f"non-target feature {k} has invalid length {len(output[k])}")
    
    y_pred["Za"] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(features["Za"], features["N"])))
    
    if targets is not None:
        for k, v in targets.items():
            if is_target(k):
                if is_atomic(k):
                    y_truth[k] = list(map(lambda x: x.detach().cpu().numpy(), torch.split(v, features["N"])))
                else:
                    y_truth[k] = v.detach().cpu().numpy()
        y_pred["data_key"] = targets["data_key"]
        y_truth["data_key"] = targets["data_key"]
    y_truth["Za"] = y_pred["Za"]

    # Delta-learning inverse: priors must sit next to residual Qa / Sa on the metric dict
    for _prior_key in ("Q_init_a", "S_init_a"):
        if _prior_key in features and is_atomic(_prior_key):
            split_prior = list(
                map(
                    lambda x: x.detach().cpu().numpy(),
                    torch.split(features[_prior_key], features["N"]),
                )
            )
            y_pred[_prior_key] = split_prior
            if targets is not None:
                y_truth[_prior_key] = split_prior

    # Graph-level Q / S targets with atomic-only model output (e.g. ODE flow returns Qa, Sa only).
    if targets is not None:
        if (
            is_target("Q")
            and "Q" not in y_pred
            and "Qa" in y_pred
            and y_pred["Qa"]
        ):
            y_pred["Q"] = np.array([float(np.sum(x)) for x in y_pred["Qa"]], dtype=np.float64)
        if (
            is_target("S")
            and "S" not in y_pred
            and "Sa" in y_pred
            and y_pred["Sa"]
        ):
            y_pred["S"] = np.array([float(np.sum(x)) for x in y_pred["Sa"]], dtype=np.float64)

    return y_pred, (y_truth if y_truth else None)


def _pyg_to_device(batch: Iterable[Tuple[Batch, Batch]], device: torch.device) -> Tuple[Batch, Batch]:
    features, targets = batch
    return features.to(device), targets.to(device)


def _to_device(batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]], device: torch.device) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    features, targets = batch
    batch_features = dict()
    batch_targets = dict()
    for k, v in features.items():
        if k != "N":
            batch_features[k] = v.to(device)
        else:
            batch_features[k] = v
    for k, v in targets.items():
        if k != "data_key":
            batch_targets[k] = v.to(device)
        else:
            batch_targets[k] = v
    return batch_features, batch_targets