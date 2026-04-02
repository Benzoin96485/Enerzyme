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


def _decorate_pyg_batch_input(
    batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]],
    dtype: torch.dtype,
    device: torch.device,
    otf_graph: bool = True,
    generator_config: Optional[Dict[str, Any]] = None,
    generator_training: bool = True,
) -> Tuple[Batch, Batch]:
    features, targets, data_keys = zip(*batch)
    feature_list = []
    for feature in features:
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
            data_dict["N"] = feature["N"]
        if "idx_i" in feature and "idx_j" in feature:
            data_dict["idx_i"] = torch.tensor(feature["idx_i"], dtype=torch.long)
            data_dict["idx_j"] = torch.tensor(feature["idx_j"], dtype=torch.long)
            feature_list.append(Data(edge_index=torch.stack([data_dict["idx_i"], data_dict["idx_j"]], dim=0), num_nodes=feature["N"], **data_dict))
        elif otf_graph:
            idx_i, idx_j = full_neighbor_list(feature["N"])
            data_dict["idx_i"] = torch.tensor(idx_i, dtype=torch.long)
            data_dict["idx_j"] = torch.tensor(idx_j, dtype=torch.long)
            feature_list.append(Data(edge_index=torch.stack([data_dict["idx_i"], data_dict["idx_j"]], dim=0), num_nodes=feature["N"], **data_dict))
    batch_features = Batch.from_data_list(feature_list)
    for k, v in batch_features.items():
        if requires_grad(k):
            v.requires_grad_(True)

    target_list = []
    for i, target in enumerate(targets):
        data_dict = dict()
        for k, v in target.items():
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
    batch_seg = []
    count = 0

    built_graph = False
    for i, feature in enumerate(features):
        if "idx_i" in feature:
            batch_idx_i.append(feature["idx_i"][:feature["N_pair"]] + count)
            batch_idx_j.append(feature["idx_j"][:feature["N_pair"]] + count)
            built_graph = True
        elif otf_graph and not built_graph:
            idx_i, idx_j = full_neighbor_list(feature["N"])
            batch_idx_i.append(idx_i + count)
            batch_idx_j.append(idx_j + count)
            built_graph = True
        batch_seg.append(np.full(feature["N"], i, dtype=int))
        count += feature["N"]
    batch_features["N"] = [feature["N"] for feature in features]
    batch_features["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long)
    if built_graph:
        batch_features["idx_i"] = torch.tensor(np.concatenate(batch_idx_i), dtype=torch.long)
        batch_features["idx_j"] = torch.tensor(np.concatenate(batch_idx_j), dtype=torch.long)

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