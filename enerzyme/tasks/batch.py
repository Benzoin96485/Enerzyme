from typing import Iterable, Tuple, Dict, Any, List, Union, Optional
import torch
import numpy as np
from torch import Tensor
from ..data import is_atomic, is_int, is_idx, requires_grad, is_target, is_target_uq, full_neighbor_list


def _decorate_batch_input(batch: Iterable[Tuple[Dict[str, Tensor], Dict[str, Tensor]]], dtype: torch.dtype, device: torch.device) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    features, targets = zip(*batch)
    batch_features = dict()
    batch_targets = dict()
    
    for k in features[0]:
        if is_atomic(k):
            batch_features[k] = torch.tensor(
                np.concatenate([feature[k][:feature["N"]] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
                requires_grad=requires_grad(k)
            ).to(device)
        elif not is_idx(k):
            batch_features[k] = torch.tensor(
                np.array([feature[k] for feature in features]), 
                dtype=torch.long if is_int(k) else dtype,
            ).to(device)

    batch_idx_i = []
    batch_idx_j = []
    batch_seg = []
    count = 0

    for i, feature in enumerate(features):
        if "idx_i" in feature:
            batch_idx_i.append(feature["idx_i"][:feature["N_pair"]] + count)
            batch_idx_j.append(feature["idx_j"][:feature["N_pair"]] + count)
        else:
            idx_i, idx_j = full_neighbor_list(feature["N"])
            batch_idx_i.append(idx_i + count)
            batch_idx_j.append(idx_j + count)
        batch_seg.append(np.full(feature["N"], i, dtype=int))
        count += feature["N"]
    batch_features["N"] = [feature["N"] for feature in features]
    batch_features["batch_seg"] = torch.tensor(np.concatenate(batch_seg), dtype=torch.long).to(device)
    batch_features["idx_i"] = torch.tensor(np.concatenate(batch_idx_i), dtype=torch.long).to(device)
    batch_features["idx_j"] = torch.tensor(np.concatenate(batch_idx_j), dtype=torch.long).to(device)

    if targets[0] is not None:
        for k in targets[0]:
            if is_atomic(k): 
                batch_targets[k] = torch.tensor(
                    np.concatenate([target[k][:features[i]["N"]] for i, target in enumerate(targets)]), 
                    dtype=torch.long if is_int(k) else dtype
                ).to(device)
            else:
                batch_targets[k] = torch.tensor(
                    np.array([target[k] for target in targets]), 
                    dtype=torch.long if is_int(k) else dtype,
                ).to(device)
    
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
    y_truth["Za"] = y_pred["Za"]

    return y_pred, (y_truth if y_truth else None)