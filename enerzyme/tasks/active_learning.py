from typing import List
import numpy as np
from tqdm import tqdm
from ..utils import logger


def max_Fa_norm_std_picking(y_preds, lb, ub) -> List[int]:
    committee_size = len(y_preds)
    sample_size = len(y_preds[0]["Fa"])
    picked = []
    for i in tqdm(range(sample_size)):
        Fas = np.array([y_preds[j]["Fa"] for j in range(committee_size)])
        Fa_mean = np.mean(Fas, axis=0, keepdims=True)
        Fa_norm_dev = np.linalg.norm(Fas - Fa_mean, axis=2)
        Fa_norm_std = np.mean(Fa_norm_dev, axis=0)
        max_Fa_norm_std = np.max(Fa_norm_std)
        if lb < max_Fa_norm_std and max_Fa_norm_std < ub:
            picked.append(i)
    logger.info(f"({len(picked)} / {sample_size}) picked!")
    return picked
        
