from typing import List
import numpy as np
from tqdm import tqdm
from ..utils import logger


def max_Fa_norm_std_picking(y_preds, lb, ub) -> List[int]:
    committee_size = len(y_preds)
    sample_size = len(y_preds[0]["Fa"])
    picked = []
    lower = []
    upper = []
    for i in tqdm(range(sample_size)):
        Fas = np.array([y_preds[j]["Fa"][i] for j in range(committee_size)])
        Fa_mean = np.mean(Fas, axis=0, keepdims=True)
        Fa_norm_dev = np.linalg.norm(Fas - Fa_mean, axis=2)
        Fa_norm_std = np.mean(Fa_norm_dev, axis=0)
        max_Fa_norm_std = np.max(Fa_norm_std)
        if max_Fa_norm_std < lb:
            lower.append(i)
        elif ub < max_Fa_norm_std:
            upper.append(i)
        else:
            picked.append(i)
    logger.info(f"({len(picked)} / {sample_size}) picked, {len(lower)} lower, {len(upper)} upper!")
    return picked
        
