from typing import List, Callable, Literal
import numpy as np
from tqdm import tqdm
from ..utils import logger


def build_Fa_picking(criterion: Literal["std_mean", "norm_std_max"]) -> Callable:
    def picking_func(y_preds, error_lower_bound: float, error_upper_bound: float, mode: Literal["single", "committee"]) -> List[int]:
        if mode == "single":
            Fas = y_preds["Fa"]
        elif mode == "committee":
            Fas = [np.stack(y_pred, axis=-1) for y_pred in zip(*[y_pred_single["Fa"] for y_pred_single in y_preds])]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        sample_size = len(Fas)
        if criterion == "std_mean":
            if mode == "single" and "Fa_var" in y_preds:
                # shape of Fa_var: (N, 3)
                est_error = np.array([np.mean(np.sqrt(Fa_var)) for Fa_var in y_preds["Fa_var"]])
            else:
                assert Fas[0].ndim == 3
                # shape of Fa should be: (N, 3, ensemble_size)
                est_error = np.array([np.mean(np.std(Fa, axis=-1, ddof=1)) for Fa in Fas])
        elif criterion == "norm_std_max":
            assert Fas[0].ndim == 3
            est_error = []
            for Fa in Fas:
                Fa_mean = np.mean(Fa, axis=-1, keepdims=True) # (N, 3, 1)
                Fa_norm_dev = np.linalg.norm(Fa - Fa_mean, axis=1) # (N, ensemble_size)
                Fa_norm_std = np.mean(Fa_norm_dev, axis=-1) # (N, )
                est_error.append(np.max(Fa_norm_std))
            est_error = np.array(est_error)
        upper_bool = est_error > error_upper_bound
        lower_bool = est_error < error_lower_bound
        picked = np.where(~(upper_bool | lower_bool))[0].tolist()
        upper = np.where(upper_bool)[0].tolist()
        lower = np.where(lower_bool)[0].tolist()
        logger.info(f"Estimated error: {np.mean(est_error):.4f} +/- {np.std(est_error):.4f}")
        logger.info(f"({len(picked)} / {sample_size}) picked, {len(lower)} lower, {len(upper)} upper!")
        return picked
    return picking_func


def random_picking(y_preds) -> List[int]:
    sample_size = len(y_preds[0]["Fa"])
    picked = list(range(sample_size))
    logger.info(f"({len(picked)} / {sample_size}) picked!")
    return picked


PICKING_REGISTER = {
    "max_Fa_norm_std": build_Fa_picking("norm_std_max"),
    "mean_Fa_std": build_Fa_picking("std_mean"),
    "random": random_picking
}