"""Shared inputs and runners for ``bench_cores.py`` and ``snapshot_cores.py``.

Every architecture is built from its own default layer stack, so the numbers
track what a user gets from ``build_model(name)``.
"""
import copy
import os
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ARCHITECTURES = [
    "schnet", "physnet", "spookynet", "mace",
    "painn", "newtonnet", "et", "visnet", "freecg", "quinnet", "visnet_lsrm", "geodite",
    "leftnet", "alphanet",
    "reann", "equireann", "dpa3", "allscaip", "so3krates", "so3lr", "efa", "so3lr_efa",
    "botnet", "mace_polar", "nequip", "allegro", "equflash", "equflashv2", "nequix", "eqnorm",
    "equiformer",
    "escn", "esen", "equiformer_v2", "equiformer_v3", "dpa4", "tece", "e2former", "e2former_v2",
    "e2former_lsr",
    "dimenet", "dimenet++", "gemnet", "gemnet-q", "gemnet-oc", "m3gnet", "mattersim", "chgnet",
    "alignn", "spinconv",
    "tensornet", "tensornet2", "cace", "camp", "grace", "facto", "dpa4c", "tace",
]

TARGETS = ("E", "Fa", "Qa", "Ea", "M2", "E_ele_a", "E_disp_a", "nh_loss")


def random_molecules(seed: int, sizes: Iterable[int], elements=(1, 6, 7, 8), min_dist=0.95, density=0.08):
    rng = np.random.default_rng(seed)
    mols = []
    for n in sizes:
        box = (n / density) ** (1.0 / 3.0)
        pos: List[np.ndarray] = []
        while len(pos) < n:
            p = rng.random(3) * box
            if all(np.linalg.norm(p - q) >= min_dist for q in pos):
                pos.append(p)
        mols.append(dict(Ra=np.asarray(pos), Za=rng.choice(elements, size=n), N=n, Q=0.0, S=0.0))
    return mols


def make_batch(mols, dtype=torch.float64) -> Dict:
    from enerzyme.tasks.batch import _decorate_batch_input

    batch = [(dict(m), None) for m in mols]
    features, _ = _decorate_batch_input(batch, dtype=dtype, device=torch.device("cpu"), otf_graph=True)
    return features


def clone_batch(batch: Dict, dtype: torch.dtype) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.detach().clone()
            if v.is_floating_point():
                v = v.to(dtype)
        else:
            v = copy.deepcopy(v)
        out[k] = v
    return out


def build(arch: str, layer_params: Optional[List] = None, build_params: Optional[Dict] = None):
    from enerzyme.models.ff import build_model, get_ff_core

    _, default_build, default_layers = get_ff_core(arch)
    layers = copy.deepcopy(default_layers if layer_params is None else layer_params)
    params = copy.deepcopy(default_build if build_params is None else build_params)
    return build_model(arch, layers, params, verbose=0)


def forward(model, batch: Dict, dtype: torch.dtype, train: bool = False) -> Dict:
    model.train(train)
    inp = clone_batch(batch, dtype)
    inp["Ra"].requires_grad_(True)
    return model(inp)


def targets(out: Dict) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in out.items() if k in TARGETS and torch.is_tensor(v)}
