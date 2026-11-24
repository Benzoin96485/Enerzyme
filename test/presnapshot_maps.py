"""Map pre-refactor snapshot state_dicts onto rebuilt Cores.

``KEY_MAPS[arch]`` is a list of ``(regex, replacement)`` pairs applied in order
to every old key. The legacy architectures (SchNet, PhysNet, SpookyNet, MACE)
go through the ``main``-branch checkpoint converter. Any other architecture
without an entry first tries an identical key set, then pairs parameters positionally (both sides in ``named_parameters``
order, shapes must agree). Buffers are copied by name when present on both
sides; a buffer only the new model has must be a deterministic constant.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

_ORBITAL_KEYS = [
    (r"^(rs|inta|hyper|coeff)\b", r"embed.\1"),
    (r"^updates\.", "blocks."),
]
KEY_MAPS: Dict[str, List[Tuple[str, str]]] = {
    "reann": _ORBITAL_KEYS,
    "equireann": _ORBITAL_KEYS,
}


def _rename(state: Dict[str, torch.Tensor], rules: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        for pattern, repl in rules:
            key = re.sub(pattern, repl, key)
        if key is not None and key != "":
            out[key] = value
    return out


def remap_snapshot(arch: str, old_state: Dict[str, torch.Tensor], old_param_names: List[str],
                   model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    new_state = model.state_dict()
    from enerzyme.models.compat.main_branch import KEY_RULES, convert_state_dict

    legacy = getattr(model, "architecture", None)
    if legacy in KEY_RULES and set(old_state) != set(new_state):
        return dict(convert_state_dict(legacy, old_state)[0])
    if arch in KEY_MAPS:
        renamed = _rename(old_state, KEY_MAPS[arch])
        merged = dict(new_state)
        for key, value in renamed.items():
            if key in merged:
                merged[key] = value
        missing = [k for k, _ in model.named_parameters() if k not in renamed]
        if missing:
            raise KeyError(f"{arch}: parameters without a snapshot source: {missing[:8]}")
        return merged
    if set(old_state) == set(new_state):
        return dict(old_state)
    new_params = [k for k, _ in model.named_parameters()]
    old_params = [k for k in old_param_names if k in old_state]
    if len(new_params) != len(old_params):
        raise KeyError(
            f"{arch}: {len(old_params)} snapshot parameters vs {len(new_params)} model parameters; "
            "add a KEY_MAPS entry"
        )
    merged = dict(new_state)
    for old_key, new_key in zip(old_params, new_params):
        old_value = old_state[old_key]
        if tuple(old_value.shape) != tuple(new_state[new_key].shape):
            raise ValueError(
                f"{arch}: positional match {old_key} {tuple(old_value.shape)} -> "
                f"{new_key} {tuple(new_state[new_key].shape)}; add a KEY_MAPS entry"
            )
        merged[new_key] = old_value
    param_set = set(new_params)
    for key in new_state:
        if key not in param_set and key in old_state and old_state[key].shape == new_state[key].shape:
            merged[key] = old_state[key]
    return merged
