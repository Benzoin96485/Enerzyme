"""Load ``main``-branch checkpoints of SchNet, PhysNet, SpookyNet and MACE.

``main`` (and the pre-refactor devel tree) stored these Cores with their own
module trees. The presets in :mod:`enerzyme.models.presets` rebuild them on
:class:`~enerzyme.models.mp.core.ComposableCore` with the same parameters
under new names. This module maps the old names onto the new tree and
reorders EMA shadow parameters and optimizer state, which both follow
``model.parameters()`` order.

Old Cores also exposed pre/post layers as attributes (``atom_embedding``,
``radial_basis_function``, ...), so their state dicts hold duplicate entries.
Duplicates are dropped after checking they equal the canonical
``pre_sequence`` / ``post_sequence`` entry.

``main`` YAML model sections need no translation: the legacy presets keep
``main``'s Core parameter names, layer names resolve through the layer
registry or :mod:`.layers`, and ``build_layer`` ignores keys a preset does
not take.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ...utils import logger

CHECKPOINT_FORMAT = 2
FORMAT_KEY = "enerzyme_format"

KEY_RULES: Dict[str, List[Tuple[str, str]]] = {
    "schnet": [
        (r"^interactions\.(\d+)\.mlp\.", r"blocks.\1.message.filter."),
        (r"^interactions\.(\d+)\.conv\.nn\.", r"blocks.\1.message.filter."),
        (r"^interactions\.(\d+)\.conv\.lin1\.", r"blocks.\1.message.pre."),
        (r"^interactions\.(\d+)\.conv\.lin2\.", r"blocks.\1.update.net.0."),
        (r"^interactions\.(\d+)\.lin\.", r"blocks.\1.update.net.2."),
        (r"^lin1\.", r"head.readout.0."),
        (r"^lin2\.", r"head.readout.2."),
    ],
    "physnet": [
        (r"^interaction_block\.(\d+)\.", r"blocks.\1."),
        (r"^output_block\.(\d+)\.", r"head.readouts.\1."),
    ],
    "spookynet": [
        (r"^interaction\.(\d+)\.", r"blocks.\1."),
        (r"^output\.", r"head.readout."),
    ],
    "mace": [
        (r"^interactions\.(\d+)\.", r"blocks.\1.interaction."),
        (r"^products\.(\d+)\.", r"blocks.\1.product."),
        (r"^readouts\.(\d+)\.", r"head.readouts.\1."),
    ],
}

# Attribute aliases of pre/post layers on the old Cores.
ALIAS_PREFIXES = (
    "calculate_distance", "range_separation", "atom_embedding", "charge_embedding", "spin_embedding",
    "radial_basis_function", "charge_conservation", "gather_embedding", "spherical_harmonics",
)
LEGACY_ARCHITECTURES = tuple(KEY_RULES)

# Readout attributes that ``main``'s Muon setup (``enerzyme.tasks.optimizer``)
# put in the auxiliary AdamW group together with pre/post layers, gains and
# biases.
MAIN_MUON_HEADS: Dict[str, set] = {
    "schnet": {"lin1", "lin2"},
    "physnet": {"output_block"},
    "spookynet": {"output"},
    "mace": {"readouts"},
}


def _architecture(model: nn.Module) -> Optional[str]:
    name = getattr(model, "architecture", None)
    return name if name in KEY_RULES else None


def _rename(key: str, rules: Sequence[Tuple[str, str]]) -> str:
    for pattern, repl in rules:
        if re.match(pattern, key):
            return re.sub(pattern, repl, key, count=1)
    return key


def _resolve_alias(prefix: str, entries: Dict[str, Tensor], kept: Mapping[str, Tensor]) -> Dict[str, str]:
    """Find the ``pre_sequence.N`` / ``post_sequence.N`` layer an alias points to."""
    suffixes = [key[len(prefix) + 1:] for key in entries]
    candidates = sorted({
        key.rsplit("." + suffixes[0], 1)[0]
        for key in kept
        if key.endswith("." + suffixes[0]) and key.startswith(("pre_sequence.", "post_sequence."))
    })
    for owner in candidates:
        mapping = {f"{prefix}.{s}": f"{owner}.{s}" for s in suffixes}
        if all(target in kept and torch.equal(kept[target], entries[old]) for old, target in mapping.items()):
            return mapping
    raise ValueError(f"alias {prefix!r} does not match any pre/post layer in the checkpoint")


def convert_state_dict(architecture: str, state_dict: Mapping[str, Tensor]) -> Tuple["OrderedDict[str, Tensor]", Dict[str, str]]:
    """Rename a legacy state dict for ``architecture``.

    Returns the new state dict and the full old→new key map (aliases map to
    their canonical key).
    """
    rules = KEY_RULES[architecture]
    out: "OrderedDict[str, Tensor]" = OrderedDict()
    key_map: Dict[str, str] = {}
    aliases: Dict[str, Dict[str, Tensor]] = {}
    for key, value in state_dict.items():
        head = key.split(".", 1)[0]
        if head in ALIAS_PREFIXES:
            aliases.setdefault(head, {})[key] = value
            continue
        new_key = _rename(key, rules)
        if new_key in out:
            if not torch.equal(out[new_key], value):
                raise ValueError(f"{key} and an earlier entry both map to {new_key} but differ")
        else:
            out[new_key] = value
        key_map[key] = new_key
    for prefix, entries in aliases.items():
        key_map.update(_resolve_alias(prefix, entries, out))
    return out, key_map


def old_parameter_order(state_dict: Mapping[str, Tensor], key_map: Mapping[str, str],
                        parameter_names: Sequence[str]) -> List[Tuple[str, str]]:
    """``(old name, new name)`` of the old model's parameters, in the old ``model.parameters()`` order.

    ``state_dict`` and ``named_parameters`` walk modules in the same preorder,
    and shared modules appear in ``named_parameters`` only at their first
    position, so the first occurrence of each parameter in the state dict
    gives the old order and the old name.
    """
    wanted = set(parameter_names)
    order: List[Tuple[str, str]] = []
    seen = set()
    for key in state_dict:
        new_key = key_map.get(key, key)
        if new_key in wanted and new_key not in seen:
            seen.add(new_key)
            order.append((key, new_key))
    return order


def _permutation(old_order: Sequence[str], new_names: Sequence[str]) -> List[int]:
    position = {name: i for i, name in enumerate(old_order)}
    missing = [name for name in new_names if name not in position]
    if missing or len(old_order) != len(new_names):
        raise ValueError(f"cannot match parameters between checkpoint and model: missing {missing[:5]}")
    return [position[name] for name in new_names]


def main_optimizer_groups(architecture: str, old_order: Sequence[Tuple[str, str]],
                          shapes: Mapping[str, Tuple[int, ...]], num_groups: int) -> Optional[List[List[str]]]:
    """New names of the parameters in each optimizer group as ``main`` built them.

    One group is ``optimizer(model.parameters())``; two groups are ``main``'s
    Muon split (hidden weights, then gains, biases, pre/post layers and
    readouts). Other layouts return ``None``.
    """
    if num_groups == 1:
        return [[new for _, new in old_order]]
    if num_groups != 2:
        return None
    hidden, gains, auxiliary = [], [], []
    for old, new in old_order:
        prefix = old.split(".", 1)[0]
        if prefix in ("pre_sequence", "post_sequence") or prefix in MAIN_MUON_HEADS[architecture]:
            auxiliary.append(new)
        elif len(shapes[new]) >= 2:
            hidden.append(new)
        else:
            gains.append(new)
    return [hidden, gains + auxiliary]


def convert_ema_state_dict(ema_state: Dict[str, Any], permutation: Sequence[int]) -> Dict[str, Any]:
    ema_state = dict(ema_state)
    for key in ("shadow_params", "collected_params"):
        params = ema_state.get(key)
        if params:
            if len(params) != len(permutation):
                raise ValueError(f"EMA {key} has {len(params)} entries, model has {len(permutation)} parameters")
            ema_state[key] = [params[i] for i in permutation]
    return ema_state


def convert_optimizer_state_dict(opt_state: Dict[str, Any], old_groups: Optional[Sequence[Sequence[str]]],
                                 new_groups: Sequence[Sequence[str]],
                                 shapes: Mapping[str, Tuple[int, ...]]) -> Optional[Dict[str, Any]]:
    """Re-index optimizer state from ``old_groups`` to ``new_groups`` (lists of parameter names).

    Every group must hold the same parameters in both layouts; per-parameter
    state tensors must match the parameter shapes. ``None`` otherwise.
    """
    saved = opt_state.get("param_groups", [])
    if old_groups is None or not len(saved) == len(old_groups) == len(new_groups):
        return None
    state = opt_state.get("state", {})
    old_index: Dict[str, Any] = {}
    for group, names in zip(saved, old_groups):
        if len(group["params"]) != len(names):
            return None
        old_index.update(zip(names, group["params"]))
    new_state: Dict[int, Any] = {}
    param_groups = []
    position = 0
    for group, old_names, new_names in zip(saved, old_groups, new_groups):
        if set(old_names) != set(new_names):
            return None
        indices = []
        for name in new_names:
            entry = state.get(old_index[name])
            if entry is not None:
                if any(torch.is_tensor(v) and v.dim() > 0 and tuple(v.shape) != shapes[name] for v in entry.values()):
                    return None
                new_state[position] = entry
            indices.append(position)
            position += 1
        param_groups.append({**group, "params": indices})
    return {"state": new_state, "param_groups": param_groups}


def needs_upgrade(model: nn.Module, loaded_info: Mapping[str, Any]) -> bool:
    if loaded_info.get(FORMAT_KEY, 1) >= CHECKPOINT_FORMAT or _architecture(model) is None:
        return False
    state = loaded_info.get("model_state_dict")
    return state is not None and set(state) != set(model.state_dict())


def upgrade_checkpoint(model: nn.Module, loaded_info: Dict[str, Any],
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Convert a legacy checkpoint dict in the trainer format for ``model``.

    Rewrites ``model_state_dict``, ``ema_state_dict`` and
    ``optimizer_state_dict``. Optimizer state is re-indexed for the group
    layout of ``optimizer`` (one group over ``model.parameters()`` when
    ``None``); state that cannot be matched is dropped with a warning. EMA
    state must match.
    """
    if not needs_upgrade(model, loaded_info):
        return loaded_info
    architecture = _architecture(model)
    old_state = loaded_info["model_state_dict"]
    new_state, key_map = convert_state_dict(architecture, old_state)
    loaded_info = dict(loaded_info)
    loaded_info["model_state_dict"] = new_state
    named = list(model.named_parameters())
    names = [name for name, _ in named]
    old_order = old_parameter_order(old_state, key_map, names)
    try:
        permutation = _permutation([new for _, new in old_order], names)
    except ValueError:
        permutation = None
    if "ema_state_dict" in loaded_info:
        if permutation is None:
            raise ValueError(f"EMA state of this {architecture} checkpoint does not match the model parameters")
        loaded_info["ema_state_dict"] = convert_ema_state_dict(loaded_info["ema_state_dict"], permutation)
    if "optimizer_state_dict" in loaded_info:
        converted = None
        if permutation is not None:
            shapes = {name: tuple(p.shape) for name, p in named}
            if optimizer is None:
                new_groups = [names]
            else:
                name_of = {id(p): name for name, p in named}
                new_groups = [[name_of[id(p)] for p in group["params"]] for group in optimizer.param_groups]
            saved = loaded_info["optimizer_state_dict"]
            old_groups = main_optimizer_groups(architecture, old_order, shapes, len(saved.get("param_groups", [])))
            converted = convert_optimizer_state_dict(saved, old_groups, new_groups, shapes)
        if converted is None:
            logger.warning(f"dropping optimizer state of the legacy {architecture} checkpoint: layout not mappable")
            loaded_info.pop("optimizer_state_dict")
        else:
            loaded_info["optimizer_state_dict"] = converted
    loaded_info[FORMAT_KEY] = CHECKPOINT_FORMAT
    logger.info(f"converted a legacy {architecture} checkpoint to format {CHECKPOINT_FORMAT}")
    return loaded_info
