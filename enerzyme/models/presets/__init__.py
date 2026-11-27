"""Architecture registry.

Each architecture name (and its aliases) maps to a Core factory, its default
``build_params`` and ``layers``, and any architecture-specific losses. Factories
are preset functions with explicit keyword arguments that return a
:class:`~enerzyme.models.mp.core.ComposableCore` (or, for wrapped external
models, any ``BaseFFCore``). ``build_layer`` reads the factory signature, so a
preset is used exactly like a Core class in a YAML layer stack.

Built-in entries are lazy ``"module:attr"`` strings under ``enerzyme.models``;
nothing is imported until an architecture is requested.
``register_architecture`` adds an entry at import time (presets, plugins).
"""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Ref = Union[str, Any]

_PACKAGE = "enerzyme.models"


def _load(ref: Ref) -> Any:
    if not isinstance(ref, str):
        return ref
    module, _, attr = ref.partition(":")
    mod = importlib.import_module(module if not module.startswith(".") else _PACKAGE + module)
    return getattr(mod, attr) if attr else mod


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    core: Ref
    build_params: Ref
    layer_params: Ref
    losses: Optional[Ref] = None
    aliases: Tuple[str, ...] = ()
    family: str = ""

    def resolve(self) -> Tuple[Callable[..., Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        core = _load(self.core)
        build = _load(self.build_params)
        layers = _load(self.layer_params)
        losses = _load(self.losses) if self.losses is not None else {}
        return core, build, layers, losses


_REGISTRY: Dict[str, ArchitectureSpec] = {}
_ALIASES: Dict[str, str] = {}


def _key(name: str) -> str:
    return name.lower()


def add_architecture(spec: ArchitectureSpec, replace: bool = False) -> ArchitectureSpec:
    key = _key(spec.name)
    if key in _REGISTRY and not replace:
        raise KeyError(f"architecture {spec.name!r} is already registered")
    _REGISTRY[key] = spec
    for alias in (spec.name,) + tuple(spec.aliases):
        _ALIASES[_key(alias)] = key
    return spec


def register_architecture(
    name: str,
    *,
    build_params: Ref,
    layer_params: Ref,
    aliases: Tuple[str, ...] = (),
    losses: Optional[Ref] = None,
    family: str = "",
    replace: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a preset function as architecture ``name``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        add_architecture(
            ArchitectureSpec(name=name, core=fn, build_params=build_params, layer_params=layer_params,
                             losses=losses, aliases=tuple(aliases), family=family),
            replace=replace,
        )
        fn.architecture = name
        return fn

    return decorate


def get_architecture(name: str) -> ArchitectureSpec:
    key = _ALIASES.get(_key(name))
    if key is None:
        raise NotImplementedError(f"Architecture {name} not implemented")
    return _REGISTRY[key]


def available_architectures(family: Optional[str] = None) -> List[str]:
    return sorted(s.name for s in _REGISTRY.values() if family is None or s.family == family)


def default_params(name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Deep copies of the default ``build_params`` and ``layers``."""
    _, build, layers, _ = get_architecture(name).resolve()
    return copy.deepcopy(build), copy.deepcopy(layers)


from . import builtin  # noqa: E402,F401  (populates the lazy table)
