# Copyright (c) EquiformerV2 authors (Liao, Wood, Das, Smidt, ICLR 2024).
# Ported from https://github.com/atomicarchitects/equiformer_v2 (MIT License).

"""EquiformerV2 package — Core exports are lazy to avoid import cycles."""

__all__ = [
    "EquiformerV2Core",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]


def __getattr__(name: str):
    if name in {"EquiformerV2Core", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"}:
        from .core import EquiformerV2Core, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

        mapping = {
            "EquiformerV2Core": EquiformerV2Core,
            "DEFAULT_BUILD_PARAMS": DEFAULT_BUILD_PARAMS,
            "DEFAULT_LAYER_PARAMS": DEFAULT_LAYER_PARAMS,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
