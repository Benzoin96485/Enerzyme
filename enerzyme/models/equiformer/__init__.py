# Copyright (c) Equiformer authors (Liao & Smidt, ICLR 2023).
# Ported from https://github.com/atomicarchitects/equiformer (MIT License).

# Keep this import light so ``layers`` can re-export EquiformerNodeEmbedding
# without circularly loading EquiformerCore.
from .node_embedding_layer import EquiformerNodeEmbedding

__all__ = [
    "EquiformerCore",
    "EquiformerNodeEmbedding",
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]


def __getattr__(name: str):
    if name in {"EquiformerCore", "DEFAULT_BUILD_PARAMS", "DEFAULT_LAYER_PARAMS"}:
        from .core import EquiformerCore, DEFAULT_BUILD_PARAMS, DEFAULT_LAYER_PARAMS

        mapping = {
            "EquiformerCore": EquiformerCore,
            "DEFAULT_BUILD_PARAMS": DEFAULT_BUILD_PARAMS,
            "DEFAULT_LAYER_PARAMS": DEFAULT_LAYER_PARAMS,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
