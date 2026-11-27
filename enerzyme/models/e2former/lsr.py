"""E2Former-LSR Core (Wang et al., arXiv:2601.03774).

Short-range E2Former + fragment bipartite long-range attention + late fusion.
Adapted from UBio-MolFM tag E2Former-LSR (MIT):
https://github.com/IQuestLab/UBio-MolFM
"""

from __future__ import annotations

from copy import deepcopy

from .defaults import (
    DEFAULT_BUILD_PARAMS as _E2_BUILD,
    DEFAULT_LAYER_PARAMS as _E2_LAYERS,
)

DEFAULT_BUILD_PARAMS = {
    **_E2_BUILD,
    "cutoff_lr": 15.0,
}

DEFAULT_LAYER_PARAMS = deepcopy(_E2_LAYERS)
# Paper-ish short depth; long layers / fragment knobs live on Core params.
for _layer in DEFAULT_LAYER_PARAMS:
    if _layer.get("name") == "Core":
        _layer["params"] = {
            **_layer["params"],
            "num_layers": 4,
            "long_layers": 2,
            "long_max_neighbors": 64,
            "fragment_mode": "kmeans",
            "min_nodes_per_group": 24,
            "kmeans_random_state": 0,
            "cutoff_lr": 15.0,
        }
        break
