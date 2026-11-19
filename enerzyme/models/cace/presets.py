"""Published CACE widths.

The numbers follow the training scripts in BingqingCheng/cacefit, which are
the configurations behind Cheng, npj Comput. Mater. 10, 157 (2024). Edge
channels are ``dim_embedding ** 2`` because the edge type is the tensor
product of the receiver and sender embeddings. ``dim_embedding``, ``cutoff``,
and ``num_rbf`` are recorded here so a stack can be built from the preset;
the Core reads the embedding width and the raw radial width from the layer
stack's build parameters.
"""

from __future__ import annotations

from typing import Dict


_COMMON = {
    "shared_channels": False,
    "shared_l": False,
    "separate_receiver": True,
    "bchi_layers": 1,
}


PAPER_PRESETS: Dict[str, Dict] = {
    "water": {
        **_COMMON,
        "lmax": 3,
        "max_nu": 3,
        "n_radial": 12,
        "num_message_passing": 1,
        "messages": "b",
        "envelope_p": 6,
        "avg_num_neighbors": 1.0,
        "readout_hidden": (32, 16),
        "dim_embedding": 3,
        "cutoff": 5.5,
        "num_rbf": 6,
    },
    "water-local": {
        **_COMMON,
        "lmax": 3,
        "max_nu": 3,
        "n_radial": 12,
        "num_message_passing": 0,
        "messages": "b",
        "envelope_p": 6,
        "avg_num_neighbors": 1.0,
        "readout_hidden": (32, 16),
        "dim_embedding": 3,
        "cutoff": 5.5,
        "num_rbf": 6,
    },
    "ethanol": {
        **_COMMON,
        "lmax": 4,
        "max_nu": 3,
        "n_radial": 12,
        "num_message_passing": 1,
        "messages": "full",
        "envelope_p": 5,
        "avg_num_neighbors": 10.0,
        "readout_hidden": (32, 16),
        "dim_embedding": 4,
        "cutoff": 4.0,
        "num_rbf": 6,
    },
    "bpa": {
        **_COMMON,
        "lmax": 4,
        "max_nu": 3,
        "n_radial": 12,
        "num_message_passing": 1,
        "messages": "full",
        "envelope_p": 2,
        "avg_num_neighbors": 10.0,
        "readout_hidden": (32, 16),
        "dim_embedding": 4,
        "cutoff": 6.0,
        "num_rbf": 8,
    },
    "hea": {
        **_COMMON,
        "lmax": 3,
        "max_nu": 3,
        "n_radial": 8,
        "num_message_passing": 1,
        "messages": "b",
        "envelope_p": 6,
        "avg_num_neighbors": 10.0,
        "readout_hidden": (24, 12),
        "dim_embedding": 4,
        "cutoff": 4.5,
        "num_rbf": 6,
    },
    "custom": {
        **_COMMON,
        "lmax": 2,
        "max_nu": 2,
        "n_radial": 4,
        "num_message_passing": 0,
        "messages": "b",
        "envelope_p": 6,
        "avg_num_neighbors": 1.0,
        "readout_hidden": (8,),
        "dim_embedding": 4,
        "cutoff": 5.0,
        "num_rbf": 4,
    },
}


_CORE_KEYS = (
    "lmax",
    "max_nu",
    "n_radial",
    "num_message_passing",
    "messages",
    "envelope_p",
    "avg_num_neighbors",
    "readout_hidden",
    "shared_channels",
    "shared_l",
    "separate_receiver",
    "bchi_layers",
)

_DEFAULTS = {
    "num_message_passing": 0,
    "messages": "b",
    "envelope_p": 6,
    "avg_num_neighbors": 1.0,
    "readout_hidden": (32, 16),
    "shared_channels": False,
    "shared_l": False,
    "separate_receiver": True,
    "bchi_layers": 1,
}


def resolve_cace_preset(preset, **overrides) -> Dict:
    """Merge a named preset with explicit Core arguments.

    An explicit argument replaces the preset value. ``None`` keeps the preset
    (or the small default when no preset is named).
    """
    if preset is None:
        spec = {}
    else:
        key = str(preset)
        if key not in PAPER_PRESETS:
            known = ", ".join(sorted(PAPER_PRESETS))
            raise ValueError(f"Unknown CACE preset {preset!r}. Known presets: {known}")
        spec = {name: PAPER_PRESETS[key][name] for name in _CORE_KEYS}
    for name, value in overrides.items():
        if value is not None:
            spec[name] = value
    for name, value in _DEFAULTS.items():
        spec.setdefault(name, value)
    missing = [name for name in ("lmax", "max_nu", "n_radial") if name not in spec]
    if missing:
        raise ValueError(
            "CACE needs lmax, max_nu, and n_radial when preset is omitted; "
            f"missing {missing}"
        )
    spec["readout_hidden"] = tuple(int(width) for width in spec["readout_hidden"])
    return spec
