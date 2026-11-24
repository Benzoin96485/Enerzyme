"""Published FACTO parameter sets and pipeline defaults."""

from typing import Optional, Sequence

PAPER_PRESETS = {
    "S": {
        "scalar_dim": 384,
        "num_groups": 3,
        "tensor_channels": (128, 64, 32, 16),
        "num_radial": 32,
        "num_blocks": 5,
        "tensor_transport": False,
    },
    "M": {
        "scalar_dim": 768,
        "num_groups": 6,
        "tensor_channels": (128, 64, 32, 16),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": False,
    },
    "L": {
        "scalar_dim": 1024,
        "num_groups": 4,
        "tensor_channels": (256, 128, 64, 32),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": True,
    },
    "XL": {
        "scalar_dim": 2048,
        "num_groups": 4,
        "tensor_channels": (512, 256, 128, 64, 32, 16),
        "num_radial": 64,
        "num_blocks": 5,
        "tensor_transport": True,
    },
}

_CUSTOM_PRESETS = {"custom", "none", ""}


def resolve_facto_config(
    preset: Optional[str],
    *,
    scalar_dim: Optional[int] = None,
    num_groups: Optional[int] = None,
    tensor_channels: Optional[Sequence[int]] = None,
    num_radial: Optional[int] = None,
    num_blocks: Optional[int] = None,
    tensor_transport: Optional[bool] = None,
) -> dict:
    """Merge a paper preset with explicit overrides. ``None`` keeps the preset."""
    key = "" if preset is None else str(preset).strip()
    if key.lower() in _CUSTOM_PRESETS:
        config = {
            "scalar_dim": 32,
            "num_groups": 4,
            "tensor_channels": (8, 4),
            "num_radial": 8,
            "num_blocks": 2,
            "tensor_transport": False,
        }
    else:
        name = key.upper()
        if name not in PAPER_PRESETS:
            known = ", ".join(("S", "M", "L", "XL", "custom"))
            raise ValueError(f"Unknown FACTO preset {preset!r}. Expected one of: {known}.")
        config = dict(PAPER_PRESETS[name])
    overrides = {
        "scalar_dim": scalar_dim,
        "num_groups": num_groups,
        "tensor_channels": tensor_channels,
        "num_radial": num_radial,
        "num_blocks": num_blocks,
        "tensor_transport": tensor_transport,
    }
    for name, value in overrides.items():
        if value is not None:
            config[name] = value
    config["tensor_channels"] = tuple(int(width) for width in config["tensor_channels"])
    return config


def _validate_config(config: dict, cutoff_sr: float, r_short: float) -> None:
    scalar_dim = int(config["scalar_dim"])
    num_groups = int(config["num_groups"])
    channels = config["tensor_channels"]
    num_blocks = int(config["num_blocks"])
    num_radial = int(config["num_radial"])
    if scalar_dim < 1 or scalar_dim % num_groups != 0:
        raise ValueError(
            f"scalar_dim={scalar_dim} must be positive and divisible by num_groups={num_groups}"
        )
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if num_radial < 2:
        raise ValueError(f"num_radial must be >= 2, got {num_radial}")
    if not channels or any(int(width) < 1 for width in channels):
        raise ValueError(f"tensor_channels must be positive, got {channels}")
    if not (0.0 < float(r_short) < float(cutoff_sr)):
        raise ValueError(
            f"r_short must lie in (0, cutoff_sr), got Rs={r_short}, Rc={cutoff_sr}"
        )


DEFAULT_BUILD_PARAMS = {
    "max_Za": 94,
    "cutoff_sr": 6.0,
    "cutoff_fn": "facto",
    "Hartree_in_E": 1.0,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "Core", "params": {"preset": "S"}},
    {"name": "FACTOEnergyReadout"},
    {"name": "ZBLRepulsionEnergy"},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
