"""Published CAMP widths (Wen et al., npj Comput. Mater. 2025, Table S5).

The reference hyperparameter ``max_u`` is the highest channel index. Every
Cartesian tensor therefore has ``max_u + 1`` channels, and that width is
``dim_embedding``. The grid in the paper is ``u ∈ {16, 32, 48, 64}``.
Chebyshev degree 8 and the degree-6 polynomial envelope are shared by those
fits. ``avg_num_neighbors`` is the training-set mean and is not in the table.
"""

from __future__ import annotations

from typing import Any, Optional


PUBLISHED_CHEBYSHEV_DEGREE = 8
PUBLISHED_ENVELOPE_P = 6

# cutoff_sr is recorded for the YAML. The Core receives the stack's cutoff_sr
# rather than overwriting it, so a preset and RangeSeparation stay on one radius.
PAPER_PRESETS = {
    "water": {"max_u": 32, "max_v": 2, "num_layers": 3, "cutoff_sr": 5.0},
    "lips-10": {"max_u": 16, "max_v": 3, "num_layers": 2, "cutoff_sr": 6.0},
    "lips-100": {"max_u": 16, "max_v": 3, "num_layers": 2, "cutoff_sr": 6.0},
    "lips-1000": {"max_u": 48, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
    "lips-2500": {"max_u": 64, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
    "lips-19000": {"max_u": 48, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
    "aspirin": {"max_u": 48, "max_v": 3, "num_layers": 2, "cutoff_sr": 6.0},
    "ethanol": {"max_u": 48, "max_v": 3, "num_layers": 2, "cutoff_sr": 6.0},
    "malonaldehyde": {"max_u": 32, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
    "naphthalene": {"max_u": 32, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
    "salicylic": {"max_u": 48, "max_v": 3, "num_layers": 2, "cutoff_sr": 5.0},
    "toluene": {"max_u": 48, "max_v": 3, "num_layers": 2, "cutoff_sr": 6.0},
    "uracil": {"max_u": 48, "max_v": 3, "num_layers": 3, "cutoff_sr": 5.0},
    "graphene": {"max_u": 48, "max_v": 3, "num_layers": 3, "cutoff_sr": 6.0},
}

for _spec in PAPER_PRESETS.values():
    _spec["dim_embedding"] = int(_spec["max_u"]) + 1
    _spec["max_chebyshev_degree"] = PUBLISHED_CHEBYSHEV_DEGREE
    _spec["envelope_p"] = PUBLISHED_ENVELOPE_P


def resolve_camp_preset(
    preset: Optional[str],
    max_v: Optional[int] = None,
    num_layers: Optional[int] = None,
    max_chebyshev_degree: Optional[int] = None,
    envelope_p: Optional[int] = None,
    radial_mlp_hidden_layers: Any = None,
    avg_num_neighbors: Optional[float] = None,
    output_mlp_hidden_layers: Any = None,
) -> dict:
    """Fill omitted CAMP widths from a Table S5 preset or the small custom stack."""
    key = "custom" if preset is None else str(preset)
    if key in {"custom", "none"}:
        spec = {
            "max_v": 2,
            "num_layers": 2,
            "max_chebyshev_degree": PUBLISHED_CHEBYSHEV_DEGREE,
            "envelope_p": PUBLISHED_ENVELOPE_P,
            "radial_mlp_hidden_layers": 2,
            "avg_num_neighbors": 1.0,
            "output_mlp_hidden_layers": 2,
        }
    elif key in PAPER_PRESETS:
        published = PAPER_PRESETS[key]
        spec = {
            "max_v": int(published["max_v"]),
            "num_layers": int(published["num_layers"]),
            "max_chebyshev_degree": PUBLISHED_CHEBYSHEV_DEGREE,
            "envelope_p": PUBLISHED_ENVELOPE_P,
            "radial_mlp_hidden_layers": 2,
            "avg_num_neighbors": 1.0,
            "output_mlp_hidden_layers": 2,
        }
    else:
        known = ", ".join(["custom", *PAPER_PRESETS])
        raise ValueError(f"Unknown CAMP preset {preset!r}. Known presets: {known}")

    overrides = {
        "max_v": max_v,
        "num_layers": num_layers,
        "max_chebyshev_degree": max_chebyshev_degree,
        "envelope_p": envelope_p,
        "radial_mlp_hidden_layers": radial_mlp_hidden_layers,
        "avg_num_neighbors": avg_num_neighbors,
        "output_mlp_hidden_layers": output_mlp_hidden_layers,
    }
    for name, value in overrides.items():
        if value is not None:
            spec[name] = value
    return spec
