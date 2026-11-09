"""Published GRACE widths.

``1L`` / ``2L`` presets follow Lysogorskiy, Bochkarev, and Drautz,
npj Comput. Mater. (2025), Supplementary Table S1: 6 Å cutoff, 128-d
chemical embedding, Chebyshev radial basis, product order 4, and the listed
angular schedule. Product-channel counts that the table does not print are
taken from the public preset table (``prod_func_n_max``): ``1L`` and ``1L-M``
keep one channel group per radial function, and ``1L-L`` doubles the product
channels, matching the table's doubled invariant count.

``1L-latest`` and ``2L-latest`` are the current GRACE_1LAYER / GRACE_2LAYER
defaults documented with the reference package. ``FS`` is the Finnis–Sinclair
student used for distillation (linear radial, spherical Bessel, powers
``1, 1/2, 2, 3/4``). ``custom`` and ``custom-2L`` are tiny stacks for tests.

These presets are hyperparameters, not a checkpoint layout. Enerzyme does not
load TensorPotential weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .coupling import OrderSpec


def _orders(
    channels: Sequence[int],
    lmax: Sequence[int],
    lmax_in: Optional[Sequence[Optional[int]]] = None,
) -> List[OrderSpec]:
    if len(channels) != len(lmax):
        raise ValueError("channels and lmax must have one entry per product order")
    if len(channels) < 1 or len(channels) > 4:
        raise ValueError(f"need 1 to 4 product orders, got {len(channels)}")
    limits: Sequence[Optional[int]]
    if lmax_in is None:
        limits = [None] * len(channels)
    else:
        if len(lmax_in) != len(channels):
            raise ValueError("lmax_in must cover every product order")
        limits = lmax_in
    return [
        OrderSpec(int(width), int(degree), None if limit is None else int(limit))
        for width, degree, limit in zip(channels, lmax, limits)
    ]


def _layer(
    channels: Sequence[int],
    lmax: Sequence[int],
    *,
    lmax_in: Optional[Sequence[Optional[int]]] = None,
    indicator_lmax: Optional[int] = None,
    indicator_channels: int = 16,
) -> dict:
    return {
        "orders": _orders(channels, lmax, lmax_in),
        "indicator_lmax": indicator_lmax,
        "indicator_channels": int(indicator_channels),
    }


def _graph(
    *,
    basis: str,
    n_rad_base: int,
    cutoff_order: int,
    radial_hidden: int,
    radial_activation: str,
    layers: Sequence[dict],
    readout: str,
    n_mlp_dens: int = 0,
    readout_hidden: int = 64,
    embedding_size: int = 128,
    fs_terms: Optional[Sequence[Sequence[float]]] = None,
    cutoff: float = 6.0,
) -> dict:
    return {
        "basis": basis,
        "n_rad_base": int(n_rad_base),
        "cutoff_order": int(cutoff_order),
        "radial_hidden": int(radial_hidden),
        "radial_activation": radial_activation,
        "layers": list(layers),
        "readout": readout,
        "n_mlp_dens": int(n_mlp_dens),
        "readout_hidden": int(readout_hidden),
        "embedding_size": int(embedding_size),
        "fs_terms": None if fs_terms is None else tuple(tuple(term) for term in fs_terms),
        "cutoff": float(cutoff),
    }


def _ace_layer(
    n_rad: int,
    prod: int,
    lmax: Sequence[int],
    *,
    indicator_lmax: Optional[int] = None,
    indicator_channels: int = 16,
    lmax_in: Optional[Sequence[Optional[int]]] = None,
) -> dict:
    channels = [int(n_rad)] + [int(prod)] * (len(lmax) - 1)
    return _layer(
        channels,
        lmax,
        lmax_in=lmax_in,
        indicator_lmax=indicator_lmax,
        indicator_channels=indicator_channels,
    )


_FS_TERMS = ((1.0, 1.0), (1.0, 0.5), (1.0, 2.0), (1.0, 0.75))

# Table S1 angular schedules, with product channels filled in as above.
_PAPER_1L = _graph(
    basis="chebyshev",
    n_rad_base=8,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[_ace_layer(32, 32, (4, 4, 0, 0))],
    readout="linear_mlp",
    n_mlp_dens=12,
    embedding_size=128,
)
_PAPER_1L_M = _graph(
    basis="chebyshev",
    n_rad_base=10,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[_ace_layer(32, 32, (4, 4, 0, 0))],
    readout="linear_mlp",
    n_mlp_dens=16,
    embedding_size=128,
)
_PAPER_1L_L = _graph(
    basis="chebyshev",
    n_rad_base=10,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[_ace_layer(32, 64, (4, 4, 0, 0))],
    readout="linear_mlp",
    n_mlp_dens=16,
    embedding_size=128,
)
_PAPER_2L = _graph(
    basis="chebyshev",
    n_rad_base=8,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[
        _ace_layer(32, 32, (4, 4, 1, 0), indicator_lmax=1),
        _ace_layer(32, 32, (4, 4, 0, 0)),
    ],
    readout="linear_mlp",
    n_mlp_dens=12,
    embedding_size=128,
)
_PAPER_2L_M = _graph(
    basis="chebyshev",
    n_rad_base=10,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[
        _ace_layer(42, 42, (4, 4, 1, 1), indicator_lmax=1),
        _ace_layer(32, 64, (3, 3, 0, 0)),
    ],
    readout="linear_mlp",
    n_mlp_dens=16,
    embedding_size=128,
)
_PAPER_2L_L = _graph(
    basis="chebyshev",
    n_rad_base=10,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[
        _ace_layer(42, 42, (4, 4, 3, 1), indicator_lmax=3),
        _ace_layer(32, 64, (3, 3, 0, 0)),
    ],
    readout="linear_mlp",
    n_mlp_dens=16,
    embedding_size=128,
)
_LATEST_1L = _graph(
    basis="chebyshev",
    n_rad_base=10,
    cutoff_order=16,
    radial_hidden=64,
    radial_activation="silu",
    layers=[_ace_layer(32, 64, (4, 4, 0, 0))],
    readout="linear_mlp",
    n_mlp_dens=16,
    embedding_size=128,
)
_FS = _graph(
    basis="sbessel",
    n_rad_base=20,
    cutoff_order=5,
    radial_hidden=0,
    radial_activation="silu",
    layers=[
        _layer(
            (20, 15, 10, 5),
            (5, 3, 0, 0),
            lmax_in=(5, 5, 4, 3),
        )
    ],
    readout="finnis_sinclair",
    embedding_size=64,
    fs_terms=_FS_TERMS,
    cutoff=7.0,
    readout_hidden=0,
)
_CUSTOM = _graph(
    basis="chebyshev",
    n_rad_base=4,
    cutoff_order=4,
    radial_hidden=8,
    radial_activation="silu",
    layers=[_ace_layer(4, 4, (2, 1, 0), indicator_channels=4)],
    readout="linear_mlp",
    n_mlp_dens=2,
    readout_hidden=8,
    embedding_size=8,
)
_CUSTOM_2L = _graph(
    basis="chebyshev",
    n_rad_base=4,
    cutoff_order=4,
    radial_hidden=8,
    radial_activation="silu",
    layers=[
        _ace_layer(4, 4, (2, 1, 0), indicator_lmax=1, indicator_channels=4),
        _ace_layer(4, 4, (1, 0), indicator_channels=4),
    ],
    readout="linear_mlp",
    n_mlp_dens=2,
    readout_hidden=8,
    embedding_size=8,
)

PAPER_PRESETS: Dict[str, dict] = {
    "1L": _PAPER_1L,
    "1L-M": _PAPER_1L_M,
    "1L-L": _PAPER_1L_L,
    "2L": _PAPER_2L,
    "2L-M": _PAPER_2L_M,
    "2L-L": _PAPER_2L_L,
    "1L-latest": _LATEST_1L,
    "2L-latest": _PAPER_2L_L,
    "FS": _FS,
    "custom": _CUSTOM,
    "custom-2L": _CUSTOM_2L,
}

_ALIASES = {name.lower(): name for name in PAPER_PRESETS}


def resolve_grace_preset(preset: Optional[str]) -> tuple:
    """Return ``(canonical_name, config)``. ``config`` layers hold :class:`OrderSpec`."""
    key = "1L" if preset is None else str(preset).strip()
    canonical = _ALIASES.get(key.lower())
    if canonical is None:
        known = ", ".join(PAPER_PRESETS)
        raise ValueError(f"Unknown GRACE preset {preset!r}. Expected one of: {known}.")
    return canonical, PAPER_PRESETS[canonical]
