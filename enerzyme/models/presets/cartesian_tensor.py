"""Cartesian tensor and atomic cluster expansion presets."""

from __future__ import annotations

from typing import Optional, Sequence

from ..mp.core import ComposableCore
from ..mp.graph import GraphRequires
from ..mp.head import FeatureHead


def camp(
    dim_embedding: int = 33,
    max_Za: int = 94,
    cutoff_sr: float = 5.0,
    preset: Optional[str] = "water",
    max_v: Optional[int] = None,
    num_layers: Optional[int] = None,
    max_chebyshev_degree: Optional[int] = None,
    envelope_p: Optional[int] = None,
    radial_mlp_hidden_layers: Optional[Sequence[int]] = None,
    avg_num_neighbors: Optional[float] = None,
    output_mlp_hidden_layers: Optional[Sequence[int]] = None,
    atomic_numbers: Optional[Sequence[int]] = None,
) -> ComposableCore:
    """Atomic moments of species-pair Chebyshev expansions and a tapped scalar head."""
    from ..camp.moment import CAMPEmbed, CAMPLayer
    from ..camp.presets import resolve_camp_preset
    from ..camp.radial import resolve_hidden_layers, species_lookup

    spec = resolve_camp_preset(
        preset,
        max_v=max_v,
        num_layers=num_layers,
        max_chebyshev_degree=max_chebyshev_degree,
        envelope_p=envelope_p,
        radial_mlp_hidden_layers=radial_mlp_hidden_layers,
        avg_num_neighbors=avg_num_neighbors,
        output_mlp_hidden_layers=output_mlp_hidden_layers,
    )
    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if int(spec["max_v"]) < 0 or int(spec["max_v"]) > 6:
        raise ValueError(f"max_v must lie in 0..6, got {spec['max_v']}")
    if int(spec["num_layers"]) < 1:
        raise ValueError(f"num_layers must be >= 1, got {spec['num_layers']}")
    if int(spec["max_chebyshev_degree"]) < 0:
        raise ValueError(
            f"max_chebyshev_degree must be >= 0, got {spec['max_chebyshev_degree']}"
        )
    if int(spec["envelope_p"]) < 1:
        raise ValueError(f"envelope_p must be >= 1, got {spec['envelope_p']}")
    if float(spec["avg_num_neighbors"]) <= 0:
        raise ValueError(
            f"avg_num_neighbors must be positive, got {spec['avg_num_neighbors']}"
        )
    lookup, n_species = species_lookup(max_Za, atomic_numbers)
    width = int(dim_embedding)
    rank = int(spec["max_v"])
    depth = int(spec["num_layers"])
    degree = int(spec["max_chebyshev_degree"])
    hidden = resolve_hidden_layers(spec["radial_mlp_hidden_layers"], width)
    blocks = [
        CAMPLayer(
            n_channels=width, n_species=n_species, max_v=rank,
            max_in_rank=0 if index == 0 else rank,
            max_out_rank=0 if index == depth - 1 else rank,
            residual=index > 0, avg_num_neighbors=spec["avg_num_neighbors"],
            degree=degree, cutoff=cutoff_sr, envelope_p=spec["envelope_p"], radial_hidden=hidden,
        )
        for index in range(depth)
    ]
    return ComposableCore(
        CAMPEmbed(width, lookup, rank), blocks, FeatureHead("layers"),
        GraphRequires(rbf=False, optional_cutoff_values=False),
        preset=preset, is_camp_core=True, dim_embedding=width, max_Za=max_Za,
        n_species=n_species, cutoff_sr=cutoff_sr, max_v=rank, num_layers=depth,
        max_chebyshev_degree=degree, envelope_p=spec["envelope_p"], avg_num_neighbors=spec["avg_num_neighbors"],
        output_mlp_hidden_layers=spec["output_mlp_hidden_layers"],
        atomic_numbers=None if atomic_numbers is None else list(atomic_numbers),
        dim_feature_out=width, num_output_blocks=depth, feature_irreps=f"{width}x0e",
    )
