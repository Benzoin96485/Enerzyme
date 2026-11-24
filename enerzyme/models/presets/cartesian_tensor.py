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


def cace(
    dim_embedding: int = 3,
    cutoff_sr: float = 5.5,
    num_rbf: int = 6,
    preset: Optional[str] = "water",
    lmax: Optional[int] = None,
    max_nu: Optional[int] = None,
    n_radial: Optional[int] = None,
    num_message_passing: Optional[int] = None,
    messages: Optional[str] = None,
    envelope_p: Optional[int] = None,
    avg_num_neighbors: Optional[float] = None,
    readout_hidden: Optional[tuple] = None,
    shared_channels: Optional[bool] = None,
    shared_l: Optional[bool] = None,
    separate_receiver: Optional[bool] = None,
    bchi_layers: Optional[int] = None,
) -> ComposableCore:
    """Cartesian densities, shared invariant contraction and concatenated scalars."""
    from ..cace.blocks import CACEEmbed, CACEBlock
    from ..cace.interaction import parse_messages
    from ..cace.presets import resolve_cace_preset

    spec = resolve_cace_preset(
        preset,
        lmax=lmax,
        max_nu=max_nu,
        n_radial=n_radial,
        num_message_passing=num_message_passing,
        messages=messages,
        envelope_p=envelope_p,
        avg_num_neighbors=avg_num_neighbors,
        readout_hidden=readout_hidden,
        shared_channels=shared_channels,
        shared_l=shared_l,
        separate_receiver=separate_receiver,
        bchi_layers=bchi_layers,
    )
    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if int(num_rbf) < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if int(spec["lmax"]) < 0:
        raise ValueError(f"lmax must be >= 0, got {spec['lmax']}")
    if int(spec["n_radial"]) < 1:
        raise ValueError(f"n_radial must be >= 1, got {spec['n_radial']}")
    if int(spec["num_message_passing"]) < 0:
        raise ValueError(
            f"num_message_passing must be >= 0, got {spec['num_message_passing']}"
        )
    if float(spec["avg_num_neighbors"]) <= 0:
        raise ValueError(
            f"avg_num_neighbors must be positive, got {spec['avg_num_neighbors']}"
        )
    if int(spec["envelope_p"]) < 1:
        raise ValueError(f"envelope_p must be >= 1, got {spec['envelope_p']}")
    preset = None if preset is None else str(preset)
    dim_embedding = int(dim_embedding)
    cutoff_sr = float(cutoff_sr)
    num_rbf = int(num_rbf)
    lmax = int(spec["lmax"])
    max_nu = int(spec["max_nu"])
    n_radial = int(spec["n_radial"])
    num_message_passing = int(spec["num_message_passing"])
    messages = parse_messages(spec["messages"])
    envelope_p = int(spec["envelope_p"])
    avg_num_neighbors = float(spec["avg_num_neighbors"])
    readout_hidden = tuple(spec["readout_hidden"])
    shared_channels = bool(spec["shared_channels"])
    shared_l = bool(spec["shared_l"])
    separate_receiver = bool(spec["separate_receiver"])
    bchi_layers = int(spec["bchi_layers"])
    n_channels = dim_embedding * dim_embedding
    norm = avg_num_neighbors ** -0.5

    embed = CACEEmbed(dim_embedding, cutoff_sr, num_rbf, lmax, max_nu, n_radial, envelope_p, separate_receiver)
    n_invariants = embed.n_invariants
    blocks = [
        CACEBlock(
            messages, n_raw=num_rbf, n_radial=n_radial, n_angular=embed.n_angular, n_channels=n_channels,
            n_invariants=n_invariants, lmax=lmax, shells=embed.angular.shells, l_index=embed.angular.l_index,
            cutoff=cutoff_sr, norm=norm, shared_channels=shared_channels, shared_l=shared_l, bchi_layers=bchi_layers,
            mix=embed.mix, symmetrizer=embed.symmetrizer,
        )
        for _ in range(num_message_passing)
    ]
    width = (num_message_passing + 1) * n_radial * n_invariants * n_channels
    return ComposableCore(
        embed, blocks, FeatureHead("concat", include_embedding=True), GraphRequires(),
        is_cace_core=True, dim_feature_out=width, feature_irreps=f"{width}x0e",
        preset=preset, dim_embedding=dim_embedding, cutoff_sr=cutoff_sr,
        num_rbf=num_rbf, lmax=lmax, max_nu=max_nu,
        n_radial=n_radial, num_message_passing=num_message_passing, messages=messages,
        envelope_p=envelope_p, avg_num_neighbors=avg_num_neighbors, readout_hidden=readout_hidden,
        shared_channels=shared_channels, shared_l=shared_l, separate_receiver=separate_receiver,
        bchi_layers=bchi_layers, n_channels=n_channels, n_invariants=n_invariants,
    )


def grace(
    max_Za: int = 94,
    dim_embedding: int = 128,
    cutoff_sr: float = 6.0,
    preset: Optional[str] = "1L",
    avg_n_neigh: float = 1.0,
    rms_epsilon: float = 1e-6,
) -> ComposableCore:
    """Recursive ACE order blocks with species-dependent density projection."""
    from ..grace.blocks import GRACEBlock, GRACEEmbed, GRACEDensityHead
    from ..grace.coupling import DensityProjection
    from ..grace.presets import resolve_grace_preset

    name, config = resolve_grace_preset(preset)
    if dim_embedding < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if cutoff_sr <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if avg_n_neigh <= 0:
        raise ValueError(f"avg_n_neigh must be positive, got {avg_n_neigh}")
    preset = name
    max_Za = int(max_Za)
    dim_embedding = int(dim_embedding)
    cutoff_sr = float(cutoff_sr)
    basis_type = str(config["basis"])
    n_rad_base = int(config["n_rad_base"])
    cutoff_order = int(config["cutoff_order"])
    radial_hidden = int(config["radial_hidden"])
    radial_activation = str(config["radial_activation"])
    readout_head = str(config["readout"])
    readout_hidden = int(config["readout_hidden"])
    readout_activation = str(config["radial_activation"])
    recommended_embedding = int(config["embedding_size"])
    recommended_cutoff = float(config["cutoff"])
    fs_terms = config["fs_terms"]
    rms_epsilon = float(rms_epsilon)
    layer_specs = list(config["layers"])
    if readout_head == "linear_mlp":
        n_densities = int(config["n_mlp_dens"]) + 1
    elif readout_head == "finnis_sinclair":
        n_densities = len(fs_terms)
    else:
        raise ValueError(f"Unknown readout {readout_head!r} in preset {name}")
    if n_densities < 1:
        raise ValueError("GRACE must emit at least one density")
    num_layers = len(layer_specs)
    order_channels = [
        [order.channels for order in spec["orders"]] for spec in layer_specs
    ]
    order_lmax = [
        [order.lmax for order in spec["orders"]] for spec in layer_specs
    ]
    indicator_lmax = [spec["indicator_lmax"] for spec in layer_specs]
    dim_feature_out = n_densities
    feature_irreps = f"{n_densities}x0e"

    built = []
    for index, spec in enumerate(layer_specs):
        incoming = None
        if index > 0:
            incoming = layer_specs[index - 1]["indicator_lmax"]
            if incoming is None:
                raise ValueError(
                    f"GRACE layer {index - 1} must set indicator_lmax to feed the next layer"
                )
            incoming = int(incoming)
        outgoing_channels = None
        outgoing_lmax = None
        if index + 1 < len(layer_specs):
            if spec["indicator_lmax"] is None:
                raise ValueError(
                    f"GRACE layer {index} must set indicator_lmax to feed the next layer"
                )
            outgoing_lmax = int(spec["indicator_lmax"])
            outgoing_channels = int(layer_specs[index + 1]["orders"][0].channels)
        built.append(
            GRACEBlock(
                spec["orders"],
                n_basis=n_rad_base,
                cutoff=cutoff_sr,
                cutoff_order=cutoff_order,
                basis=basis_type,
                radial_hidden=radial_hidden,
                radial_activation=radial_activation,
                embedding_dim=dim_embedding,
                incoming_lmax=incoming,
                outgoing_lmax=outgoing_lmax,
                indicator_channels=int(spec["indicator_channels"]),
                indicator_out=outgoing_channels,
                n_species=max_Za + 1,
                epsilon=rms_epsilon,
            )
        )
    flat_channels = [
        channels for layer_channels in order_channels for channels in layer_channels
    ]
    density = DensityProjection(
        flat_channels, n_densities, max_Za + 1
    )
    max_l = max(layer.lmax for layer in built)
    return ComposableCore(
        GRACEEmbed(dim_embedding, avg_n_neigh, max_l), built, GRACEDensityHead(density),
        GraphRequires(rbf=False, optional_cutoff_values=False),
        is_grace_core=True, num_blocks=1,
        preset=preset, max_Za=max_Za, dim_embedding=dim_embedding,
        cutoff_sr=cutoff_sr, basis_type=basis_type, n_rad_base=n_rad_base,
        cutoff_order=cutoff_order, radial_hidden=radial_hidden, radial_activation=radial_activation,
        readout_head=readout_head, readout_hidden=readout_hidden, readout_activation=readout_activation,
        recommended_embedding=recommended_embedding, recommended_cutoff=recommended_cutoff, fs_terms=fs_terms,
        rms_epsilon=rms_epsilon, n_densities=n_densities, num_layers=num_layers,
        order_channels=order_channels, order_lmax=order_lmax, indicator_lmax=indicator_lmax,
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps, max_l=max_l,
    )
