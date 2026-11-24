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
        positional_fields=('atom_embedding', 'Za', 'Dij_sr', 'vij_sr', 'idx_i_sr', 'idx_j_sr'),
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
        positional_fields=('atom_embedding', 'rbf', 'Dij_sr', 'vij_sr', 'idx_i_sr', 'idx_j_sr', 'cutoff_values_sr'),
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
        positional_fields=('atom_embedding', 'Za', 'Dij_sr', 'vij_sr', 'idx_i_sr', 'idx_j_sr'),
    )


def facto(
    max_Za: int = 94,
    cutoff_sr: float = 6.0,
    r_short: float = 1.0,
    preset: Optional[str] = "S",
    scalar_dim: Optional[int] = None,
    num_groups: Optional[int] = None,
    tensor_channels: Optional[Sequence[int]] = None,
    num_radial: Optional[int] = None,
    num_blocks: Optional[int] = None,
    tensor_transport: Optional[bool] = None,
    residual_scale: Optional[float] = None,
    rms_epsilon: float = 1e-6,
) -> ComposableCore:
    """Hierarchical atom-centered tensor updates with matching isolated-atom taps."""
    from ..facto.blocks import FACTOBlock, FACTOEmbed, FACTOHead
    from ..facto.defaults import resolve_facto_config, _validate_config
    from ..facto.interaction import RESIDUAL_SCALE

    config = resolve_facto_config(
        preset,
        scalar_dim=scalar_dim,
        num_groups=num_groups,
        tensor_channels=tensor_channels,
        num_radial=num_radial,
        num_blocks=num_blocks,
        tensor_transport=tensor_transport,
    )
    _validate_config(config, cutoff_sr, r_short)
    max_Za = int(max_Za)
    cutoff_sr = float(cutoff_sr)
    r_short = float(r_short)
    preset = None if preset is None else str(preset)
    scalar_dim = int(config["scalar_dim"])
    num_groups = int(config["num_groups"])
    tensor_channels = tuple(config["tensor_channels"])
    num_radial = int(config["num_radial"])
    num_blocks = int(config["num_blocks"])
    tensor_transport = bool(config["tensor_transport"])
    residual_scale = (
        RESIDUAL_SCALE if residual_scale is None else float(residual_scale)
    )
    rms_epsilon = float(rms_epsilon)
    lmax = len(tensor_channels)
    num_output_blocks = num_blocks
    dim_feature_out = scalar_dim
    feature_irreps = f"{scalar_dim}x0e"

    embed = FACTOEmbed(max_Za, scalar_dim, num_radial, tensor_channels, cutoff_sr, r_short)
    blocks = [
        FACTOBlock(
            scalar_dim, tensor_channels, num_groups, update_tensors=index + 1 < num_blocks,
            tensor_transport=tensor_transport, residual_scale=residual_scale, rms_epsilon=rms_epsilon,
        )
        for index in range(num_blocks)
    ]
    return ComposableCore(
        embed, blocks, FACTOHead(), GraphRequires(rbf=False, optional_cutoff_values=False),
        node_field="Za", emits_isolated_atom_feature=True,
        max_Za=max_Za, cutoff_sr=cutoff_sr, r_short=r_short,
        preset=preset, scalar_dim=scalar_dim, num_groups=num_groups,
        tensor_channels=tensor_channels, num_radial=num_radial, num_blocks=num_blocks,
        tensor_transport=tensor_transport, residual_scale=residual_scale, rms_epsilon=rms_epsilon,
        lmax=lmax, num_output_blocks=num_output_blocks, dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        positional_fields=('Za', 'Dij_sr', 'vij_sr', 'idx_i_sr', 'idx_j_sr'),
    )


def reann(
    dim_embedding: int = 32,
    max_Za: int = 94,
    cutoff_sr: float = 5.0,
    lmax: int = 2,
    nwave: int = 8,
    n_orbit: Optional[int] = None,
    num_iterations: int = 1,
    nblock: int = 1,
    hidden_dims: Sequence[int] = (128, 128),
    activation: str = "relu_like",
    layer_norm: bool = True,
    dropout: float = 0.0,
    elemental: bool = False,
    atomic_numbers: Optional[Sequence[int]] = None,
    coeff_init: float = 0.1,
) -> ComposableCore:
    return _orbital_core(
        dim_embedding, max_Za, cutoff_sr, lmax, nwave, n_orbit, num_iterations,
        nblock, hidden_dims, activation, layer_norm, dropout, elemental, atomic_numbers, coeff_init,
        pass_orbitals=False,
    )


def equireann(
    dim_embedding: int = 32,
    max_Za: int = 94,
    cutoff_sr: float = 6.0,
    lmax: int = 2,
    nwave: int = 20,
    n_orbit: Optional[int] = None,
    num_iterations: int = 2,
    nblock: int = 1,
    hidden_dims: Sequence[int] = (32, 32),
    activation: str = "tanh_like",
    layer_norm: bool = False,
    dropout: float = 0.0,
    elemental: bool = False,
    atomic_numbers: Optional[Sequence[int]] = None,
    coeff_init: float = 0.1,
) -> ComposableCore:
    return _orbital_core(
        dim_embedding, max_Za, cutoff_sr, lmax, nwave, n_orbit, num_iterations,
        nblock, hidden_dims, activation, layer_norm, dropout, elemental, atomic_numbers, coeff_init,
        pass_orbitals=True,
    )


def _orbital_core(
    dim_embedding, max_Za, cutoff_sr, lmax, nwave, n_orbit, num_iterations, nblock, hidden_dims,
    activation, layer_norm, dropout, elemental, atomic_numbers, coeff_init, *, pass_orbitals,
):
    from functools import partial
    from ..reann.blocks import OrbitalEmbed, OrbitalUpdate, build_orbital_stack
    from ..reann.density import default_n_orbit
    from ..reann.network import REANNMLP, resolve_reann_activation

    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(max_Za) < 1:
        raise ValueError(f"max_Za must be >= 1, got {max_Za}")
    if float(cutoff_sr) <= 0.0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if int(lmax) < 0 or int(lmax) > 6:
        raise ValueError(f"lmax must lie in 0..6, got {lmax}")
    if int(nwave) < 1:
        raise ValueError(f"nwave must be >= 1, got {nwave}")
    if int(num_iterations) < 0:
        raise ValueError(f"num_iterations must be >= 0, got {num_iterations}")
    if int(nblock) < 0:
        raise ValueError(f"nblock must be >= 0, got {nblock}")
    orbit = default_n_orbit(int(nwave), int(lmax)) if n_orbit is None else int(n_orbit)
    if orbit < 1:
        raise ValueError(f"n_orbit must be >= 1, got {orbit}")
    if elemental:
        if not atomic_numbers:
            raise ValueError(
                "elemental=True requires atomic_numbers, the species that own a network"
            )
        numbers = [int(number) for number in atomic_numbers]
        if any(number < 0 or number > int(max_Za) for number in numbers):
            raise ValueError(
                f"atomic_numbers must lie in 0..max_Za ({max_Za}), got {numbers}"
            )
    else:
        numbers = None
    dim_embedding = int(dim_embedding)
    max_Za = int(max_Za)
    cutoff_sr = float(cutoff_sr)
    lmax = int(lmax)
    nwave = int(nwave)
    n_orbit = orbit
    num_iterations = int(num_iterations)
    nblock = int(nblock)
    hidden_dims = tuple(int(width) for width in hidden_dims)
    activation = resolve_reann_activation(activation)
    layer_norm = bool(layer_norm)
    dropout = float(dropout)
    elemental = bool(elemental)
    atomic_numbers = None if numbers is None else tuple(numbers)
    coeff_init = float(coeff_init)
    dim_feature_out = n_orbit
    feature_irreps = f"{n_orbit}x0e"
    embed = OrbitalEmbed(dim_embedding, max_Za, cutoff_sr, lmax, nwave, n_orbit, coeff_init)
    network = partial(REANNMLP, n_in=n_orbit, n_out=nwave, hidden_dims=hidden_dims,
                      nblock=nblock, activation=activation, layer_norm=layer_norm, dropout=dropout)
    blocks = [
        OrbitalUpdate(nwave, atomic_numbers, network, i + 1 if pass_orbitals else None)
        for i in range(num_iterations)
    ]
    if pass_orbitals:
        embed.enable_orbital_messages(num_iterations)
    return ComposableCore(
        embed, blocks, FeatureHead(), GraphRequires(rbf=False, optional_cutoff_values=False),
        builder=build_orbital_stack,
        dim_embedding=dim_embedding, max_Za=max_Za, cutoff_sr=cutoff_sr,
        lmax=lmax, nwave=nwave, n_orbit=n_orbit,
        num_iterations=num_iterations, nblock=nblock, hidden_dims=hidden_dims,
        activation=activation, layer_norm=layer_norm, dropout=dropout,
        elemental=elemental, atomic_numbers=atomic_numbers, coeff_init=coeff_init,
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps,
        positional_fields=('atom_embedding', 'Za', 'idx_i_sr', 'idx_j_sr', 'vij_sr', 'Dij_sr'),
    )


def dpa4c(
    dim_embedding: int,
    channels: Optional[int] = None,
    lmax: int = 2,
    radial_modes: int = 0,
    n_radial: int = 16,
    basis_type: str = "bessel",
    cutoff_sr: float = 6.0,
) -> ComposableCore:
    from ..dpa4c.blocks import DPA4CBlock, DPA4CHead, build_descriptor_stack
    from ..mp.core import DelegatingCore, FieldEmbed

    block = DPA4CBlock(dim_embedding, channels, lmax, radial_modes, n_radial, basis_type, cutoff_sr)
    head = DPA4CHead(block.dim_feature_out, block.channels)
    return DelegatingCore(
        FieldEmbed(), [block], head, GraphRequires(distances=False, rbf=False, optional_cutoff_values=False),
        builder=build_descriptor_stack, dim_embedding=dim_embedding, channels=block.channels, lmax=lmax,
        radial_modes=radial_modes, n_radial=n_radial, cutoff_sr=cutoff_sr, degree_channels=block.degree_channels,
        dim_feature_out=block.dim_feature_out, feature_irreps=block.feature_irreps,
        delegates={
            "raw_descriptor": "blocks.0.raw_descriptor",
            "radial": "blocks.0.radial", "pair_film": "blocks.0.pair_film",
            "fit_output_calibration": "head.fit_output_calibration",
            "reset_output_calibration": "head.reset_output_calibration",
            "calib_scale": "head.calib_scale", "calib_mean": "head.calib_mean",
        },
        positional_fields=("atom_embedding", "idx_i_sr", "idx_j_sr", "vij_sr"),
    )
