"""Presets for directional, line-graph and repflow message passing."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

from torch import nn

from ..mp.block import FnBlock
from ..mp.core import ComposableCore
from ..mp.graph import GraphRequires, QuadrupletRequest, ReverseEdgeRequest, TripletRequest
from ..mp.head import FeatureHead
from ..mp.message.triplet_graph import GemNetBlock, GemNetHead


HiddenSpec = Union[int, Sequence[int], None]


def _positive(name: str, value: int) -> int:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _nonnegative(name: str, value: int) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def dpa3(
    dim_embedding: int,
    n_dim: Optional[int] = None,
    e_dim: int = 32,
    a_dim: int = 32,
    nlayers: int = 2,
    e_rcut: float = 6.0,
    e_rcut_smth: float = 5.0,
    e_sel: int = 32,
    a_rcut: float = 4.0,
    a_rcut_smth: float = 3.5,
    a_sel: int = 16,
    axis_neuron: int = 4,
    n_multi_edge_message: int = 1,
    update_angle: bool = True,
    update_style: str = "res_residual",
    update_residual: float = 0.1,
    update_residual_init: str = "const",
    a_compress_rate: int = 0,
    a_compress_e_rate: int = 1,
    a_compress_use_split: bool = False,
    sequential_update: bool = False,
    smooth_edge_update: bool = True,
    edge_init_use_dist: bool = False,
    switch: str = "polynomial",
    switch_decay: float = 20.0,
    dynamic_normalize: bool = False,
    sel_reduce_factor: float = 10.0,
    fix_stat_std: float = 0.3,
    env_protection: float = 0.0,
    activation: str = "silu",
) -> ComposableCore:
    from ..dpa3.stages import (
        DPA3Embed,
        build_stack,
        repflow_step,
    )
    from ..dpa3.interaction import RepFlowBlock
    from ..mp.nn.activation import resolve_scalar_activation

    n_dim = dim_embedding if n_dim is None else _positive("n_dim", n_dim)
    if int(n_dim) != int(dim_embedding):
        raise ValueError(
            f"n_dim ({n_dim}) must equal dim_embedding ({dim_embedding}); "
            "the atom embedding is the initial node state"
        )
    e_dim = _positive("e_dim", e_dim)
    a_dim = _positive("a_dim", a_dim)
    nlayers = _positive("nlayers", nlayers)
    e_sel = _positive("e_sel", e_sel)
    a_sel = _positive("a_sel", a_sel)
    axis_neuron = _positive("axis_neuron", axis_neuron)
    n_multi_edge_message = _positive("n_multi_edge_message", n_multi_edge_message)
    if axis_neuron > n_dim or axis_neuron > e_dim:
        raise ValueError(f"axis_neuron ({axis_neuron}) must be <= n_dim ({n_dim}) and e_dim ({e_dim})")
    if float(e_rcut) <= 0.0 or float(a_rcut) <= 0.0:
        raise ValueError("e_rcut and a_rcut must be positive")
    if float(a_rcut) > float(e_rcut) + 1e-8:
        raise ValueError(f"a_rcut ({a_rcut}) must be <= e_rcut ({e_rcut})")
    if not 0.0 <= float(e_rcut_smth) < float(e_rcut):
        raise ValueError(f"e_rcut_smth ({e_rcut_smth}) must lie in [0, e_rcut) with e_rcut={e_rcut}")
    if not 0.0 <= float(a_rcut_smth) < float(a_rcut):
        raise ValueError(f"a_rcut_smth ({a_rcut_smth}) must lie in [0, a_rcut) with a_rcut={a_rcut}")
    if update_style not in ("res_avg", "res_incr", "res_residual"):
        raise ValueError(f"unknown update_style {update_style!r}")
    if update_residual_init not in ("const", "norm"):
        raise ValueError(f"update_residual_init must be 'const' or 'norm', got {update_residual_init!r}")
    if sequential_update and update_style != "res_residual":
        raise ValueError("sequential_update only supports update_style='res_residual'")
    if sequential_update and not update_angle:
        raise ValueError("sequential_update requires update_angle=True")
    if int(a_compress_rate) < 0:
        raise ValueError(f"a_compress_rate must be >= 0, got {a_compress_rate}")
    if int(a_compress_e_rate) < 1:
        raise ValueError(f"a_compress_e_rate must be >= 1, got {a_compress_e_rate}")
    if int(a_compress_rate) > 0:
        if a_dim % int(a_compress_rate) != 0:
            raise ValueError(f"a_dim ({a_dim}) must be divisible by a_compress_rate ({a_compress_rate})")
        if (a_dim * int(a_compress_e_rate)) % (2 * int(a_compress_rate)) != 0:
            raise ValueError("a_dim * a_compress_e_rate must be divisible by 2 * a_compress_rate")
        n_compress = a_dim // int(a_compress_rate)
        e_compress = a_dim // (2 * int(a_compress_rate)) * int(a_compress_e_rate)
        if a_compress_use_split and (n_dim < n_compress or e_dim < e_compress):
            raise ValueError("split compression needs n_dim >= " f"{n_compress} and e_dim >= {e_compress}")
    if float(sel_reduce_factor) <= 0.0:
        raise ValueError(f"sel_reduce_factor must be positive, got {sel_reduce_factor}")
    if float(fix_stat_std) < 0.0:
        raise ValueError(f"fix_stat_std must be >= 0, got {fix_stat_std}")
    if float(env_protection) < 0.0:
        raise ValueError(f"env_protection must be >= 0, got {env_protection}")
    if switch not in ("polynomial", "exponential"):
        raise ValueError(f"switch must be 'polynomial' or 'exponential', got {switch!r}")
    if switch == "exponential" and (float(e_rcut_smth) <= 0.0 or float(a_rcut_smth) <= 0.0):
        raise ValueError("exponential switch needs a positive smoothing distance")
    if float(switch_decay) <= 0.0:
        raise ValueError(f"switch_decay must be positive, got {switch_decay}")

    dim_embedding = int(dim_embedding)
    n_dim = int(n_dim)
    e_rcut = float(e_rcut)
    e_rcut_smth = float(e_rcut_smth)
    a_rcut = float(a_rcut)
    a_rcut_smth = float(a_rcut_smth)
    update_angle = bool(update_angle)
    edge_init_use_dist = bool(edge_init_use_dist)
    switch_decay = float(switch_decay)
    env_protection = float(env_protection)
    stat_std = float(fix_stat_std) if float(fix_stat_std) > 0.0 else 1.0
    dim_feature_out = n_dim
    feature_irreps = f"{n_dim}x0e"
    act = resolve_scalar_activation(activation)

    if dynamic_normalize:
        neighbor_scale = e_sel / float(sel_reduce_factor)
        angular_scale = a_sel / float(sel_reduce_factor)
    else:
        neighbor_scale = float(e_sel)
        angular_scale = float(a_sel)

    edge_embed = nn.Linear(1, e_dim)
    angle_embed = nn.Linear(1, a_dim, bias=False) if update_angle else None
    block_kwargs = dict(
        n_dim=n_dim,
        e_dim=e_dim,
        a_dim=a_dim,
        axis_neuron=axis_neuron,
        n_multi_edge_message=n_multi_edge_message,
        update_angle=update_angle,
        update_style=update_style,
        update_residual=float(update_residual),
        update_residual_init=update_residual_init,
        activation=act,
        a_compress_rate=int(a_compress_rate),
        a_compress_e_rate=int(a_compress_e_rate),
        a_compress_use_split=bool(a_compress_use_split),
        sequential_update=bool(sequential_update),
        smooth_edge_update=bool(smooth_edge_update),
        neighbor_scale=neighbor_scale,
        angular_scale=angular_scale,
    )
    blocks = [RepFlowBlock(**block_kwargs) for _ in range(nlayers)]

    embed = DPA3Embed(
        n_dim,
        a_dim,
        e_rcut,
        e_rcut_smth,
        a_rcut,
        a_rcut_smth,
        update_angle,
        edge_init_use_dist,
        switch,
        switch_decay,
        env_protection,
        stat_std,
        act,
        edge_embed,
        angle_embed,
    )
    blocks = [FnBlock(block, repflow_step) for block in blocks]
    head = FeatureHead()
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(rbf=False, optional_cutoff_values=False, topology=(
            TripletRequest("receiver", edge_cutoff=e_rcut, cutoff=a_rcut, strict=True, include_self=True),
        ) if update_angle else ()),
        builder=build_stack,
        positional_fields=("atom_embedding", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture="dpa3",
        # Fusing both endpoint reductions can drop a RepFlow force-gradient term.
        _cpu_force_training_option="cpp.max_horizontal_fusion_size",
        _cpu_force_inference_option=("cpp.simdlen", "cpp.max_horizontal_fusion_size"),
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        n_dim=n_dim,
        e_dim=e_dim,
        a_dim=a_dim,
        nlayers=nlayers,
        e_rcut=e_rcut,
    )


def alignn(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    num_angle_rbf: int = 40,
    embedding_features: int = 64,
    hidden_features: Optional[int] = None,
    alignn_layers: int = 2,
    gcn_layers: int = 2,
    norm: Optional[str] = "layer",
    residual: bool = True,
    update_order: str = "code",
    backtracking: bool = True,
    multiply_envelope: bool = False,
    envelope_exponent: int = 5,
    gate_eps: float = 1e-6,
) -> ComposableCore:
    from ..alignn.stages import (
        ALIGNNEmbed,
        build_stack,
        alignn_step,
        gcn_step,
        _norm_name,
    )
    from ..alignn.interaction import ALIGNNConv, EdgeGatedGraphConv, FeatureMLP, embedding_mlp
    from ..layers.rbf import alignn_gaussian_terms

    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 2:
        raise ValueError(f"num_rbf must be >= 2, got {num_rbf}")
    if int(num_angle_rbf) < 2:
        raise ValueError(f"num_angle_rbf must be >= 2, got {num_angle_rbf}")
    if int(embedding_features) < 1:
        raise ValueError(f"embedding_features must be >= 1, got {embedding_features}")
    if int(alignn_layers) < 0:
        raise ValueError(f"alignn_layers must be >= 0, got {alignn_layers}")
    if int(gcn_layers) < 0:
        raise ValueError(f"gcn_layers must be >= 0, got {gcn_layers}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if update_order not in ("code", "paper"):
        raise ValueError(f"update_order must be 'code' or 'paper', got {update_order!r}")
    if int(envelope_exponent) < 1:
        raise ValueError(f"envelope_exponent must be >= 1, got {envelope_exponent}")
    if float(gate_eps) <= 0:
        raise ValueError(f"gate_eps must be positive, got {gate_eps}")
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    cutoff_sr = float(cutoff_sr)
    num_angle_rbf = int(num_angle_rbf)
    embedding_features = int(embedding_features)
    hidden_features = int(dim_embedding if hidden_features is None else hidden_features)
    if hidden_features < 1:
        raise ValueError(f"hidden_features must be >= 1, got {hidden_features}")
    alignn_layers = int(alignn_layers)
    gcn_layers = int(gcn_layers)
    norm = _norm_name(norm)
    residual = bool(residual)
    update_order = str(update_order)
    backtracking = bool(backtracking)
    multiply_envelope = bool(multiply_envelope)
    envelope_exponent = int(envelope_exponent)
    gate_eps = float(gate_eps)
    dim_feature_out = hidden_features
    feature_irreps = f"{hidden_features}x0e"

    atom_mlp = FeatureMLP(dim_embedding, hidden_features, norm)
    edge_mlp = embedding_mlp(
        num_rbf,
        embedding_features,
        hidden_features,
        norm,
    )
    width = hidden_features
    if alignn_layers > 0:
        centers, coeff = alignn_gaussian_terms(-1.0, 1.0, num_angle_rbf)
        angle_centers = centers
        angle_coeff = coeff
        angle_mlp = embedding_mlp(
            num_angle_rbf,
            embedding_features,
            width,
            norm,
        )
        alignn = [ALIGNNConv(width, norm, residual, gate_eps) for _ in range(alignn_layers)]
    else:
        angle_centers = angle_coeff = angle_mlp = None
        alignn = []
    gcn = [EdgeGatedGraphConv(width, norm, residual, gate_eps) for _ in range(gcn_layers)]

    embed = ALIGNNEmbed(
        dim_embedding,
        num_rbf,
        cutoff_sr,
        alignn_layers,
        update_order,
        backtracking,
        multiply_envelope,
        envelope_exponent,
        atom_mlp,
        edge_mlp,
        angle_mlp,
        angle_centers,
        angle_coeff,
    )
    blocks = [FnBlock(layer, alignn_step) for layer in alignn]
    blocks.extend(FnBlock(layer, gcn_step) for layer in gcn)
    head = FeatureHead()
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(optional_cutoff_values=False, topology=(
            TripletRequest("line_graph", exclude_backtracking=not backtracking),
        ) if alignn_layers else ()),
        builder=build_stack,
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture="alignn",
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        alignn_layers=alignn_layers,
        gcn_layers=gcn_layers,
        hidden_features=hidden_features,
        update_order=update_order,
        norm=norm,
        cutoff_sr=cutoff_sr,
        num_rbf=num_rbf,
    )


def chgnet(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    bond_cutoff: float = 3.0,
    num_angular: int = 9,
    num_blocks: int = 4,
    dim_bond: Optional[int] = None,
    dim_angle: Optional[int] = None,
    atom_hidden: HiddenSpec = None,
    bond_hidden: HiddenSpec = None,
    angle_hidden: HiddenSpec = 0,
    dropout: float = 0.0,
    envelope_exponent: int = 5,
    gmlp_norm: Optional[str] = "layer",
    readout_norm: Optional[str] = "layer",
    conv_norm: Optional[str] = None,
    update_bond: bool = True,
    update_angle: bool = True,
    learnable_frequency: bool = True,
    mlp_out_bias: bool = False,
) -> ComposableCore:
    from ..chgnet.stages import (
        CHGNetEmbed,
        ChargeFeatureHead,
        BondGraphBlock,
        _hidden_widths,
        _norm_choice,
        build_stack,
    )
    from torch.nn import Linear
    from ..chgnet.basis import FourierExpansion
    from ..chgnet.geometry import BondGraphRequest
    from ..chgnet.interaction import AtomConv, BondConv, AngleUpdate, _feature_norm

    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if int(num_angular) < 1 or int(num_angular) % 2 == 0:
        raise ValueError(f"num_angular must be a positive odd integer, got {num_angular}")
    if int(num_blocks) < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if float(bond_cutoff) <= 0:
        raise ValueError(f"bond_cutoff must be positive, got {bond_cutoff}")
    if float(bond_cutoff) > float(cutoff_sr) + 1e-6:
        raise ValueError(
            f"bond_cutoff ({bond_cutoff}) cannot exceed cutoff_sr ({cutoff_sr}): "
            "angles are pairs of edges inside the neighbor list"
        )
    if float(dropout) < 0 or float(dropout) >= 1:
        raise ValueError(f"dropout must lie in [0, 1), got {dropout}")
    if int(envelope_exponent) < 1:
        raise ValueError(f"envelope_exponent must be >= 1, got {envelope_exponent}")
    if update_angle and not update_bond:
        raise ValueError(
            "update_angle requires update_bond: the reference network "
            "skips the angle update when the bond convolution is off"
        )
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    cutoff_sr = float(cutoff_sr)
    bond_cutoff = float(bond_cutoff)
    num_angular = int(num_angular)
    num_blocks = int(num_blocks)
    dim_bond = int(dim_embedding if dim_bond is None else dim_bond)
    dim_angle = int(dim_embedding if dim_angle is None else dim_angle)
    if dim_bond < 1 or dim_angle < 1:
        raise ValueError(f"dim_bond and dim_angle must be >= 1, got {dim_bond}, {dim_angle}")
    atom_hidden = _hidden_widths(atom_hidden, dim_embedding, "atom_hidden")
    bond_hidden = _hidden_widths(bond_hidden, dim_embedding, "bond_hidden")
    angle_hidden = _hidden_widths(angle_hidden, 0, "angle_hidden")
    dropout = float(dropout)
    envelope_exponent = int(envelope_exponent)
    gmlp_norm = _norm_choice(gmlp_norm, "gmlp_norm")
    readout_norm_name = _norm_choice(readout_norm, "readout_norm")
    conv_norm = _norm_choice(conv_norm, "conv_norm")
    update_bond = bool(update_bond)
    update_angle = bool(update_angle)
    learnable_frequency = bool(learnable_frequency)
    mlp_out_bias = bool(mlp_out_bias)
    dim_feature_out = dim_embedding
    feature_irreps = f"{dim_embedding}x0e"
    order = (num_angular - 1) // 2

    bond_embed = Linear(num_rbf, dim_bond, bias=False)
    atom_weight = Linear(num_rbf, dim_embedding, bias=False)
    bond_weight = Linear(num_rbf, dim_bond, bias=False)
    angle_embed = Linear(num_angular, dim_angle, bias=False)
    fourier = FourierExpansion(order, learnable=learnable_frequency)
    atom_convs = [
        AtomConv(
            dim_embedding,
            dim_bond,
            atom_hidden,
            dropout,
            gmlp_norm,
            conv_norm,
            mlp_out_bias,
        )
        for _ in range(num_blocks)
    ]
    n_line = num_blocks - 1
    bond_convs = [
        BondConv(
            dim_embedding,
            dim_bond,
            dim_angle,
            bond_hidden,
            dropout,
            gmlp_norm,
            conv_norm,
            mlp_out_bias,
        )
        for _ in range(n_line if update_bond else 0)
    ]
    angle_updates = [
        AngleUpdate(
            dim_embedding,
            dim_bond,
            dim_angle,
            angle_hidden,
            dropout,
            gmlp_norm,
            conv_norm,
        )
        for _ in range(n_line if update_angle else 0)
    ]
    readout_norm = _feature_norm(readout_norm_name, dim_embedding)

    embed = CHGNetEmbed(
        dim_embedding,
        num_rbf,
        bond_cutoff,
        num_angular,
        bond_embed,
        atom_weight,
        bond_weight,
        angle_embed,
        fourier,
    )
    blocks = [
        BondGraphBlock(
            atom,
            bond_convs[i] if i < len(bond_convs) else None,
            angle_updates[i] if i < len(angle_updates) else None,
            i == num_blocks - 1,
        )
        for i, atom in enumerate(atom_convs)
    ]
    head = ChargeFeatureHead(readout_norm)
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(optional_cutoff_values=False, fields=("rbf_bg",), topology=(BondGraphRequest(bond_cutoff),)),
        builder=build_stack,
        positional_fields=("atom_embedding", "rbf", "rbf_bg", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture="chgnet",
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        num_blocks=num_blocks,
        bond_cutoff=bond_cutoff,
        cutoff_sr=cutoff_sr,
        num_rbf=num_rbf,
        envelope_exponent=envelope_exponent,
    )


def spinconv(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    max_Za: int,
    hidden_channels: int = 32,
    mid_hidden_channels: int = 256,
    num_interactions: int = 3,
    sphere_size_lat: int = 16,
    sphere_size_long: int = 12,
    max_num_neighbors: int = 40,
    embedding_size: Optional[int] = None,
    num_embedding_basis: int = 8,
    group_channels: int = 8,
    scale_distances: bool = True,
    distance_scale_max: float = 2.0,
    basis_width_scalar: float = 3.0,
    residual_first: bool = False,
) -> ComposableCore:
    from ..spinconv.stages import (
        SpinConvEmbed,
        SpinMessage,
        SpinHead,
        build_stack,
    )
    from torch.nn import Linear
    from ..layers.rbf import spinconv_gaussian_terms
    from ..spinconv.interaction import EmbeddingBlock, MessageBlock, PairDistanceScale, _swish

    dim_embedding = _positive("dim_embedding", dim_embedding)
    num_rbf = _positive("num_rbf", num_rbf)
    hidden_channels = _positive("hidden_channels", hidden_channels)
    mid_hidden_channels = _positive("mid_hidden_channels", mid_hidden_channels)
    num_interactions = _positive("num_interactions", num_interactions)
    sphere_size_lat = _positive("sphere_size_lat", sphere_size_lat)
    sphere_size_long = _positive("sphere_size_long", sphere_size_long)
    max_num_neighbors = _positive("max_num_neighbors", max_num_neighbors)
    num_embedding_basis = _positive("num_embedding_basis", num_embedding_basis)
    group_channels = _positive("group_channels", group_channels)
    max_Za = _positive("max_Za", max_Za)
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if sphere_size_lat < 2 or sphere_size_long < 2:
        raise ValueError(
            "sphere_size_lat and sphere_size_long must be >= 2, got "
            f"{sphere_size_lat} x {sphere_size_long}"
        )
    if hidden_channels % group_channels != 0:
        raise ValueError(
            f"hidden_channels ({hidden_channels}) must be divisible by " f"group_channels ({group_channels})"
        )
    groups = hidden_channels // group_channels
    if mid_hidden_channels % groups != 0:
        raise ValueError(
            f"mid_hidden_channels ({mid_hidden_channels}) must be divisible by "
            f"hidden_channels / group_channels ({groups})"
        )
    if float(basis_width_scalar) <= 0:
        raise ValueError(f"basis_width_scalar must be positive, got {basis_width_scalar}")
    embedding_size = dim_embedding if embedding_size is None else _positive("embedding_size", embedding_size)

    cutoff_sr = float(cutoff_sr)
    scale_distances = bool(scale_distances)
    distance_scale_max = float(distance_scale_max)
    basis_width_scalar = float(basis_width_scalar)
    residual_first = bool(residual_first)
    message_divisor = float(max_num_neighbors) / 2.0 + 1.0
    dim_feature_out = mid_hidden_channels
    feature_irreps = f"{mid_hidden_channels}x0e"
    num_output_blocks = 1

    center, coeff = spinconv_gaussian_terms(cutoff_sr, num_rbf, basis_width_scalar)
    gaussian_center = center
    gaussian_coeff = coeff

    act = _swish()
    distance_map = Linear(num_rbf, mid_hidden_channels)
    init_fc1 = Linear(mid_hidden_channels, mid_hidden_channels)
    init_fc2 = Linear(mid_hidden_channels, mid_hidden_channels)
    mix_kwargs = dict(
        dim_embedding=dim_embedding,
        embedding_size=embedding_size,
        num_basis=num_embedding_basis,
    )
    init_mix = EmbeddingBlock(
        mid_hidden_channels,
        hidden_channels,
        mid_hidden_channels,
        **mix_kwargs,
    )
    interactions = [
        MessageBlock(
            hidden_channels,
            mid_hidden_channels,
            dim_embedding,
            embedding_size,
            num_embedding_basis,
            sphere_size_lat,
            sphere_size_long,
            group_channels,
            cutoff_sr,
        )
        for _ in range(num_interactions)
    ]
    energy_mix = EmbeddingBlock(
        hidden_channels,
        mid_hidden_channels,
        mid_hidden_channels,
        project=False,
        **mix_kwargs,
    )
    pair_scale = PairDistanceScale(max_Za, distance_scale_max) if scale_distances else None

    embed = SpinConvEmbed(
        dim_embedding,
        num_rbf,
        max_num_neighbors,
        act,
        distance_map,
        init_fc1,
        init_fc2,
        init_mix,
        pair_scale,
        gaussian_center,
        gaussian_coeff,
        cutoff_sr,
        sphere_size_lat,
        sphere_size_long,
    )
    blocks = [SpinMessage(layer, residual_first or i > 0) for i, layer in enumerate(interactions)]
    head = SpinHead(energy_mix, hidden_channels, message_divisor)
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(optional_cutoff_values=False, fields=("Za",), topology=(embed.projection,)),
        builder=build_stack,
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr", "Za"),
        architecture="spinconv",
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        hidden_channels=hidden_channels,
        mid_hidden_channels=mid_hidden_channels,
        num_interactions=num_interactions,
        sphere_size_lat=sphere_size_lat,
        sphere_size_long=sphere_size_long,
        max_num_neighbors=max_num_neighbors,
        message_divisor=message_divisor,
        scale_distances=scale_distances,
        residual_first=residual_first,
        num_output_blocks=num_output_blocks,
        num_rbf=num_rbf,
        cutoff_sr=cutoff_sr,
        basis_width_scalar=basis_width_scalar,
        max_Za=max_Za,
    )


def gemnet(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    num_blocks: int = 4,
    num_spherical: int = 7,
    emb_size_edge: Optional[int] = None,
    emb_size_trip: int = 64,
    emb_size_quad: int = 32,
    emb_size_rbf: int = 16,
    emb_size_cbf: int = 16,
    emb_size_sbf: int = 32,
    emb_size_bil_trip: int = 64,
    emb_size_bil_quad: int = 32,
    num_before_skip: int = 1,
    num_after_skip: int = 1,
    num_concat: int = 1,
    num_atom: int = 2,
    triplets_only: bool = True,
    emb_cutoff: Optional[float] = None,
    int_cutoff: Optional[float] = None,
    envelope_exponent: int = 5,
    output_mode: Literal["feature", "last"] = "feature",
) -> ComposableCore:
    from ..gemnet.stages import (
        GemNetEmbed,
        build_stack,
    )
    from ..gemnet.layers import AtomUpdate, BasisDownProjection, EdgeEmbedding, GemNetDense, GemNetInteraction
    from ..gemnet.basis import SphericalBesselRadial

    dim_embedding = _positive("dim_embedding", dim_embedding)
    num_rbf = _positive("num_rbf", num_rbf)
    num_blocks = _positive("num_blocks", num_blocks)
    num_spherical = _positive("num_spherical", num_spherical)
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if output_mode not in ("feature", "last"):
        raise ValueError(f"output_mode must be 'feature' or 'last', got {output_mode!r}")
    emb_edge = dim_embedding if emb_size_edge is None else _positive("emb_size_edge", emb_size_edge)
    emb_cutoff_v = float(cutoff_sr if emb_cutoff is None else emb_cutoff)
    if triplets_only:
        int_cutoff_v = emb_cutoff_v
    else:
        int_cutoff_v = float(emb_cutoff_v if int_cutoff is None else int_cutoff)
    if emb_cutoff_v <= 0 or int_cutoff_v <= 0:
        raise ValueError("emb_cutoff and int_cutoff must be positive")
    if emb_cutoff_v > float(cutoff_sr) + 1e-6:
        raise ValueError(f"emb_cutoff ({emb_cutoff_v}) cannot exceed cutoff_sr ({cutoff_sr})")
    if int_cutoff_v > float(cutoff_sr) + 1e-6:
        raise ValueError(f"int_cutoff ({int_cutoff_v}) cannot exceed cutoff_sr ({cutoff_sr})")
    if (not triplets_only) and int_cutoff_v + 1e-6 < emb_cutoff_v:
        raise ValueError(f"int_cutoff ({int_cutoff_v}) must be >= emb_cutoff ({emb_cutoff_v})")

    triplets_only = bool(triplets_only)
    cutoff_sr = float(cutoff_sr)
    emb_cutoff = emb_cutoff_v
    int_cutoff = int_cutoff_v
    emb_size_edge = emb_edge
    num_output_blocks = num_blocks + 1
    dim_feature_out = dim_embedding
    feature_irreps = f"{dim_embedding}x0e"
    envelope_exponent = int(envelope_exponent)

    edge_emb = EdgeEmbedding(dim_embedding, num_rbf, emb_edge)
    triplet_radial = SphericalBesselRadial(
        num_spherical, num_rbf, emb_cutoff, envelope_exponent, repeat_m=False
    )
    mlp_rbf3 = GemNetDense(num_rbf, _positive("emb_size_rbf", emb_size_rbf), activation=False)
    mlp_cbf3 = BasisDownProjection(num_spherical, num_rbf, _positive("emb_size_cbf", emb_size_cbf))
    mlp_rbf_h = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    mlp_rbf_out = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    if triplets_only:
        quad_cbf_radial = None
        quad_sbf_radial = None
        mlp_rbf4 = None
        mlp_cbf4 = None
        mlp_sbf4 = None
    else:
        quad_cbf_radial = SphericalBesselRadial(
            num_spherical, num_rbf, int_cutoff, envelope_exponent, repeat_m=False
        )
        quad_sbf_radial = SphericalBesselRadial(
            num_spherical, num_rbf, emb_cutoff, envelope_exponent, repeat_m=True
        )
        mlp_rbf4 = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        mlp_cbf4 = GemNetDense(num_spherical * num_rbf, emb_size_cbf, activation=False)
        mlp_sbf4 = BasisDownProjection(
            quad_sbf_radial.angular_dim,
            num_rbf,
            _positive("emb_size_sbf", emb_size_sbf),
        )

    block_kwargs = dict(
        emb_atom=dim_embedding,
        emb_edge=emb_edge,
        emb_trip=_positive("emb_size_trip", emb_size_trip),
        emb_quad=_positive("emb_size_quad", emb_size_quad) if not triplets_only else 1,
        emb_rbf=emb_size_rbf,
        emb_cbf=emb_size_cbf,
        emb_sbf=_positive("emb_size_sbf", emb_size_sbf) if not triplets_only else 1,
        emb_bil_trip=_positive("emb_size_bil_trip", emb_size_bil_trip),
        emb_bil_quad=_positive("emb_size_bil_quad", emb_size_bil_quad) if not triplets_only else 1,
        num_before_skip=_nonnegative("num_before_skip", num_before_skip),
        num_after_skip=_nonnegative("num_after_skip", num_after_skip),
        num_concat=_nonnegative("num_concat", num_concat),
        num_atom=_nonnegative("num_atom", num_atom),
        triplets_only=triplets_only,
    )
    interactions = [GemNetInteraction(**block_kwargs) for _ in range(num_blocks)]
    readouts = [
        AtomUpdate(dim_embedding, emb_edge, emb_size_rbf, block_kwargs["num_atom"])
        for _ in range(num_output_blocks)
    ]

    embed = GemNetEmbed(
        triplets_only,
        num_rbf,
        emb_cutoff,
        int_cutoff,
        num_spherical,
        edge_emb,
        triplet_radial,
        mlp_rbf3,
        mlp_cbf3,
        mlp_rbf_h,
        mlp_rbf_out,
        quad_cbf_radial,
        quad_sbf_radial,
        mlp_rbf4,
        mlp_cbf4,
        mlp_sbf4,
    )
    blocks = [GemNetBlock(interaction) for interaction in interactions]
    head = GemNetHead(readouts, output_mode, include_nodes=False)
    topology = (ReverseEdgeRequest(emb_cutoff), TripletRequest("receiver", edge_cutoff=emb_cutoff,
                                                             exclude_same_sender=True))
    if not triplets_only:
        topology += (QuadrupletRequest(emb_cutoff, int_cutoff),)
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(optional_cutoff_values=False, topology=topology),
        builder=build_stack,
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture="gemnet" if triplets_only else "gemnet-q",
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        num_blocks=num_blocks,
        num_output_blocks=num_output_blocks,
        output_mode=output_mode,
        triplets_only=triplets_only,
        emb_cutoff=emb_cutoff,
    )


def gemnet_oc(
    dim_embedding: int,
    num_rbf: int,
    cutoff_sr: float,
    num_blocks: int = 4,
    num_spherical: int = 7,
    emb_size_edge: int = 512,
    emb_size_trip_in: int = 64,
    emb_size_trip_out: int = 64,
    emb_size_quad_in: int = 32,
    emb_size_quad_out: int = 32,
    emb_size_aint_in: int = 64,
    emb_size_aint_out: int = 64,
    emb_size_rbf: int = 16,
    emb_size_cbf: int = 16,
    emb_size_sbf: int = 32,
    num_before_skip: int = 2,
    num_after_skip: int = 2,
    num_concat: int = 1,
    num_atom: int = 3,
    num_atom_emb_layers: int = 2,
    num_output_afteratom: int = 3,
    cutoff: Optional[float] = None,
    cutoff_qint: Optional[float] = None,
    cutoff_aeaint: Optional[float] = None,
    cutoff_aint: Optional[float] = None,
    max_neighbors: int = 30,
    max_neighbors_qint: int = 8,
    max_neighbors_aeaint: int = 20,
    max_neighbors_aint: int = 1000,
    quad_interaction: bool = True,
    atom_edge_interaction: bool = True,
    edge_atom_interaction: bool = True,
    atom_interaction: bool = True,
    envelope_exponent: int = 5,
    output_mode: Literal["feature", "last"] = "feature",
) -> ComposableCore:
    from ..gemnet.stages_oc import (
        GemNetOCEmbed,
        _cutoff,
        build_stack,
    )
    from ..gemnet.layers import BasisDownProjection, EdgeEmbedding, GemNetDense
    from ..gemnet.oc import GemNetOCBlock, GemNetOCOutput

    dim_embedding = _positive("dim_embedding", dim_embedding)
    num_rbf = _positive("num_rbf", num_rbf)
    num_blocks = _positive("num_blocks", num_blocks)
    num_spherical = _positive("num_spherical", num_spherical)
    cutoff_sr_v = _cutoff("cutoff_sr", cutoff_sr)
    cutoff_aint_v = cutoff_sr_v if cutoff_aint is None else _cutoff("cutoff_aint", cutoff_aint)
    if cutoff_aint_v > cutoff_sr_v + 1e-6:
        raise ValueError(f"cutoff_aint ({cutoff_aint_v}) cannot exceed cutoff_sr ({cutoff_sr_v})")
    cutoff_v = cutoff_aint_v if cutoff is None else _cutoff("cutoff", cutoff)
    cutoff_q_v = cutoff_v if cutoff_qint is None else _cutoff("cutoff_qint", cutoff_qint)
    cutoff_ae_v = cutoff_v if cutoff_aeaint is None else _cutoff("cutoff_aeaint", cutoff_aeaint)
    if cutoff_v > cutoff_aint_v + 1e-6:
        raise ValueError(f"cutoff ({cutoff_v}) cannot exceed cutoff_aint ({cutoff_aint_v})")
    if cutoff_q_v > cutoff_aint_v + 1e-6:
        raise ValueError(f"cutoff_qint ({cutoff_q_v}) cannot exceed cutoff_aint ({cutoff_aint_v})")
    if cutoff_ae_v > cutoff_aint_v + 1e-6:
        raise ValueError(f"cutoff_aeaint ({cutoff_ae_v}) cannot exceed cutoff_aint ({cutoff_aint_v})")
    if output_mode not in ("feature", "last"):
        raise ValueError(f"output_mode must be 'feature' or 'last', got {output_mode!r}")
    if int(envelope_exponent) < 1:
        raise ValueError(f"envelope_exponent must be >= 1, got {envelope_exponent}")

    quad_interaction = bool(quad_interaction)
    atom_edge_interaction = bool(atom_edge_interaction)
    edge_atom_interaction = bool(edge_atom_interaction)
    atom_interaction = bool(atom_interaction)
    cutoff_sr = cutoff_sr_v
    cutoff = cutoff_v
    cutoff_qint = cutoff_q_v
    cutoff_aeaint = cutoff_ae_v
    cutoff_aint = cutoff_aint_v
    max_neighbors = _positive("max_neighbors", max_neighbors)
    max_neighbors_qint = _positive("max_neighbors_qint", max_neighbors_qint)
    max_neighbors_aeaint = _positive("max_neighbors_aeaint", max_neighbors_aeaint)
    max_neighbors_aint = _positive("max_neighbors_aint", max_neighbors_aint)
    emb_size_edge = _positive("emb_size_edge", emb_size_edge)
    emb_size_rbf = _positive("emb_size_rbf", emb_size_rbf)
    emb_size_cbf = _positive("emb_size_cbf", emb_size_cbf)
    emb_size_sbf = _positive("emb_size_sbf", emb_size_sbf)
    envelope_exponent = int(envelope_exponent)
    num_output_blocks = num_blocks + 1
    dim_feature_out = dim_embedding
    feature_irreps = f"{dim_embedding}x0e"

    edge_emb = EdgeEmbedding(dim_embedding, num_rbf, emb_size_edge)
    mlp_rbf_tint = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    mlp_rbf_h = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    mlp_rbf_out = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    mlp_cbf_tint = BasisDownProjection(num_spherical, num_rbf, emb_size_cbf)
    if quad_interaction:
        mlp_rbf_qint = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        mlp_cbf_qint = BasisDownProjection(num_spherical, num_rbf, emb_size_cbf)
        mlp_sbf_qint = BasisDownProjection(num_spherical * num_spherical, num_rbf, emb_size_sbf)
    else:
        mlp_rbf_qint = None
        mlp_cbf_qint = None
        mlp_sbf_qint = None
    if atom_edge_interaction:
        mlp_rbf_aeint = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        mlp_cbf_aeint = BasisDownProjection(num_spherical, num_rbf, emb_size_cbf)
    else:
        mlp_rbf_aeint = None
        mlp_cbf_aeint = None
    if edge_atom_interaction:
        mlp_rbf_eaint = GemNetDense(num_rbf, emb_size_rbf, activation=False)
        mlp_cbf_eaint = BasisDownProjection(num_spherical, num_rbf, emb_size_cbf)
    else:
        mlp_rbf_eaint = None
        mlp_cbf_eaint = None
    if atom_interaction:
        mlp_rbf_aint = GemNetDense(num_rbf, emb_size_rbf, activation=False)
    else:
        mlp_rbf_aint = None

    block_kwargs = dict(
        emb_atom=dim_embedding,
        emb_edge=emb_size_edge,
        emb_trip_in=_positive("emb_size_trip_in", emb_size_trip_in),
        emb_trip_out=_positive("emb_size_trip_out", emb_size_trip_out),
        emb_quad_in=_positive("emb_size_quad_in", emb_size_quad_in),
        emb_quad_out=_positive("emb_size_quad_out", emb_size_quad_out),
        emb_aint_in=_positive("emb_size_aint_in", emb_size_aint_in),
        emb_aint_out=_positive("emb_size_aint_out", emb_size_aint_out),
        emb_rbf=emb_size_rbf,
        emb_cbf=emb_size_cbf,
        emb_sbf=emb_size_sbf,
        num_before_skip=_nonnegative("num_before_skip", num_before_skip),
        num_after_skip=_nonnegative("num_after_skip", num_after_skip),
        num_concat=_nonnegative("num_concat", num_concat),
        num_atom=_nonnegative("num_atom", num_atom),
        num_atom_emb_layers=_nonnegative("num_atom_emb_layers", num_atom_emb_layers),
        quad_interaction=quad_interaction,
        atom_edge_interaction=atom_edge_interaction,
        edge_atom_interaction=edge_atom_interaction,
        atom_interaction=atom_interaction,
    )
    interactions = [GemNetOCBlock(**block_kwargs) for _ in range(num_blocks)]
    n_after = _nonnegative("num_output_afteratom", num_output_afteratom)
    readouts = [
        GemNetOCOutput(
            dim_embedding,
            emb_size_edge,
            emb_size_rbf,
            block_kwargs["num_atom"],
            n_after,
        )
        for _ in range(num_output_blocks)
    ]

    embed = GemNetOCEmbed(
        quad_interaction,
        atom_edge_interaction,
        edge_atom_interaction,
        atom_interaction,
        dim_embedding,
        num_rbf,
        cutoff,
        cutoff_qint,
        cutoff_aeaint,
        cutoff_aint,
        max_neighbors,
        max_neighbors_qint,
        max_neighbors_aeaint,
        max_neighbors_aint,
        num_spherical,
        emb_size_cbf,
        edge_emb,
        mlp_rbf_tint,
        mlp_rbf_h,
        mlp_rbf_out,
        mlp_cbf_tint,
        mlp_rbf_qint,
        mlp_cbf_qint,
        mlp_sbf_qint,
        mlp_rbf_aeint,
        mlp_cbf_aeint,
        mlp_rbf_eaint,
        mlp_cbf_eaint,
        mlp_rbf_aint,
    )
    blocks = [GemNetBlock(interaction) for interaction in interactions]
    head = GemNetHead(readouts, output_mode, include_nodes=True)
    return ComposableCore(
        embed,
        blocks,
        head,
        GraphRequires(optional_cutoff_values=False, topology=(embed.graph_request,)),
        builder=build_stack,
        positional_fields=("atom_embedding", "rbf", "Dij_sr", "vij_sr", "idx_i_sr", "idx_j_sr"),
        architecture="gemnet-oc",
        dim_embedding=dim_embedding,
        dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
        num_blocks=num_blocks,
        num_output_blocks=num_output_blocks,
        output_mode=output_mode,
        atom_edge_interaction=atom_edge_interaction,
        edge_atom_interaction=edge_atom_interaction,
        atom_interaction=atom_interaction,
        quad_interaction=quad_interaction,
        cutoff_sr=cutoff_sr,
        num_rbf=num_rbf,
        envelope_exponent=envelope_exponent,
    )
