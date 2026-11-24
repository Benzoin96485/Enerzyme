"""e3nn irreps message-passing presets.

``mace`` loads ``main``-branch checkpoints through
:mod:`enerzyme.models.compat.main_branch`; its parameter names and numerics
follow ``main``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

from ..mp.core import ComposableCore, FeatureHead, FieldEmbed, ReadoutHead
from ..mp.graph import GraphRequires

from ..equiformer.defaults import _AVG_DEGREE, _AVG_NUM_NODES
from ..eqnorm.defaults import PUBLISHED_MPTRJ_CORE_PARAMS
from ..nequix.defaults import PUBLISHED_MP_CORE_PARAMS
from ..mp.block import FnBlock
from ..mp.message.irreps import (
    ChannelwiseBlock, IrrepsEmbed, dual_branch_update, equflash_update,
    nequip_update, separable_update,
)

_IRREPS_INPUTS = ("atom_embedding", "rbf", "idx_i_sr", "idx_j_sr", "vij_sr")


MACE_OUTPUT_MODES = ("direct", "feature", "layers")


def _mace_banner(output_mode: str, feature_irreps: str) -> str:
    return f"""
########################################################
# MACE (NeurIPS 2022, arXiv:2206.07697)                #
# output_mode={output_mode} features={feature_irreps}
########################################################
"""


def mace(
    max_Za: int, max_ell: int, dim_embedding: int, num_rbf: int,
    additional_hidden_irreps: str,
    interaction_cls_first: str = "RealAgnosticResidualInteractionBlock",
    interaction_cls: str = "RealAgnosticResidualInteractionBlock",
    correlation: Union[int, List[int]] = 3,
    num_interactions: int = 2,
    avg_num_neighbors: float = 1.0,
    MLP_irreps: str = "16x0e",
    radial_MLP: Optional[List[int]] = None,
    gate: str = "silu",
    shallow_ensemble_size: int = 1,
    output_mode: Literal["direct", "feature", "layers"] = "direct",
    keep_last_layer_irreps: bool = False,
) -> ComposableCore:
    """MACE: uvu convolution + symmetric contraction per layer, per-layer readouts summed.

    Messages flow ``idx_i → idx_j`` (``edge_convention="swap"``) as on ``main``.
    The last layer keeps only its scalars unless ``keep_last_layer_irreps``
    (forced by ``output_mode="layers"``).
    """
    from e3nn.o3 import Irreps

    from ..mace.defaults import GATE_FUNCTIONS
    from ..mace.interaction import (
        INTERACTION_CLASSES, EquivariantProductBasisBlock, LinearReadoutBlock, MACEBlock, NonLinearReadoutBlock,
    )
    from ..mp.equivariant.e3nn import scalar_0e_dim

    if output_mode not in MACE_OUTPUT_MODES:
        raise ValueError(f"output_mode must be 'direct', 'feature', or 'layers', got {output_mode!r}")
    if output_mode == "layers":
        keep_last_layer_irreps = True
    num_interactions = int(num_interactions)
    if num_interactions < 1:
        raise ValueError(f"num_interactions must be >= 1, got {num_interactions}")
    if isinstance(correlation, int):
        correlation = [correlation] * num_interactions
    if len(correlation) != num_interactions:
        raise ValueError(f"correlation has length {len(correlation)}, expected {num_interactions}")
    if radial_MLP is None:
        radial_MLP = [64, 64, 64]
    for name in (interaction_cls_first, interaction_cls):
        if name not in INTERACTION_CLASSES:
            raise KeyError(f"Unknown interaction class {name!r}. Choices: {sorted(INTERACTION_CLASSES)}")

    num_elements = max_Za + 1
    sh_irreps = Irreps.spherical_harmonics(max_ell)
    node_attrs_irreps = Irreps([(num_elements, (0, 1))])
    edge_feats_irreps = Irreps([(num_rbf, (0, 1))])
    hidden_body = f"{dim_embedding}x0e"
    if additional_hidden_irreps:
        hidden_body = hidden_body + "+" + additional_hidden_irreps
    hidden_irreps = Irreps(hidden_body).simplify()
    interaction_irreps = (sh_irreps * dim_embedding).sort()[0].simplify()

    blocks = []
    readouts = []
    layer_out = []
    node_irreps = Irreps([(dim_embedding, (0, 1))])
    for i in range(num_interactions):
        first = i == 0
        last = i == num_interactions - 1
        out_irreps = hidden_irreps
        if last and not first and not keep_last_layer_irreps:
            out_irreps = Irreps(str(hidden_irreps[0]))
        cls_name = interaction_cls_first if first else interaction_cls
        interaction = INTERACTION_CLASSES[cls_name](
            node_attrs_irreps=node_attrs_irreps, node_feats_irreps=node_irreps, edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps, target_irreps=interaction_irreps, hidden_irreps=out_irreps,
            avg_num_neighbors=avg_num_neighbors, radial_MLP=radial_MLP,
        )
        product = EquivariantProductBasisBlock(
            node_feats_irreps=interaction.target_irreps if first else interaction_irreps,
            target_irreps=out_irreps, correlation=correlation[i], num_elements=num_elements,
            use_sc=("Residual" in cls_name) if first else True,
        )
        blocks.append(MACEBlock(interaction, product, num_elements, max_ell))
        layer_out.append(Irreps(out_irreps).simplify())
        if output_mode == "direct":
            if last and not first:
                readouts.append(NonLinearReadoutBlock(out_irreps, MLP_irreps, GATE_FUNCTIONS[gate], shallow_ensemble_size))
            else:
                readouts.append(LinearReadoutBlock(hidden_irreps, shallow_ensemble_size))
        node_irreps = hidden_irreps

    if output_mode == "layers":
        emitted, num_output_blocks = hidden_irreps, num_interactions
        head = FeatureHead("layers", source="equi")
    elif output_mode == "feature":
        emitted, num_output_blocks = layer_out[-1], 1
        head = FeatureHead("feature", source="equi")
    else:
        emitted, num_output_blocks = layer_out[-1], num_interactions
        head = ReadoutHead(readouts=readouts, mode="per_block", source="equi", ensemble_size=shallow_ensemble_size)

    requires = GraphRequires(distances=False, edge_convention="swap", fields=("Za",), optional_cutoff_values=False)
    return ComposableCore(
        FieldEmbed("atom_embedding", add_fields=("charge_embedding", "spin_embedding"), slot="equi"),
        blocks, head, requires,
        banner=_mace_banner(output_mode, str(emitted)), architecture="mace",
        output_mode=output_mode, keep_last_layer_irreps=bool(keep_last_layer_irreps),
        max_Za=max_Za, max_ell=int(max_ell), dim_embedding=int(dim_embedding), num_rbf=int(num_rbf),
        num_interactions=num_interactions, hidden_irreps=hidden_irreps,
        shallow_ensemble_size=shallow_ensemble_size, num_output_blocks=num_output_blocks,
        feature_irreps=str(emitted), dim_feature_out=scalar_0e_dim(emitted),
    )

def nequip(
    dim_embedding: int,
    num_rbf: int,
    preset: Optional[str] = None,
    num_layers: Optional[int] = None,
    l_max: Optional[int] = None,
    parity: Optional[bool] = None,
    num_features: Optional[Union[int, Sequence[int]]] = None,
    radial_mlp_depth: Optional[int] = None,
    radial_mlp_width: Optional[int] = None,
    use_sc: Optional[bool] = None,
    sc_first: Optional[bool] = None,
    resnet: Optional[bool] = None,
    resnet_first: Optional[bool] = None,
    nonlinearity_type: Optional[str] = None,
    nonlinearity_scalars: Optional[Dict[str, str]] = None,
    nonlinearity_gates: Optional[Dict[str, str]] = None,
    avg_num_neighbors: Optional[float] = None,
) -> ComposableCore:
    """Condition equivariant convolutions and self-connections on species embeddings."""
    from e3nn import o3
    from ..mp.equivariant.e3nn import scalar_0e_dim
    from ..nequip.interaction import NequIPConv, hidden_irreps
    from ..nequip.defaults import _resolve

    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    cfg = _resolve(
        preset,
        {
            "num_layers": num_layers,
            "l_max": l_max,
            "parity": parity,
            "num_features": num_features,
            "radial_mlp_depth": radial_mlp_depth,
            "radial_mlp_width": radial_mlp_width,
            "use_sc": use_sc,
            "sc_first": sc_first,
            "resnet": resnet,
            "resnet_first": resnet_first,
            "nonlinearity_type": nonlinearity_type,
            "avg_num_neighbors": avg_num_neighbors,
        },
    )
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    num_layers = int(cfg["num_layers"])
    l_max = int(cfg["l_max"])
    parity = bool(cfg["parity"])
    num_features = cfg["num_features"]
    use_sc = bool(cfg["use_sc"])
    sc_first = bool(cfg["sc_first"])
    resnet = bool(cfg["resnet"])
    resnet_first = bool(cfg["resnet_first"])
    nonlinearity_type = str(cfg["nonlinearity_type"])
    avg_num_neighbors = float(cfg["avg_num_neighbors"])
    radial_mlp_depth = int(cfg["radial_mlp_depth"])
    radial_mlp_width = int(cfg["radial_mlp_width"])
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    if avg_num_neighbors <= 0:
        raise ValueError(
            f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
        )

    irreps_edge_sh = o3.Irreps.spherical_harmonics(l_max)
    sph = o3.SphericalHarmonics(
        irreps_edge_sh, normalize=True, normalization="component"
    )
    irreps_node_attr = o3.Irreps(f"{dim_embedding}x0e")
    hidden = hidden_irreps(num_features, l_max, parity)
    if isinstance(num_features, int):
        scalar_mul = int(num_features)
    else:
        scalar_mul = int(num_features[0])
    if scalar_mul < 1:
        raise ValueError(
            "the l=0 feature count must be >= 1 so the latent has scalar channels"
        )
    scalar_out = o3.Irreps([(scalar_mul, (0, 1))])
    targets = [hidden] * (num_layers - 1) + [scalar_out]

    irreps = irreps_node_attr
    blocks = []
    for layer_i, target in enumerate(targets):
        conv = NequIPConv(
            irreps_in=irreps,
            feature_irreps_hidden=target,
            irreps_edge_attr=irreps_edge_sh,
            irreps_node_attr=irreps_node_attr,
            num_rbf=num_rbf,
            radial_mlp_depth=radial_mlp_depth,
            radial_mlp_width=radial_mlp_width,
            avg_num_neighbors=avg_num_neighbors,
            use_sc=use_sc and (layer_i > 0 or sc_first),
            resnet=resnet and (layer_i > 0 or resnet_first),
            nonlinearity_type=nonlinearity_type,
            nonlinearity_scalars=nonlinearity_scalars,
            nonlinearity_gates=nonlinearity_gates,
        )
        blocks.append(FnBlock(conv, nequip_update))
        irreps = conv.irreps_out
    feature_irreps = str(irreps)
    dim_feature_out = scalar_0e_dim(irreps)
    return ComposableCore(
        IrrepsEmbed(dim_embedding, num_rbf, sph),
        blocks, FeatureHead(source="equi"), GraphRequires(distances=False, optional_cutoff_values=False),
        positional_fields=_IRREPS_INPUTS, architecture="nequip",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        l_max=l_max, parity=parity, num_features=num_features,
        irreps_node_attr=irreps_node_attr, feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        banner=f"""
################################################################################
# NequIP (Batzner et al., Nat. Commun. 2022; mir-group/nequip convolutions)    #
# {num_layers} layers, l_max={l_max}, parity={parity}, latent {feature_irreps:<24} #
################################################################################
""",
    )


def nequix(
    dim_embedding: int = 89,
    num_rbf: int = 8,
    num_layers: int = 2,
    lmax: int = 2,
    hidden_irreps: str = "8x0e + 4x1o + 4x2e",
    radial_mlp_size: int = 32,
    radial_mlp_layers: int = 2,
    mlp_init_scale: float = 4.0,
    avg_num_neighbors: float = PUBLISHED_MP_CORE_PARAMS["avg_num_neighbors"],
    layer_norm: bool = True,
) -> ComposableCore:
    """Apply separable convolutions with a species-independent residual."""
    from e3nn import o3
    from ..mp.equivariant.e3nn import scalar_0e_dim
    from ..nequix.interaction import NequixConvolution
    from ..nequix.interaction import _keep_even_scalars

    if dim_embedding < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if num_rbf < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    if radial_mlp_size < 1:
        raise ValueError(f"radial_mlp_size must be >= 1, got {radial_mlp_size}")
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    num_layers = int(num_layers)
    lmax = int(lmax)
    hidden_irreps = o3.Irreps(hidden_irreps).regroup()
    if len(hidden_irreps) == 0:
        raise ValueError("hidden_irreps is empty")
    if hidden_irreps.lmax > lmax:
        raise ValueError(
            f"hidden irreps reach l={hidden_irreps.lmax}, "
            f"but the edge spherical harmonics stop at lmax={lmax}"
        )
    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    sph = o3.SphericalHarmonics(
        irreps_sh,
        normalize=True,
        normalization="component",
    )
    scalar = o3.Irreps(f"{dim_embedding}x0e")
    scalar_out = _keep_even_scalars(hidden_irreps)
    blocks = []
    for layer in range(num_layers):
        irreps_in = scalar if layer == 0 else hidden_irreps
        irreps_out = (
            hidden_irreps if layer < num_layers - 1 else scalar_out
        )
        blocks.append(
            NequixConvolution(
                irreps_in,
                irreps_out,
                irreps_sh,
                num_rbf=num_rbf,
                radial_mlp_size=int(radial_mlp_size),
                radial_mlp_layers=int(radial_mlp_layers),
                mlp_init_scale=float(mlp_init_scale),
                avg_num_neighbors=float(avg_num_neighbors),
                layer_norm=bool(layer_norm),
            )
        )
    feature_irreps = str(blocks[-1].irreps_out)
    dim_feature_out = scalar_0e_dim(feature_irreps)
    blocks = [FnBlock(block, separable_update) for block in blocks]

    return ComposableCore(
        IrrepsEmbed(dim_embedding, num_rbf, sph),
        blocks, FeatureHead(source="equi"), GraphRequires(distances=False, optional_cutoff_values=False),
        positional_fields=_IRREPS_INPUTS, architecture="nequix",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        lmax=lmax, hidden_irreps=hidden_irreps, irreps_sh=irreps_sh,
        feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        banner=f"""
################################################################################
# Nequix — separable NequIP conv, species-free residual, RMSNorm, scalar out  #
# lmax={lmax} layers={num_layers} hidden={hidden_irreps} -> {feature_irreps}
################################################################################
""",
    )


def equflash(
    dim_embedding: int = 256,
    num_rbf: int = 8,
    lmax: int = 3,
    num_layers: int = 10,
    parity: bool = True,
    radial_hidden: Optional[Sequence[int]] = None,
    conv_denominator: float = 1.0,
    train_denominator: bool = False,
    use_bias: bool = False,
    normalize_sph: bool = True,
    feed_forward: bool = False,
    act_radial: str = "silu",
    act_scalar_even: str = "silu",
    act_scalar_odd: str = "tanh",
    act_gate_even: str = "silu",
    act_gate_odd: str = "tanh",
    irreps_manual: Optional[Sequence[str]] = None,
    irreps_schedule: Literal["natural", "closure"] = "natural",
) -> ComposableCore:
    """Convolve with radial-weighted tensor products and equivariant gates."""
    from e3nn import o3
    from ..mp.equivariant.e3nn import scalar_0e_dim
    from ..equflash.interaction import EquFlashInteraction
    from ..equflash.irreps import equflash_irreps, spherical_harmonics_irreps

    if radial_hidden is None:
        radial_hidden = [64, 64]
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    lmax = int(lmax)
    parity = bool(parity)
    normalize_sph = bool(normalize_sph)

    schedule = equflash_irreps(
        dim_embedding,
        lmax,
        int(num_layers) if irreps_manual is None else max(len(irreps_manual) - 1, 1),
        parity,
        irreps_manual,
        schedule=irreps_schedule,
    )
    if irreps_manual is not None:
        num_layers = len(schedule) - 1
    num_layers = int(num_layers)
    scalar_in = o3.Irreps(f"{dim_embedding}x0e")
    if schedule[0] != scalar_in:
        raise ValueError(
            "EquFlash starts from the scalar atom embedding "
            f"({scalar_in}), got input irreps {schedule[0]}"
        )

    irreps_sh = spherical_harmonics_irreps(lmax, parity)
    sh_in = "1o" if parity else "1e"
    sph = o3.SphericalHarmonics(
        irreps_sh,
        normalize=normalize_sph,
        normalization="component",
        irreps_in=sh_in,
    )
    blocks = []
    irreps_in = schedule[0]
    recorded = [str(irreps_in)]
    for layer in range(num_layers):
        block = EquFlashInteraction(
            irreps_in=irreps_in,
            irreps_sh=irreps_sh,
            irreps_out=schedule[layer + 1],
            radial_hidden=radial_hidden,
            num_rbf=num_rbf,
            act_radial=act_radial,
            scalar_even=act_scalar_even,
            scalar_odd=act_scalar_odd,
            gate_even=act_gate_even,
            gate_odd=act_gate_odd,
            denominator=conv_denominator,
            train_denominator=train_denominator,
            bias=use_bias,
            feed_forward=feed_forward,
        )
        blocks.append(FnBlock(block, equflash_update))
        irreps_in = block.irreps_out
        recorded.append(str(irreps_in))
    # Gate output can reorder relative to the requested schedule.
    feature_irreps = recorded[-1]
    dim_feature_out = scalar_0e_dim(feature_irreps)
    irreps_schedule = recorded
    return ComposableCore(
        IrrepsEmbed(dim_embedding, num_rbf, sph),
        blocks, FeatureHead(source="equi"), GraphRequires(distances=False, optional_cutoff_values=False),
        positional_fields=_IRREPS_INPUTS, architecture="equflash",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        lmax=lmax, parity=parity, irreps_sh=irreps_sh,
        irreps_schedule=irreps_schedule, feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        banner="""
################################################################################
# EquFlash (SamsungDS/GGNN)                                                    #
# uvu Clebsch–Gordan convolution + equivariant gate; atom_feature irreps      #
################################################################################
""",
    )


def equflashv2(
    dim_embedding: int = 32,
    num_rbf: int = 8,
    num_layers: int = 2,
    lmax: int = 2,
    is_parity: bool = True,
    irreps_manual: Optional[Sequence[str]] = None,
    interaction_type: str = "equflash",
    conv_type: str = "full",
    sc_type: str = "identity",
    nonlinear_type: str = "bilinear_gate",
    nonlinear_scale: float = 1.0,
    nonlinear_kwargs: Optional[dict] = None,
    after_nonlinear: str = "linear",
    feedforward_type: str = "feedforward",
    ff_nonlinear_type: str = "bilinear_gate",
    ff_nonlinear_scale: float = 1.0,
    ff_nonlinear_kwargs: Optional[dict] = None,
    ff_sc_type: str = "identity",
    radial_hidden: Optional[Sequence[int]] = None,
    denominator: float = 1.0,
    train_denominator: bool = False,
    normalize_sph: bool = True,
) -> ComposableCore:
    """Alternate channelwise convolutions with node feed-forward blocks."""
    from e3nn import o3
    from ..mp.equivariant.e3nn import scalar_0e_dim
    from torch.nn import Identity
    from ..equflash.interaction_v2 import INTERACTION_TYPES, FeedForwardBlock
    from ..equflash.irreps import equflash_schedule, parse_schedule, spherical_harmonics_irreps
    import warnings

    if interaction_type not in INTERACTION_TYPES:
        raise ValueError(
            f"Unknown interaction_type {interaction_type!r}; "
            f"expected {sorted(INTERACTION_TYPES)}"
        )
    if conv_type not in ("full", "efficient"):
        raise ValueError(
            f"Unknown conv_type {conv_type!r}; expected 'full' or 'efficient'"
        )
    if conv_type == "efficient":
        warnings.warn(
            "EquFlashV2 conv_type='efficient' runs the same e3nn "
            "channelwise convolution as 'full'. This port has no separate "
            "fused kernel.",
            UserWarning,
            stacklevel=2,
        )
    if feedforward_type not in ("feedforward", "identity"):
        raise ValueError(
            f"Unknown feedforward_type {feedforward_type!r}; "
            "expected 'feedforward' or 'identity'"
        )
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    lmax = int(lmax)
    is_parity = bool(is_parity)
    normalize_sph = bool(normalize_sph)
    if irreps_manual:
        schedule = parse_schedule(irreps_manual)
    else:
        schedule = equflash_schedule(
            dim_embedding, lmax, int(num_layers), is_parity
        )
    scalar = o3.Irreps(f"{dim_embedding}x0e")
    if schedule[0] != scalar:
        raise ValueError(
            f"EquFlashV2 starts from the scalar embedding {scalar}, "
            f"got {schedule[0]}. Change dim_embedding or irreps_manual[0]."
        )
    irreps_schedule = schedule
    num_layers = len(schedule) - 1
    edge_l = max([lmax] + [block.lmax for block in schedule])
    irreps_filter = spherical_harmonics_irreps(edge_l, is_parity)
    sph = o3.SphericalHarmonics(
        irreps_filter,
        normalize=normalize_sph,
        normalization="component",
    )
    hidden = [32, 32] if radial_hidden is None else list(radial_hidden)
    interaction_cls = INTERACTION_TYPES[interaction_type]
    interactions = []
    feedforwards = []
    for layer in range(num_layers):
        irreps_in = schedule[layer]
        irreps_out = schedule[layer + 1]
        interactions.append(
            interaction_cls(
                irreps_in,
                irreps_filter,
                irreps_out,
                num_rbf=num_rbf,
                radial_hidden=hidden,
                denominator=denominator,
                train_denominator=train_denominator,
                sc_type=sc_type,
                nonlinear_type=nonlinear_type,
                nonlinear_scale=nonlinear_scale,
                nonlinear_kwargs=nonlinear_kwargs,
                after_nonlinear=after_nonlinear,
            )
        )
        if feedforward_type == "identity":
            feedforwards.append(Identity())
        else:
            feedforwards.append(
                FeedForwardBlock(
                    irreps_out,
                    irreps_out,
                    sc_type=ff_sc_type,
                    nonlinear_type=ff_nonlinear_type,
                    nonlinear_scale=ff_nonlinear_scale,
                    nonlinear_kwargs=ff_nonlinear_kwargs,
                )
            )
    blocks = [ChannelwiseBlock(interaction, ff) for interaction, ff in zip(interactions, feedforwards)]
    feature_irreps = str(schedule[-1])
    dim_feature_out = scalar_0e_dim(feature_irreps)
    return ComposableCore(
        IrrepsEmbed(dim_embedding, num_rbf, sph),
        blocks, FeatureHead(source="equi"), GraphRequires(distances=False, optional_cutoff_values=False),
        positional_fields=_IRREPS_INPUTS, architecture="equflashv2",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        lmax=lmax, is_parity=is_parity, irreps_filter=irreps_filter,
        irreps_schedule=irreps_schedule, feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        banner=f"""
################################################################################
# EquFlashV2 — channelwise CG convolution, RMSNorm, bilinear gate             #
# lmax={lmax} layers={num_layers} parity={is_parity} -> {feature_irreps}
################################################################################
""",
    )


def eqnorm(
    dim_embedding: int = 4,
    num_rbf: int = 8,
    cutoff_sr: float = 6.0,
    num_layers: int = 2,
    lmax: int = 1,
    hidden_irreps: str = "8x0e+4x1o",
    num_features: int = 8,
    invariant_layers: int = 2,
    invariant_neurons: int = 16,
    avg_num_neighbors: Optional[float] = PUBLISHED_MPTRJ_CORE_PARAMS["avg_num_neighbors"],
    normalization: str = "constant",
    message_envelope: bool = False,
    polynomial_p: int = 6,
) -> ComposableCore:
    """Accumulate invariant features alongside the equivariant tensor state."""
    from e3nn import o3
    from ..nequip.interaction import tp_path_exists
    from ..eqnorm.interaction import EqnormInteraction
    from ..eqnorm.layernorm import one_irrep_per_degree

    if int(dim_embedding) < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if int(num_rbf) < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if int(num_layers) < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if int(lmax) < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    if int(num_features) < 1:
        raise ValueError(f"num_features must be >= 1, got {num_features}")
    if float(cutoff_sr) <= 0:
        raise ValueError(f"cutoff_sr must be positive, got {cutoff_sr}")
    if int(polynomial_p) < 2:
        raise ValueError(f"polynomial_p must be >= 2, got {polynomial_p}")
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    cutoff_sr = float(cutoff_sr)
    num_layers = int(num_layers)
    lmax = int(lmax)
    num_features = int(num_features)
    normalization = str(normalization)
    message_envelope = bool(message_envelope)
    polynomial_p = int(polynomial_p)
    avg_num_neighbors = (
        None if avg_num_neighbors is None else float(avg_num_neighbors)
    )
    if normalization == "constant" and (
        avg_num_neighbors is None or avg_num_neighbors <= 0
    ):
        raise ValueError(
            "constant neighbor normalization needs a positive "
            f"avg_num_neighbors, got {avg_num_neighbors}"
        )
    if normalization not in ("constant", "degree"):
        raise ValueError(
            "normalization must be 'constant' or 'degree', "
            f"got {normalization!r}"
        )
    hidden_irreps = one_irrep_per_degree(hidden_irreps)
    if hidden_irreps.lmax > lmax:
        raise ValueError(
            f"hidden irreps reach l={hidden_irreps.lmax}, "
            f"but spherical harmonics stop at lmax={lmax}"
        )
    irreps_sh = o3.Irreps.spherical_harmonics(lmax)
    scalar_in = o3.Irreps(f"{num_features}x0e")
    missing = [
        f"{mul}x{ir}"
        for mul, ir in hidden_irreps
        if not tp_path_exists(scalar_in, irreps_sh, ir)
    ]
    if missing:
        raise ValueError(
            "hidden irreps "
            f"{missing} are not reachable from {scalar_in} and "
            f"spherical harmonics up to lmax={lmax}"
        )
    sph = o3.SphericalHarmonics(
        irreps_sh,
        normalize=True,
        normalization="component",
    )
    species_linear = o3.Linear(
        o3.Irreps(f"{dim_embedding}x0e"),
        scalar_in,
        biases=False,
    )
    blocks = []
    irreps_in = scalar_in
    for _ in range(num_layers):
        block = EqnormInteraction(
            irreps_in,
            hidden_irreps,
            irreps_sh,
            scalar_in,
            num_rbf=num_rbf,
            invariant_layers=int(invariant_layers),
            invariant_neurons=int(invariant_neurons),
            normalization=normalization,
            avg_num_neighbors=avg_num_neighbors,
        )
        blocks.append(block)
        irreps_in = block.irreps_tensor
    feature_irreps = str(scalar_in)
    dim_feature_out = num_features
    readout_hidden_width = max(num_features // 2, 1)
    blocks = [FnBlock(block, dual_branch_update) for block in blocks]

    return ComposableCore(
        IrrepsEmbed(dim_embedding, num_rbf, sph, species_linear, (cutoff_sr, polynomial_p) if message_envelope else None),
        blocks, FeatureHead(source="scalar"), GraphRequires(distances=False, optional_cutoff_values=False),
        positional_fields=_IRREPS_INPUTS, architecture="eqnorm",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        lmax=lmax, hidden_irreps=hidden_irreps, normalization=normalization,
        message_envelope=message_envelope, num_features=num_features, readout_hidden_width=readout_hidden_width,
        feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        banner=f"""
################################################################################
# EQNorm — tensor branch + scalar accumulator (yzchen08/eqnorm)               #
# {num_layers} layers, lmax={lmax}, hidden {hidden_irreps}, scalar {feature_irreps}
# normalization={normalization}, message_envelope={message_envelope}
################################################################################
""",
    )


def botnet(
    max_Za: int,
    dim_embedding: int,
    num_rbf: int,
    max_ell: int = 2,
    num_interactions: int = 2,
    avg_num_neighbors: float = 1.0,
    hidden_irreps: Optional[str] = None,
    additional_hidden_irreps: Optional[str] = None,
    variant: str = "agnostic",
    interaction_cls_first: Optional[str] = None,
    interaction_cls: Optional[str] = None,
    MLP_irreps: str = "16x0e",
    radial_MLP: Optional[List[int]] = None,
    gate: str = "silu",
    output_mode: str = "layers",
) -> ComposableCore:
    """Stack the irreps from successive body-ordered convolutions."""
    from e3nn.o3 import Irreps, SphericalHarmonics
    from ..botnet.defaults import natural_hidden_irreps
    from ..botnet.interaction import INTERACTION_CLASSES
    from ..botnet.stages import PlacedFeatureHead, SpeciesEmbed, _resolve_interactions, _union_irreps, body_ordered_update
    from ..mp.equivariant.e3nn import scalar_0e_dim

    if output_mode not in {"layers", "feature"}:
        raise ValueError(
            f"output_mode must be 'layers' or 'feature', got {output_mode!r}"
        )
    max_Za = int(max_Za)
    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    max_ell = int(max_ell)
    num_interactions = int(num_interactions)
    avg_num_neighbors = float(avg_num_neighbors)
    if max_Za < 1:
        raise ValueError(f"max_Za must be >= 1, got {max_Za}")
    if dim_embedding < 1:
        raise ValueError(f"dim_embedding must be >= 1, got {dim_embedding}")
    if num_rbf < 1:
        raise ValueError(f"num_rbf must be >= 1, got {num_rbf}")
    if max_ell < 0:
        raise ValueError(f"max_ell must be >= 0, got {max_ell}")
    if num_interactions < 1:
        raise ValueError(
            f"num_interactions must be >= 1, got {num_interactions}"
        )
    if avg_num_neighbors <= 0:
        raise ValueError(
            f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
        )
    if hidden_irreps is not None and additional_hidden_irreps:
        raise ValueError(
            "Pass hidden_irreps or additional_hidden_irreps, not both"
        )
    if hidden_irreps is None:
        if additional_hidden_irreps:
            hidden_irreps = f"{dim_embedding}x0e+{additional_hidden_irreps}"
        else:
            hidden_irreps = natural_hidden_irreps(dim_embedding, max_ell)
    target = Irreps(hidden_irreps).simplify()
    scalar_mul = scalar_0e_dim(target)
    if scalar_mul != dim_embedding:
        raise ValueError(
            "dim_embedding must equal the 0e multiplicity of hidden_irreps "
            f"({scalar_mul}), got {dim_embedding}. The species table is "
            "the initial scalar state."
        )
    requested_hidden_irreps = str(target)
    first_name, rest_name = _resolve_interactions(
        variant, interaction_cls_first, interaction_cls
    )
    interaction_cls_first = first_name
    interaction_cls = rest_name
    if radial_MLP is None:
        radial_MLP = [64, 64, 64]
    radial_MLP = [int(width) for width in radial_MLP]
    if any(width < 1 for width in radial_MLP):
        raise ValueError(f"radial_MLP widths must be >= 1, got {radial_MLP}")
    readout_mlp_irreps = str(Irreps(MLP_irreps).simplify())
    readout_gate = str(gate)
    # One interaction has only the linear head. A second interaction is
    # what makes the final head the scalar MLP in NonLinearBodyOrderedModel.
    last_readout = "nonlinear" if num_interactions > 1 else "linear"

    sh_irreps = Irreps.spherical_harmonics(max_ell)
    spherical_harmonics = SphericalHarmonics(
        sh_irreps, normalize=True, normalization="component"
    )
    node_attrs_irreps = Irreps([(max_Za + 1, (0, 1))])
    edge_feats_irreps = Irreps([(num_rbf, (0, 1))])
    node_feats_irreps = Irreps([(dim_embedding, (0, 1))])
    blocks = []
    layer_irreps: List[Irreps] = []
    for index in range(num_interactions):
        name = first_name if index == 0 else rest_name
        block = INTERACTION_CLASSES[name](
            node_attrs_irreps=node_attrs_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=target,
            hidden_irreps=target,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        blocks.append(FnBlock(block, body_ordered_update))
        produced = Irreps(block.irreps_out).simplify()
        layer_irreps.append(produced)
        node_feats_irreps = produced
    uniform_irreps = all(irreps == layer_irreps[0] for irreps in layer_irreps)
    union = layer_irreps[0] if uniform_irreps else _union_irreps(layer_irreps)
    if output_mode == "layers":
        feature_irreps = str(union)
        num_output_blocks = num_interactions
    else:
        feature_irreps = str(layer_irreps[-1])
        num_output_blocks = 1
    dim_feature_out = scalar_0e_dim(feature_irreps)

    head = FeatureHead(output_mode, source="equi")
    if output_mode == "layers" and not uniform_irreps:
        head = PlacedFeatureHead(layer_irreps, union)
    return ComposableCore(
        SpeciesEmbed(dim_embedding, num_rbf, max_Za, spherical_harmonics), blocks, head,
        GraphRequires(distances=False, optional_cutoff_values=False, edge_convention="swap", fields=("Za",)),
        positional_fields=("Za", "vij_sr", "idx_i_sr", "idx_j_sr", "rbf",
                           "atom_embedding", "charge_embedding", "spin_embedding"),
        architecture="botnet", banner=f"BOTNet ({variant}, {num_interactions} interactions)",
        max_Za=max_Za, dim_embedding=dim_embedding, num_rbf=num_rbf, max_ell=max_ell,
        requested_hidden_irreps=requested_hidden_irreps, avg_num_neighbors=avg_num_neighbors,
        interaction_cls_first=interaction_cls_first, interaction_cls=interaction_cls,
        variant=variant, num_interactions=num_interactions, uniform_irreps=uniform_irreps,
        num_output_blocks=num_output_blocks, layer_irreps=layer_irreps, output_mode=output_mode,
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps, readout_gate=readout_gate,
        readout_mlp_irreps=readout_mlp_irreps, last_readout=last_readout,
    )


def allegro(
    dim_edge: int,
    avg_num_neighbors: float,
    preset: Optional[str] = None,
    num_layers: Optional[int] = None,
    l_max: Optional[int] = None,
    parity: Optional[bool] = None,
    num_scalar_features: Optional[int] = None,
    num_tensor_features: Optional[int] = None,
    scalar_embed_mlp_depth: Optional[int] = None,
    scalar_embed_mlp_width: Optional[int] = None,
    latent_mlp_depth: Optional[int] = None,
    latent_mlp_width: Optional[int] = None,
    nonlinearity: Optional[str] = None,
    tp_path_channel_coupling: Optional[bool] = None,
    weight_individual_irreps: Optional[bool] = None,
) -> ComposableCore:
    """Update scalar and tensor tracks on directed pairs within a fixed radius."""
    from e3nn import o3
    from ..mp.nn.mlp import VariancePreservingMLP
    from ..allegro.defaults import _resolve
    from ..allegro.interaction import ChannelTensorProduct, WeightedSphericalChannels
    from ..allegro.irreps import allegro_layer_irreps, edge_sh_irreps
    from ..allegro.stages import PairEmbed, PairHead, PairProduct, pair_stack

    cfg = _resolve(
        preset,
        {
            "num_layers": num_layers,
            "l_max": l_max,
            "parity": parity,
            "num_scalar_features": num_scalar_features,
            "num_tensor_features": num_tensor_features,
            "scalar_embed_mlp_depth": scalar_embed_mlp_depth,
            "scalar_embed_mlp_width": scalar_embed_mlp_width,
            "latent_mlp_depth": latent_mlp_depth,
            "latent_mlp_width": latent_mlp_width,
            "nonlinearity": nonlinearity,
            "tp_path_channel_coupling": tp_path_channel_coupling,
            "weight_individual_irreps": weight_individual_irreps,
        },
    )
    dim_edge = int(dim_edge)
    num_layers = int(cfg["num_layers"])
    l_max = int(cfg["l_max"])
    parity = bool(cfg["parity"])
    num_scalar_features = int(cfg["num_scalar_features"])
    num_tensor_features = int(cfg["num_tensor_features"])
    nonlinearity = str(cfg["nonlinearity"])
    tp_path_channel_coupling = bool(cfg["tp_path_channel_coupling"])
    weight_individual_irreps = bool(cfg["weight_individual_irreps"])
    avg_num_neighbors = float(avg_num_neighbors)
    scalar_embed_mlp_depth = int(cfg["scalar_embed_mlp_depth"])
    latent_mlp_depth = int(cfg["latent_mlp_depth"])
    scalar_width = cfg["scalar_embed_mlp_width"]
    latent_width = cfg["latent_mlp_width"]
    scalar_embed_mlp_width = (
        num_scalar_features if scalar_width is None else int(scalar_width)
    )
    latent_mlp_width = (
        num_scalar_features if latent_width is None else int(latent_width)
    )
    # EdgeReadout uses this when its hidden width is left unset.
    readout_hidden_width = num_scalar_features
    if dim_edge < 2 or dim_edge % 2 != 0:
        raise ValueError(
            f"dim_edge must be a positive even integer, got {dim_edge}"
        )
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if l_max < 0:
        raise ValueError(f"l_max must be >= 0, got {l_max}")
    if num_scalar_features < 1:
        raise ValueError(
            f"num_scalar_features must be >= 1, got {num_scalar_features}"
        )
    if num_tensor_features < 1:
        raise ValueError(
            f"num_tensor_features must be >= 1, got {num_tensor_features}"
        )
    if avg_num_neighbors <= 0:
        raise ValueError(
            f"avg_num_neighbors must be positive, got {avg_num_neighbors}"
        )
    if scalar_embed_mlp_depth < 0 or latent_mlp_depth < 0:
        raise ValueError("MLP depths must be >= 0")
    if scalar_embed_mlp_depth > 0 and scalar_embed_mlp_width < 1:
        raise ValueError("scalar_embed_mlp_width must be >= 1 when depth > 0")
    if latent_mlp_depth > 0 and latent_mlp_width < 1:
        raise ValueError("latent_mlp_width must be >= 1 when depth > 0")

    def mlp(n_in: int, n_out: int, depth: int, width: int) -> VariancePreservingMLP:
        return VariancePreservingMLP(
            n_in,
            n_out,
            hidden_layers_depth=depth,
            hidden_layers_width=width if depth > 0 else None,
            nonlinearity=nonlinearity if depth > 0 else None,
            bias=False,
            forward_weight_init=True,
        )

    def linear(n_in: int, n_out: int) -> VariancePreservingMLP:
        return VariancePreservingMLP(
            n_in,
            n_out,
            hidden_layers_depth=0,
            nonlinearity=None,
            bias=False,
            forward_weight_init=True,
        )

    irreps_edge_sh = edge_sh_irreps(l_max)
    sph = o3.SphericalHarmonics(
        irreps_edge_sh, normalize=True, normalization="component"
    )
    weighter = WeightedSphericalChannels(
        irreps_edge_sh,
        num_tensor_features,
        weight_individual_irreps=weight_individual_irreps,
    )
    scalar_mlp = mlp(
        dim_edge,
        num_scalar_features,
        scalar_embed_mlp_depth,
        scalar_embed_mlp_width,
    )
    tensor_weight = linear(
        num_scalar_features, weighter.weight_numel
    )
    pair_projection = linear(
        num_scalar_features,
        num_scalar_features + weighter.weight_numel,
    )
    pairs = allegro_layer_irreps(l_max, parity, num_layers)
    products = []
    latents = []
    for layer, (irreps_in, irreps_out) in enumerate(pairs):
        products.append(
            ChannelTensorProduct(
                irreps_in,
                irreps_edge_sh,
                irreps_out,
                mul=num_tensor_features,
                path_channel_coupling=tp_path_channel_coupling,
            )
        )
        last = layer == num_layers - 1
        latents.append(
            mlp(
                num_scalar_features * (layer + 1) + num_tensor_features,
                num_scalar_features
                + (0 if last else weighter.weight_numel),
                latent_mlp_depth,
                latent_mlp_width,
            )
        )
    dim_feature_out = num_scalar_features * (num_layers + 1)
    feature_irreps = f"{dim_feature_out}x0e"

    embed = PairEmbed(scalar_mlp, tensor_weight, pair_projection, weighter, sph,
                      dim_edge, num_scalar_features, avg_num_neighbors)
    blocks = [PairProduct(tp, latent, weighter, num_scalar_features, num_tensor_features)
              for tp, latent in zip(products, latents)]
    return ComposableCore(
        embed, blocks, PairHead(), GraphRequires(distances=False, rbf=False, optional_cutoff_values=False),
        node_field="edge_embedding", builder=pair_stack,
        positional_fields=("edge_embedding", "idx_i_sr", "idx_j_sr", "vij_sr"),
        architecture="allegro", banner=f"Allegro ({num_layers} layers, l_max={l_max})",
        dim_edge=dim_edge, num_layers=num_layers, l_max=l_max, parity=parity,
        readout_hidden_width=readout_hidden_width, dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps, avg_num_neighbors=avg_num_neighbors,
        num_scalar_features=num_scalar_features, num_tensor_features=num_tensor_features,
    )


def equiformer(
    num_rbf: int,
    irreps_node_embedding: str = "32x0e+16x1e+8x2e",
    irreps_feature: str = "64x0e",
    irreps_node_attr: str = "1x0e",
    irreps_sh: str = "1x0e+1x1e+1x2e",
    irreps_head: str = "16x0e+8x1o+4x2e",
    irreps_mlp_mid: str = "32x0e+16x1e+8x2e",
    irreps_pre_attn: Optional[str] = None,
    num_layers: int = 2,
    num_heads: int = 2,
    fc_neurons: Optional[List[int]] = None,
    rescale_degree: bool = False,
    nonlinear_message: bool = True,
    use_attn_head: bool = False,
    norm_layer: str = "layer",
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
    out_drop: float = 0.0,
    drop_path_rate: float = 0.0,
    avg_degree: float = _AVG_DEGREE,
    avg_num_nodes: float = _AVG_NUM_NODES,
    output_mode: Literal["direct", "feature"] = "feature",
    **kwargs,
) -> ComposableCore:
    """Compose degree embeddings, graph attention, and a property or feature head."""
    import torch
    from e3nn import o3
    from ..mp.equivariant.e3nn import Activation, EquivariantDropout, LinearRS, scalar_0e_dim
    from ..equiformer.embedding import EdgeDegreeEmbeddingNetwork
    from ..equiformer.interaction import GraphAttention, TransBlock, get_norm_layer, _RESCALE
    from ..equiformer.stages import DegreeEmbed, graph_attention_update, AttentionHead

    if fc_neurons is None:
        fc_neurons = [64, 64]

    irreps_node_embedding = o3.Irreps(irreps_node_embedding)
    irreps_feature = o3.Irreps(irreps_feature)
    irreps_node_attr = o3.Irreps(irreps_node_attr)
    irreps_edge_attr = o3.Irreps(irreps_sh)
    irreps_head = o3.Irreps(irreps_head)
    irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)
    fc_neurons = [num_rbf] + list(fc_neurons)
    # Scalar readout contract: dim_feature_out is 0e channel count.
    feature_irreps = str(irreps_feature)
    dim_feature_out = scalar_0e_dim(irreps_feature)

    edge_deg_embed = EdgeDegreeEmbeddingNetwork(
        irreps_node_embedding,
        irreps_edge_attr,
        fc_neurons,
        avg_degree,
    )
    blocks = []
    for i in range(num_layers):
        irreps_block_output = (
            irreps_node_embedding
            if i != (num_layers - 1)
            else irreps_feature
        )
        blocks.append(
            TransBlock(
                irreps_node_input=irreps_node_embedding,
                irreps_node_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=irreps_block_output,
                fc_neurons=fc_neurons,
                irreps_head=irreps_head,
                num_heads=num_heads,
                irreps_pre_attn=irreps_pre_attn,
                rescale_degree=rescale_degree,
                nonlinear_message=nonlinear_message,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                drop_path_rate=drop_path_rate,
                irreps_mlp_mid=irreps_mlp_mid,
                norm_layer=norm_layer,
            )
        )
    norm = get_norm_layer(norm_layer)(irreps_feature)
    out_dropout = None
    if out_drop != 0.0:
        out_dropout = EquivariantDropout(irreps_feature, out_drop)

    readout = None
    if output_mode == "direct":
        if use_attn_head:
            readout = GraphAttention(
                irreps_node_input=irreps_feature,
                irreps_node_attr=irreps_node_attr,
                irreps_edge_attr=irreps_edge_attr,
                irreps_node_output=o3.Irreps("1x0e"),
                fc_neurons=fc_neurons,
                irreps_head=irreps_head,
                num_heads=num_heads,
                irreps_pre_attn=irreps_pre_attn,
                rescale_degree=rescale_degree,
                nonlinear_message=nonlinear_message,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
            )
        else:
            readout = torch.nn.Sequential(
                LinearRS(irreps_feature, irreps_feature, rescale=_RESCALE),
                Activation(irreps_feature, acts=[torch.nn.SiLU()]),
                LinearRS(irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
            )

    return ComposableCore(
        DegreeEmbed(edge_deg_embed, irreps_edge_attr),
        [FnBlock(block, graph_attention_update) for block in blocks],
        AttentionHead(norm, out_dropout, readout, use_attn_head),
        GraphRequires(distances=False, optional_cutoff_values=False, optional_fields=("batch_seg",)),
        positional_fields=("vij_sr", "idx_i_sr", "idx_j_sr", "rbf", "atom_embedding", "batch_seg"),
        architecture="equiformer", banner="Equiformer (ICLR 2023)",
        output_mode=output_mode, feature_irreps=feature_irreps, dim_feature_out=dim_feature_out,
        irreps_feature=irreps_feature, irreps_edge_attr=irreps_edge_attr, use_attn_head=use_attn_head,
        num_layers=num_layers, num_heads=num_heads, num_rbf=num_rbf, irreps_node_embedding=irreps_node_embedding,
    )
