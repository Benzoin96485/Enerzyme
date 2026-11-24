"""e3nn irreps message-passing presets.

``mace`` loads ``main``-branch checkpoints through
:mod:`enerzyme.models.compat.main_branch`; its parameter names and numerics
follow ``main``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Union

from ..mp.core import ComposableCore, FeatureHead, FieldEmbed, ReadoutHead
from ..mp.graph import GraphRequires

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
