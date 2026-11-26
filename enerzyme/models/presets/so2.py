"""Presets for spherical-channel and edge-frame message passing."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Union

import torch
from torch import nn

from ..esen.native.defaults import _RESCALE
from ..equiformer_v2.defaults import _AVG_DEGREE
from ..mp.block import FnBlock
from ..mp.core import ComposableCore, FeatureHead
from ..mp.graph import GraphRequires, RotationRequest
from ..mp.message.so2 import (
    ChannelBlock, ChannelEmbed, FusedChannelEmbed, SmoothChannelEmbed, SphereHead,
    channel_attention, fused_attention, smooth_convolution,
)

_CHANNEL_INPUTS = ("atom_embedding", "Za", "rbf", "idx_i_sr", "idx_j_sr", "vij_sr")


def escn(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    sphere_channels: int = 64,
    hidden_channels: int = 128,
    edge_channels: int = 64,
    lmax: int = 2,
    mmax: int = 2,
    num_layers: int = 2,
    resolution: Optional[int] = None,
) -> ComposableCore:
    """Apply spherical-channel convolutions in bond-aligned frames."""
    from torch.nn import SiLU
    from ..escn.interaction import LayerBlock
    from ..mp.equivariant import build_so3_grid_table

    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")

    lmax_list = [lmax]
    mmax_list = [mmax]
    num_resolutions = len(lmax_list)
    # Align with shared SimpleReadout contract: dim_feature_out = 0e width;
    # feature_irreps advertises that atom_feature is pure even scalars.
    dim_feature_out = num_resolutions * sphere_channels
    feature_irreps = f"{dim_feature_out}x0e"

    act = SiLU()
    sphere_proj = torch.nn.Linear(dim_embedding, dim_feature_out)

    SO3_grid = build_so3_grid_table(
        max(lmax_list),
        normalization="integral",
        resolution=resolution,
        rescale_by_mmax=False,
    )

    layer_blocks = []
    for _ in range(num_layers):
        layer_blocks.append(
            LayerBlock(
                sphere_channels,
                hidden_channels,
                edge_channels,
                lmax_list,
                mmax_list,
                num_rbf,
                max_Za,
                SO3_grid,
                act,
            )
        )
    return ComposableCore(
        ChannelEmbed(sphere_proj, lmax, mmax),
        [ChannelBlock(block, residual=i > 0) for i, block in enumerate(layer_blocks)], SphereHead(wrapped=True),
        GraphRequires(rotations=(RotationRequest(lmax, "jd", edge_sign=-1, frame_field="edge_rot_mat"),),
                      distances=False, optional_cutoff_values=False, fields=("Za",),
                      optional_fields=("edge_rot_mat",),
                      edge_convention="swap"),
        positional_fields=_CHANNEL_INPUTS, architecture="escn",
        lmax_list=lmax_list, mmax_list=mmax_list, sphere_channels=sphere_channels,
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        num_resolutions=num_resolutions, dim_feature_out=dim_feature_out, feature_irreps=feature_irreps,
    )


def esen(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    sphere_channels: int = 16,
    hidden_channels: int = 16,
    edge_channels: int = 16,
    lmax: int = 2,
    mmax: int = 2,
    num_layers: int = 2,
    norm_type: str = "rms_norm_sh",
    act_type: str = "gate",
    mlp_type: str = "spectral",
    use_envelope: bool = True,
    rescale_factor: float = _RESCALE,
    cutoff_sr: float = 6.0,
    grid_resolution: Optional[int] = None,
) -> ComposableCore:
    """Compose smooth edge-degree embedding and spherical eSEN updates."""
    from ..esen.native.interaction import EdgeDegreeEmbedding, eSENBlock
    from ..mp.equivariant import CoefficientMapping, build_so3_grid_table, get_normalization_layer

    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")
    if lmax < 1:
        raise ValueError(f"lmax must be >= 1, got {lmax}")
    if rescale_factor <= 0:
        raise ValueError(f"rescale_factor must be positive, got {rescale_factor}")
    if act_type not in ("gate", "s2"):
        raise ValueError(f"Unknown act_type {act_type!r}")
    if mlp_type not in ("spectral", "grid"):
        raise ValueError(f"Unknown mlp_type {mlp_type!r}")

    dim_embedding = int(dim_embedding)
    num_rbf = int(num_rbf)
    max_Za = int(max_Za)
    sphere_channels = int(sphere_channels)
    hidden_channels = int(hidden_channels)
    edge_channels = int(edge_channels)
    lmax_list = [int(lmax)]
    mmax_list = [int(mmax)]
    num_layers = int(num_layers)
    use_envelope = bool(use_envelope)
    rescale_factor = float(rescale_factor)
    cutoff_sr = float(cutoff_sr)
    num_resolutions = 1
    dim_feature_out = sphere_channels
    feature_irreps = f"{dim_feature_out}x0e"

    sphere_proj = nn.Linear(dim_embedding, sphere_channels)
    source_embedding = nn.Embedding(max_Za + 1, edge_channels)
    target_embedding = nn.Embedding(max_Za + 1, edge_channels)
    nn.init.uniform_(source_embedding.weight.data, -0.001, 0.001)
    nn.init.uniform_(target_embedding.weight.data, -0.001, 0.001)

    mapping = CoefficientMapping(
        lmax_list, mmax_list, torch.device("cpu")
    )
    edge_channels_list = [
        num_rbf + 2 * edge_channels,
        edge_channels,
        edge_channels,
    ]

    edge_degree = EdgeDegreeEmbedding(
        sphere_channels,
        lmax_list,
        mmax_list,
        mapping,
        edge_channels_list,
        rescale_factor,
        cutoff_sr,
        use_envelope,
    )

    SO3_grid = None
    if act_type == "s2" or mlp_type == "grid":
        SO3_grid = build_so3_grid_table(
            lmax,
            normalization="component",
            resolution=grid_resolution,
            rescale_by_mmax=True,
        )

    blocks = nn.ModuleList(
        [
            eSENBlock(
                sphere_channels,
                hidden_channels,
                lmax,
                mmax,
                mapping,
                edge_channels_list,
                cutoff_sr,
                norm_type,
                act_type,
                mlp_type,
                use_envelope,
                SO3_grid=SO3_grid,
            )
            for _ in range(num_layers)
        ]
    )
    norm = get_normalization_layer(
        norm_type, lmax=lmax, num_channels=sphere_channels
    )
    return ComposableCore(
        SmoothChannelEmbed(sphere_proj, source_embedding, target_embedding, edge_degree, lmax, mapping),
        [FnBlock(block, smooth_convolution) for block in blocks], SphereHead(norm),
        GraphRequires(rotations=(RotationRequest(lmax, "jd", frame_field="edge_rot_mat"),),
                      distances=False, optional_cutoff_values=False, fields=("Za",),
                      optional_fields=("edge_rot_mat",),
                      edge_convention="receiver_i"),
        positional_fields=_CHANNEL_INPUTS, architecture="esen",
        dim_embedding=dim_embedding, num_rbf=num_rbf, lmax_list=lmax_list,
        mmax_list=mmax_list, sphere_channels=sphere_channels, num_layers=num_layers,
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps, cutoff_sr=cutoff_sr,
    )


def equiformer_v2(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    sphere_channels: int = 64,
    attn_hidden_channels: int = 64,
    num_heads: int = 4,
    attn_alpha_channels: int = 32,
    attn_value_channels: int = 16,
    ffn_hidden_channels: int = 128,
    lmax: int = 2,
    mmax: int = 2,
    num_layers: int = 2,
    edge_channels: int = 64,
    norm_type: str = "rms_norm_sh",
    grid_resolution: Optional[int] = None,
    use_atom_edge_embedding: bool = True,
    use_m_share_rad: bool = False,
    attn_activation: str = "scaled_silu",
    use_s2_act_attn: bool = False,
    use_attn_renorm: bool = True,
    ffn_activation: str = "scaled_silu",
    use_gate_act: bool = False,
    use_grid_mlp: bool = False,
    use_sep_s2_act: bool = True,
    alpha_drop: float = 0.0,
    drop_path_rate: float = 0.0,
    proj_drop: float = 0.0,
    avg_degree: float = _AVG_DEGREE,
) -> ComposableCore:
    """Use inverse-rescaled Wigner rotations for spherical attention."""
    from ..equiformer_v2.input_block import EdgeDegreeEmbedding
    from ..equiformer_v2.interaction import TransBlockV2
    from ..mp.equivariant import CoefficientMapping, build_so3_grid_table, get_normalization_layer

    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")

    max_num_elements = max_Za + 1
    lmax_list = [lmax]
    mmax_list = [mmax]
    num_resolutions = len(lmax_list)

    dim_feature_out = num_resolutions * sphere_channels
    feature_irreps = f"{dim_feature_out}x0e"

    sphere_proj = torch.nn.Linear(dim_embedding, dim_feature_out)


    # Mapping lives on CPU at init; tensors move with module / are .to()'d in ops.
    mappingReduced = CoefficientMapping(
        lmax_list, mmax_list, torch.device("cpu")
    )

    SO3_grid = build_so3_grid_table(
        max(lmax_list),
        normalization="component",
        resolution=grid_resolution,
        rescale_by_mmax=True,
    )

    # edge_channels_list: [num_rbf, edge_channels, edge_channels]
    edge_channels_list = [num_rbf] + [edge_channels] * 2

    edge_degree_embedding = EdgeDegreeEmbedding(
        sphere_channels,
        lmax_list,
        mmax_list,
        None,
        mappingReduced,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding,
        rescale_factor=avg_degree,
    )

    blocks = []
    for _ in range(num_layers):
        blocks.append(
            TransBlockV2(
                sphere_channels,
                attn_hidden_channels,
                num_heads,
                attn_alpha_channels,
                attn_value_channels,
                ffn_hidden_channels,
                sphere_channels,
                lmax_list,
                mmax_list,
                None,
                mappingReduced,
                SO3_grid,
                max_num_elements,
                edge_channels_list,
                use_atom_edge_embedding,
                use_m_share_rad,
                attn_activation,
                use_s2_act_attn,
                use_attn_renorm,
                ffn_activation,
                use_gate_act,
                use_grid_mlp,
                use_sep_s2_act,
                norm_type,
                alpha_drop,
                drop_path_rate,
                proj_drop,
            )
        )

    norm = get_normalization_layer(
        norm_type, lmax=max(lmax_list), num_channels=sphere_channels
    )

    # Expose FFN hyperparameters for optional EquiformerV2FeedForwardReadout.
    use_sep_s2_act = use_sep_s2_act
    return ComposableCore(
        ChannelEmbed(sphere_proj, lmax, mmax, edge_degree_embedding, mappingReduced),
        [FnBlock(block, channel_attention) for block in blocks], SphereHead(norm, wrapped=True),
        GraphRequires(rotations=(RotationRequest(lmax, "jd", edge_sign=-1, frame_field="edge_rot_mat"),),
                      distances=False, optional_cutoff_values=False, fields=("Za",),
                      optional_fields=("edge_rot_mat", "batch_seg",),
                      edge_convention="swap"),
        positional_fields=_CHANNEL_INPUTS + ("batch_seg",), architecture="equiformer_v2",
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps, lmax_list=lmax_list,
        mmax_list=mmax_list, sphere_channels=sphere_channels, ffn_hidden_channels=ffn_hidden_channels,
        SO3_grid=SO3_grid, ffn_activation=ffn_activation, use_gate_act=use_gate_act,
        use_grid_mlp=use_grid_mlp, use_sep_s2_act=use_sep_s2_act, dim_embedding=dim_embedding,
        num_rbf=num_rbf, num_layers=num_layers, num_heads=num_heads,
        norm_type=norm_type,
    )


def equiformer_v3(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    sphere_channels: int = 64,
    attn_hidden_channels: int = 64,
    num_heads: int = 4,
    attn_alpha_channels: int = 32,
    attn_value_channels: int = 16,
    ffn_hidden_channels: int = 256,
    lmax: int = 2,
    mmax: int = 2,
    num_layers: int = 2,
    edge_channels: int = 64,
    norm_type: str = "merge_layer_norm",
    use_atom_edge_embedding: bool = True,
    use_envelope: bool = True,
    envelope_exponent: int = 5,
    attn_activation: str = "sep-merge_gates2_swiglu",
    ffn_activation: str = "sep-merge_gates2_swiglu",
    use_grid_mlp: bool = True,
    use_attn_renorm: bool = True,
    use_add_merge: bool = False,
    use_rad_l_parametrization: bool = True,
    softcap: Optional[float] = None,
    attn_eps: float = 1e-16,
    attn_grid_resolution: Optional[Sequence[int]] = None,
    ffn_grid_resolution: Optional[Sequence[int]] = None,
    alpha_drop: float = 0.0,
    attn_mask_rate: float = 0.0,
    attn_weights_drop: float = 0.0,
    value_drop: float = 0.0,
    drop_path_rate: float = 0.0,
    proj_drop: float = 0.0,
    ffn_drop: float = 0.0,
    avg_degree: float = _AVG_DEGREE,
    cutoff_sr: float = 5.0,
) -> ComposableCore:
    """Use fused Wigner rotations and envelope-weighted graph attention."""
    from ..equiformer_v3.defaults import _as_resolution_list
    from ..equiformer_v3.input_block import EdgeDegreeEmbedding
    from ..equiformer_v3.interaction import TransBlockV3
    from ..mp.equivariant import PolynomialEnvelope, SO3RotationFused, get_normalization_layer

    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")

    max_num_elements = max_Za + 1
    num_channels = sphere_channels

    attn_grid_resolution_list = _as_resolution_list(
        attn_grid_resolution, [2 * (lmax + 1), 2 * (mmax + 1) + 1]
    )
    ffn_grid_resolution_list = _as_resolution_list(
        ffn_grid_resolution, [2 * (lmax + 1), 2 * (lmax + 1)]
    )

    dim_feature_out = sphere_channels
    feature_irreps = f"{dim_feature_out}x0e"

    sphere_proj = torch.nn.Linear(dim_embedding, sphere_channels)

    so3_rotation = SO3RotationFused(
        lmax, mmax, use_rotation_mask=False
    )

    edge_channels_list = [num_rbf] + [edge_channels] * 2

    envelope_func = (
        PolynomialEnvelope(cutoff=cutoff_sr, exponent=envelope_exponent)
        if use_envelope
        else None
    )

    edge_degree_embedding = EdgeDegreeEmbedding(
        num_channels=sphere_channels,
        lmax=lmax,
        mmax=mmax,
        so3_rotation=None,
        max_num_elements=max_num_elements,
        edge_channels_list=edge_channels_list,
        use_atom_edge_embedding=use_atom_edge_embedding,
        rescale_factor=avg_degree,
    )

    blocks = []
    for _ in range(num_layers):
        blocks.append(
            TransBlockV3(
                num_in_channels=sphere_channels,
                attn_hidden_channels=attn_hidden_channels,
                num_heads=num_heads,
                attn_alpha_channels=attn_alpha_channels,
                attn_value_channels=attn_value_channels,
                ffn_hidden_channels=ffn_hidden_channels,
                num_out_channels=sphere_channels,
                lmax=lmax,
                mmax=mmax,
                so3_rotation=None,
                attn_grid_resolution_list=attn_grid_resolution_list,
                ffn_grid_resolution_list=ffn_grid_resolution_list,
                max_num_elements=max_num_elements,
                edge_channels_list=edge_channels_list,
                use_atom_edge_embedding=use_atom_edge_embedding,
                attn_activation=attn_activation,
                use_attn_renorm=use_attn_renorm,
                use_add_merge=use_add_merge,
                use_rad_l_parametrization=use_rad_l_parametrization,
                softcap=softcap,
                attn_eps=attn_eps,
                ffn_activation=ffn_activation,
                use_grid_mlp=use_grid_mlp,
                norm_type=norm_type,
                alpha_drop=alpha_drop,
                attn_mask_rate=attn_mask_rate,
                attn_weights_drop=attn_weights_drop,
                value_drop=value_drop,
                drop_path_rate=drop_path_rate,
                proj_drop=proj_drop,
                ffn_drop=ffn_drop,
            )
        )

    norm = get_normalization_layer(
        norm_type, lmax=lmax, num_channels=sphere_channels
    )
    return ComposableCore(
        FusedChannelEmbed(sphere_proj, lmax, so3_rotation, edge_degree_embedding, envelope_func),
        [FnBlock(block, fused_attention) for block in blocks], SphereHead(norm),
        GraphRequires(rotations=(RotationRequest(lmax, so3_rotation, mmax, -1, "edge_rot_mat"),),
                      distances=False, optional_cutoff_values=False, fields=("Za",),
                      optional_fields=("edge_rot_mat", "batch_seg", "edge_envelope_weight",),
                      edge_convention="swap"),
        positional_fields=_CHANNEL_INPUTS + ("batch_seg",), architecture="equiformer_v3",
        _cpu_force_training_option="cpp.simdlen",
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps, lmax=lmax,
        mmax=mmax, sphere_channels=sphere_channels, ffn_hidden_channels=ffn_hidden_channels,
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        num_heads=num_heads, norm_type=norm_type,
    )


def dpa4(
    dim_embedding: int,
    max_Za: int = 94,
    channels: int = 32,
    lmax: int = 2,
    mmax: int = 1,
    n_blocks: int = 2,
    mixing_layers: int = 3,
    n_focus: int = 1,
    focus_dim: int = 0,
    n_atten_head: int = 1,
    radial_so2_rank: int = 1,
    radial_so2_mode: str = "degree_channel",
    ffn_so3_grid: bool = True,
    message_node_so3: bool = False,
    use_env_seed: bool = True,
    n_radial: int = 16,
    basis_type: str = "bessel",
    cutoff_sr: float = 6.0,
    sandwich_norm: Optional[list] = None,
    ffn_blocks: int = 1,
    ffn_neurons: int = 96,
    glu_activation: bool = True,
    lebedev_quadrature: bool = True,
    envelope_exponent: int = 5,
) -> ComposableCore:
    """Combine radial focus channels with geometric and environment node embeddings."""
    from ..dpa4.embedding import (EnvironmentInitialEmbedding, FocusEmbed, GeometricInitialEmbedding,
                                  focus_update, internal_radial_stack)
    from ..dpa4.interaction import SeZMInteractionBlock
    from ..mp.basis.radial import BesselC3RadialBasis
    from ..mp.equivariant import C3CutoffEnvelope
    from ..mp.equivariant.rotation.wigner_quaternion import WignerDCalculator
    from ..mp.equivariant.sph.layer_norm import EquivariantDegreeRMSNorm
    from ..mp.nn.radial_mlp import radial_mlp

    if mmax > lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed lmax ({lmax})")


    dim_feature_out = channels
    feature_irreps = f"{channels}x0e"

    # Internal radial basis and envelope (no external RBF needed)
    radial_basis = BesselC3RadialBasis(
        rcut=cutoff_sr, n_radial=n_radial,
        basis_type=basis_type, exponent=envelope_exponent + 2,
    )
    envelope = C3CutoffEnvelope(rcut=cutoff_sr, exponent=envelope_exponent)

    # Wigner-D calculator
    wigner_calc = WignerDCalculator(lmax=lmax)

    # Project atom embedding to sphere channels
    sphere_proj = nn.Linear(dim_embedding, channels)

    # Radial feature MLP: n_radial -> (lmax+1) * channels
    radial_feat_mlp = radial_mlp(
        n_radial,
        (lmax + 1) * channels,
        hidden=[64],
    )

    # Geometric initial embedding
    gie = GeometricInitialEmbedding(lmax=lmax, channels=channels)

    # Environment initial embedding (FiLM on l=0)
    if use_env_seed:
        env_embed = EnvironmentInitialEmbedding(
            ntypes=max_Za + 1,
            n_radial=n_radial,
            channels=channels,
        )
    else:
        env_embed = None

    # Interaction blocks
    if sandwich_norm is None:
        sandwich_norm = [False, True, True, False]
    blocks = []
    for _ in range(n_blocks):
        blocks.append(SeZMInteractionBlock(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            n_focus=n_focus,
            focus_dim=focus_dim,
            mixing_layers=mixing_layers,
            n_atten_head=n_atten_head,
            radial_so2_mode=radial_so2_mode,
            radial_so2_rank=radial_so2_rank,
            n_radial=n_radial,
            ffn_neurons=ffn_neurons,
            ffn_blocks=ffn_blocks,
            ffn_so3_grid=ffn_so3_grid,
            lebedev_quadrature=lebedev_quadrature,
            glu_activation=glu_activation,
            sandwich_norm=sandwich_norm,
            message_node_so3=message_node_so3,
        ))

    # Final norm
    norm = EquivariantDegreeRMSNorm(lmax, channels)
    return ComposableCore(
        FocusEmbed(radial_basis, envelope, wigner_calc, sphere_proj, radial_feat_mlp, gie, env_embed, lmax, channels),
        [FnBlock(block, focus_update) for block in blocks], SphereHead(norm, focus=True),
        GraphRequires(distances=False, rbf=False, optional_cutoff_values=False, fields=("Za",),
                      optional_fields=("batch_seg",), edge_convention="swap"),
        positional_fields=("atom_embedding", "Za", "idx_i_sr", "idx_j_sr", "vij_sr", "batch_seg"), architecture="dpa4", builder=internal_radial_stack,
        _partitioned_norm_force_dtype=torch.float64,
        lmax=lmax, mmax=mmax, channels=channels,
        dim_embedding=dim_embedding, max_Za=max_Za, n_blocks=n_blocks,
        cutoff_sr=cutoff_sr, n_radial=n_radial, dim_feature_out=dim_feature_out,
        feature_irreps=feature_irreps,
    )


def tece(
    max_Za: int,
    dim_embedding: int,
    num_rbf: int,
    num_channel: int = 48,
    num_layers: int = 2,
    Lmax: int = 2,
    lmax: int = 2,
    mmax: Optional[int] = None,
    correlation: Union[int, List[int]] = 2,
    avg_num_neighbors: float = 8.0,
    edge_embedding: str = "nonlinear",
    edge_update: str = "element2",
    scatter_norm: str = "avg_num_neighbors",
    resnet_type: str = "BB",
    resnet_linear_type: str = "aware",
    use_first_resnet: bool = False,
    nonlinear: str = "sigmoid_gate",
    edge_nonlinear: str = "so2_sigmoid_gate",
    radial_mlp: Optional[List[int]] = None,
    radial_bias: bool = False,
    use_so2_edge_ace: bool = True,
    use_graph_softmax: bool = True,
    use_radial_phase: bool = True,
    use_temperature: bool = True,
    num_head: int = 1,
    edge_ace_hidden: Optional[int] = None,
    edge_wise_hidden: Optional[int] = None,
    so2_linear_type: str = "w1",
    gate_m0: bool = False,
    scalar_act: Optional[str] = None,
    tensor_act: Optional[str] = None,
    wigner_type: Literal["recursive", "direct"] = "recursive",
    bias: bool = True,
    agnostic_product: bool = False,
):
    """Compose edge updates, uvSO(2) interactions and cluster-expansion products."""
    from e3nn.o3 import Irreps, SphericalHarmonics
    from ..mp.equivariant import WignerD
    from ..mp.equivariant.e3nn import IrrepsLinear, scalar_0e_dim
    from ..tace.interaction import CgtpACE, EDGE_EMBEDDING, EDGE_UPDATE
    from ..tece.interaction import UvSO2Interaction
    from ..tece.stages import ClusterEmbed, ClusterExpansionBlock, _TensorSeed

    if Lmax != lmax:
        raise ValueError(
            f"TECE requires Lmax == lmax (SO2); got Lmax={Lmax}, lmax={lmax}"
        )
    if mmax is None:
        mmax = Lmax
    if mmax > Lmax:
        raise ValueError(f"mmax ({mmax}) cannot exceed Lmax ({Lmax})")
    if radial_mlp is None:
        radial_mlp = [64, 64]
    if isinstance(correlation, int):
        correlation = [correlation] * num_layers
    else:
        correlation = list(correlation)
        if len(correlation) != num_layers:
            raise ValueError(
                f"correlation list length ({len(correlation)}) must equal "
                f"num_layers ({num_layers}); got {correlation!r}"
            )

    num_elements = max_Za + 1

    target = Irreps([(1, (l, (-1) ** l)) for l in range(Lmax + 1)])
    target_irreps = target

    if dim_embedding != num_channel:
        node_proj = IrrepsLinear(
            f"{dim_embedding}x0e", f"{num_channel}x0e", bias=False
        )
    else:
        node_proj = torch.nn.Identity()

    spherical_harmonics = SphericalHarmonics(
        Irreps.spherical_harmonics(lmax, p=-1),
        normalize=True,
        normalization="component",
    )
    wigner = WignerD(
        mmax=mmax, lmax=Lmax, wigner_type=wigner_type
    )

    edge_embedding_mod = EDGE_EMBEDDING[edge_embedding](
        num_radial_basis=num_rbf,
        num_channel=num_channel,
        num_elements=num_elements,
        bias=False,
    )
    edge_emb_dim = edge_embedding_mod.out_dim

    tensor_seed = _TensorSeed(
        num_elements=num_elements,
        num_channel=num_channel,
        Lmax=Lmax,
        lmax=lmax,
        num_radial_basis=num_rbf,
        avg_num_neighbors=avg_num_neighbors,
        bias=bias,
    )

    blocks = []

    irreps_in = tensor_seed.irreps_out
    for layer in range(num_layers):
        eu = EDGE_UPDATE[edge_update](
            num_elements=num_elements,
            num_channel=num_channel,
            edge_embedding_channel=edge_emb_dim,
            bias=False,
        )
        inter = UvSO2Interaction(
            layer=layer,
            num_layers=num_layers,
            num_elements=num_elements,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            mmax=mmax,
            irreps_in=irreps_in,
            edge_feats_channel=eu.out_dim,
            num_radial_basis=num_rbf,
            avg_num_neighbors=avg_num_neighbors,
            radial_mlp=radial_mlp,
            radial_bias=radial_bias,
            scatter_norm=scatter_norm,
            nonlinear=nonlinear,
            edge_nonlinear=edge_nonlinear,
            resnet_type=resnet_type,
            resnet_linear_type=resnet_linear_type,
            use_first_resnet=use_first_resnet,
            parity=False,
            bias=bias,
            target_irreps=target,
            correlation=correlation[layer],
            use_so2_edge_ace=use_so2_edge_ace,
            use_graph_softmax=use_graph_softmax,
            use_radial_phase=use_radial_phase,
            use_temperature=use_temperature,
            num_head=num_head,
            edge_ace_hidden=edge_ace_hidden,
            edge_wise_hidden=edge_wise_hidden,
            so2_linear_type=so2_linear_type,
            gate_m0=gate_m0,
            scalar_act=scalar_act,
            tensor_act=tensor_act,
        )
        prod = CgtpACE(
            layer=layer,
            num_layers=num_layers,
            num_elements=num_elements,
            num_channel=num_channel,
            Lmax=Lmax,
            lmax=lmax,
            irreps_in=inter.irreps_out,
            correlation=correlation[layer],
            target_irreps=target,
            l1l2=None,
            parity=False,
            bias=bias,
            agnostic=agnostic_product,
        )
        if hasattr(inter, "resnetBB") and inter.irreps_sc != prod.irreps_out:
            raise ValueError(
                f"TECE BB residual irreps mismatch at layer {layer}: "
                f"skip={inter.irreps_sc} product={prod.irreps_out}"
            )
        blocks.append(ClusterExpansionBlock(eu, inter, prod))
        irreps_in = prod.irreps_out

    feature_irreps = str(irreps_in)
    dim_feature_out = scalar_0e_dim(feature_irreps)
    return ComposableCore(
        ClusterEmbed(num_elements, node_proj, spherical_harmonics, wigner, edge_embedding_mod, tensor_seed),
        blocks, FeatureHead(source="equi"),
        GraphRequires(rotations=(RotationRequest(Lmax, wigner, mmax, -1),),
                      distances=False, edge_convention="swap", fields=("Za",)),
        positional_fields=("Za", "vij_sr", "idx_i_sr", "idx_j_sr", "rbf", "atom_embedding",
                           "charge_embedding", "spin_embedding", "cutoff_values_sr"), architecture="tece",
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        Lmax=Lmax, lmax=lmax, mmax=mmax,
        num_channel=num_channel, num_elements=num_elements, target_irreps=target_irreps,
        dim_feature_out=dim_feature_out, feature_irreps=feature_irreps,
    )


def e2former(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    irreps_node_embedding: str = "64x0e+64x1e+64x2e",
    irreps_head: str = "16x0e+16x1e+16x2e",
    num_layers: int = 2,
    num_attn_heads: int = 4,
    attn_scalar_head: int = 32,
    ffn_hidden_channels: int = 128,
    max_neighbors: int = 32,
    attn_type: str = "first-order",
    tp_type: str = "QK_alpha",
    norm_layer: str = "rms_norm_sh",
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
    avg_degree: float = _AVG_DEGREE,
    use_atom_edge_embedding: bool = True,
    grid_resolution: Optional[int] = None,
    use_gate_act: bool = False,
    use_grid_mlp: bool = False,
    use_sep_s2_act: bool = True,
    ffn_activation: str = "scaled_silu",
    **kwargs,
) -> ComposableCore:
    """Apply Wigner-6j or SO(2) graph attention on bounded neighbourhoods."""
    from e3nn import o3
    from ..e2former.embedding import EdgeDegreeEmbeddingHigherOrder
    from ..e2former.interaction import TransBlock
    from ..e2former.stages import NeighborhoodEmbed, neighborhood_attention
    from ..mp.equivariant import build_so3_grid_table, get_normalization_layer

    irreps_node_embedding = o3.Irreps(irreps_node_embedding)
    irreps_head = o3.Irreps(irreps_head)
    if irreps_node_embedding[0][1].l != 0:
        raise ValueError("irreps_node_embedding must start with 0e")
    # Equal multiplicity across degrees (E2Former layout)
    scalar_dim = irreps_node_embedding[0][0]
    for mul, _ir in irreps_node_embedding:
        if mul != scalar_dim:
            raise ValueError(
                "E2Former requires equal multiplicity across degrees; "
                f"got {irreps_node_embedding}"
            )
    lmax = irreps_node_embedding[-1][1].l

    dim_feature_out = scalar_dim
    feature_irreps = f"{scalar_dim}x0e"

    sphere_proj = torch.nn.Linear(dim_embedding, scalar_dim)
    edge_deg_embed = EdgeDegreeEmbeddingHigherOrder(
        irreps_node_embedding,
        avg_aggregate_num=avg_degree,
        number_of_basis=num_rbf,
        use_atom_edge=use_atom_edge_embedding,
        max_num_elements=max_Za + 1,
    )
    SO3_grid = build_so3_grid_table(
        lmax,
        normalization="component",
        resolution=grid_resolution,
        rescale_by_mmax=True,
    )
    blocks = []
    for _ in range(num_layers):
        blocks.append(
            TransBlock(
                irreps_node_input=irreps_node_embedding,
                attn_weight_input_dim=num_rbf,
                num_attn_heads=num_attn_heads,
                attn_scalar_head=attn_scalar_head,
                irreps_head=irreps_head,
                SO3_grid=SO3_grid,
                ffn_hidden_channels=ffn_hidden_channels,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                attn_type=attn_type,
                tp_type=tp_type,
                use_gate_act=use_gate_act,
                use_grid_mlp=use_grid_mlp,
                use_sep_s2_act=use_sep_s2_act,
                ffn_activation=ffn_activation,
                atom_type_cnt=max_Za + 1,
            )
        )
    norm = get_normalization_layer(
        norm_layer, lmax=lmax, num_channels=scalar_dim
    )
    use_frames = attn_type == "so2-first-order"
    paths = getattr(blocks[0].ga.attention_order_module, "l3_sequential", None) if use_frames and blocks else None

    return ComposableCore(
        NeighborhoodEmbed(sphere_proj, edge_deg_embed, lmax, max_neighbors, use_frames, paths),
        [FnBlock(block, neighborhood_attention) for block in blocks], SphereHead(norm),
        GraphRequires(distances=False, optional_cutoff_values=False, fields=("Za", "Ra"), optional_fields=("batch_seg",)),
        positional_fields=("atom_embedding", "Za", "Ra", "rbf", "idx_i_sr", "idx_j_sr", "vij_sr", "batch_seg"), architecture="e2former",
        irreps_node_embedding=irreps_node_embedding, irreps_head=irreps_head, lmax=lmax,
        dim_embedding=dim_embedding, num_rbf=num_rbf, num_layers=num_layers,
        scalar_dim=scalar_dim, dim_feature_out=dim_feature_out, feature_irreps=feature_irreps,
        max_neighbors=max_neighbors, attn_type=attn_type, tp_type=tp_type,
        max_Za=max_Za, SO3_grid=SO3_grid,
    )


def e2former_lsr(
    dim_embedding: int,
    num_rbf: int,
    max_Za: int = 94,
    irreps_node_embedding: str = "64x0e+64x1e+64x2e",
    irreps_head: str = "16x0e+16x1e+16x2e",
    num_layers: int = 4,
    long_layers: int = 2,
    num_attn_heads: int = 4,
    attn_scalar_head: int = 32,
    ffn_hidden_channels: int = 128,
    max_neighbors: int = 32,
    long_max_neighbors: int = 64,
    attn_type: str = "first-order",
    tp_type: str = "QK_alpha",
    norm_layer: str = "rms_norm_sh",
    alpha_drop: float = 0.0,
    proj_drop: float = 0.0,
    avg_degree: float = 15.57930850982666,
    use_atom_edge_embedding: bool = True,
    grid_resolution: Optional[int] = None,
    use_gate_act: bool = False,
    use_grid_mlp: bool = False,
    use_sep_s2_act: bool = True,
    ffn_activation: str = "scaled_silu",
    cutoff_lr: float = 15.0,
    fragment_mode: str = "kmeans",
    min_nodes_per_group: int = 24,
    kmeans_random_state: int = 0,
    **kwargs,
) -> ComposableCore:
    """Fuse the short-range representation with an atom–fragment attention tower."""
    from ..e2former.interaction import ClusterTransBlock
    from ..e2former.stages import FragmentGraph, FragmentHead, fragment_attention
    from ..mp.basis.radial import GaussianSmearing
    from ..mp.equivariant import get_normalization_layer
    from ..mp.equivariant.sph.linear import SO3Linear

    short = e2former(
        dim_embedding=dim_embedding,
        num_rbf=num_rbf,
        max_Za=max_Za,
        irreps_node_embedding=irreps_node_embedding,
        irreps_head=irreps_head,
        num_layers=num_layers,
        num_attn_heads=num_attn_heads,
        attn_scalar_head=attn_scalar_head,
        ffn_hidden_channels=ffn_hidden_channels,
        max_neighbors=max_neighbors,
        attn_type=attn_type,
        tp_type=tp_type,
        norm_layer=norm_layer,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        avg_degree=avg_degree,
        use_atom_edge_embedding=use_atom_edge_embedding,
        grid_resolution=grid_resolution,
        use_gate_act=use_gate_act,
        use_grid_mlp=use_grid_mlp,
        use_sep_s2_act=use_sep_s2_act,
        ffn_activation=ffn_activation,
        **kwargs,
    )
    long_layers = int(long_layers)
    long_max_neighbors = int(long_max_neighbors)
    cutoff_lr = float(cutoff_lr)
    min_nodes_per_group = max(int(min_nodes_per_group), 1)
    kmeans_random_state = int(kmeans_random_state)

    rbf_long = GaussianSmearing(0.0, cutoff_lr, num_rbf, preserve_shape=True)
    cluster_blocks = []
    for _ in range(long_layers):
        cluster_blocks.append(
            ClusterTransBlock(
                irreps_node_input=short.irreps_node_embedding,
                attn_weight_input_dim=num_rbf,
                num_attn_heads=num_attn_heads,
                attn_scalar_head=attn_scalar_head,
                irreps_head=short.irreps_head,
                SO3_grid=short.SO3_grid,
                ffn_hidden_channels=ffn_hidden_channels,
                alpha_drop=alpha_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                attn_type=attn_type,
                tp_type=tp_type,
                use_gate_act=use_gate_act,
                use_grid_mlp=use_grid_mlp,
                use_sep_s2_act=use_sep_s2_act,
                ffn_activation=ffn_activation,
                atom_type_cnt=max_Za + 1,
            )
        )
    norm_fuse_short = get_normalization_layer(
        norm_layer, lmax=short.lmax, num_channels=short.scalar_dim
    )
    norm_fuse_long = get_normalization_layer(
        norm_layer, lmax=short.lmax, num_channels=short.scalar_dim
    )
    final_linear = SO3Linear(
        2 * short.scalar_dim, short.scalar_dim, lmax=short.lmax
    )
    use_frames = attn_type == "so2-first-order" and bool(cluster_blocks)
    paths = getattr(cluster_blocks[0].ga.attention_order_module, "l3_sequential", None) if use_frames else None
    prepare = FragmentGraph(rbf_long, cutoff_lr, long_max_neighbors, fragment_mode, min_nodes_per_group,
                            kmeans_random_state, use_frames, short.lmax, paths)
    blocks = [*short.blocks, prepare, *(FnBlock(block, fragment_attention) for block in cluster_blocks)]
    return ComposableCore(
        short.embed, blocks, FragmentHead(short.head.norm, norm_fuse_short, norm_fuse_long, final_linear, bool(cluster_blocks)),
        GraphRequires(distances=False, optional_cutoff_values=False, fields=("Za", "Ra"),
                      optional_fields=("batch_seg", "cluster_ids", "cluster_centers")),
        positional_fields=(*short.positional_fields, "cluster_ids", "cluster_centers"),
        architecture="e2former_lsr", _force_topology_block=len(short.blocks),
        dim_feature_out=short.dim_feature_out, feature_irreps=short.feature_irreps,
        dim_embedding=dim_embedding, scalar_dim=short.scalar_dim, num_layers=num_layers, num_rbf=num_rbf,
        min_nodes_per_group=min_nodes_per_group, attn_type=attn_type, long_layers=long_layers, lmax=short.lmax,
    )
