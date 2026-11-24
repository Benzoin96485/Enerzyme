"""e3nn irreps message-passing presets.

``mace`` loads ``main``-branch checkpoints through
:mod:`enerzyme.models.compat.main_branch`; its parameter names and numerics
follow ``main``.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from ..mp.core import ComposableCore, FeatureHead, FieldEmbed, ReadoutHead
from ..mp.graph import GraphRequires

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
