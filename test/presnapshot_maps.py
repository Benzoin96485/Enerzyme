"""Map pre-refactor snapshot state_dicts onto rebuilt Cores.

``KEY_MAPS[arch]`` is a list of ``(regex, replacement)`` pairs applied in order
to every old key. The legacy architectures (SchNet, PhysNet, SpookyNet, MACE)
go through the ``main``-branch checkpoint converter. Any other architecture
without an entry first tries an identical key set, then pairs parameters positionally (both sides in ``named_parameters``
order, shapes must agree). Buffers are copied by name when present on both
sides; a buffer only the new model has must be a deterministic constant.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch

KEY_MAPS: Dict[str, List[Tuple[str, str]]] = {
    "equflash": [(r"^interactions\.(\d+)\.", r"blocks.\1.module.")],
    "equflashv2": [
        (r"^interactions\.(\d+)\.", r"blocks.\1.interaction."),
        (r"^feedforwards\.(\d+)\.", r"blocks.\1.feedforward."),
    ],
    "dpa4": [
        (r"^(radial_basis|envelope|wigner_calc|sphere_proj|radial_feat_mlp|gie|env_embed)\.", r"embed.\1."),
        (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
        (r"^norm\.", "head.norm."),
    ],
    "tece": [
        (r"^node_proj\.", "embed.node_proj."),
        (r"^edge_embedding_mod\.", "embed.edge_embed."),
        (r"^tensor_seed\.", "embed.seed."),
        (r"^wigner\.", "embed.wigner."),
        (r"^edge_updates\.(\d+)\.", r"blocks.\1.edge_update."),
        (r"^interactions\.(\d+)\.", r"blocks.\1.interaction."),
        (r"^products\.(\d+)\.", r"blocks.\1.product."),
    ],
    "escn": [
        (r"^sphere_proj\.", "embed.sphere_proj."),
        (r"^layer_blocks\.(\d+)\.", r"blocks.\1.module."),
    ],
    "esen": [
        (r"^sphere_proj\.", "embed.projection."),
        (r"^source_embedding\.", "embed.source."),
        (r"^target_embedding\.", "embed.target."),
        (r"^edge_degree\.", "embed.degree."),
        (r"^norm\.", "head.norm."),
        (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
    ],
    "equiformer_v2": [
        (r"^sphere_proj\.", "embed.sphere_proj."),
        (r"^edge_degree_embedding\.", "embed.degree."),
        (r"^norm\.", "head.norm."),
        (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
    ],
    "equiformer_v3": [
        (r"^sphere_proj\.", "embed.projection."),
        (r"^edge_degree_embedding\.", "embed.degree."),
        (r"^so3_rotation\.", "embed.rotation."),
        (r"^norm\.", "head.norm."),
        (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
    ],
    "botnet": [(r"^interactions\.(\d+)\.", r"blocks.\1.module.")],
    "allegro": [
        (r"^(scalar_mlp|tensor_weight|pair_projection)\.", r"embed.\1."),
        (r"^products\.(\d+)\.", r"blocks.\1.product."),
        (r"^latents\.(\d+)\.", r"blocks.\1.latent."),
    ],
    "equiformer": [
        (r"^edge_deg_embed\.", "embed.degree."),
        (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
        (r"^head\.", "head.readout."),
        (r"^norm\.", "head.norm."),
    ],
    "nequip": [(r"^convs\.(\d+)\.", r"blocks.\1.module.")],
    "nequix": [(r"^layers\.(\d+)\.", r"blocks.\1.module.")],
    "eqnorm": [
        (r"^species_linear\.", "embed.projection."),
        (r"^layers\.(\d+)\.", r"blocks.\1.module."),
    ],
}

KEY_MAPS["painn"] = [
    (r"^filter_net\.", r"embed.filter_net."),
    (r"^interactions\.(\d+)\.", r"blocks.\1.message."),
    (r"^mixings\.(\d+)\.", r"blocks.\1.update."),
]

KEY_MAPS["newtonnet"] = [
    (r"^blocks\.(\d+)\.", r"blocks.\1.module."),
]

KEY_MAPS["et"] = [
    (r"^neighborhood\.", r"embed.neighborhood."),
    (r"^edge_embedding\.", r"embed.edge_embedding."),
    (r"^attention_layers\.(\d+)\.", r"blocks.\1.module."),
    (r"^(vec_out_norm|out_norm)\.", r"head.\1."),
]

KEY_MAPS["visnet"] = [
    (r"^neighborhood\.", r"embed.neighborhood."),
    (r"^edge_embedding\.", r"embed.edge_embedding."),
    (r"^vis_layers\.(\d+)\.", r"blocks.\1.module."),
    (r"^(vec_out_norm|out_norm)\.", r"head.\1."),
]

KEY_MAPS["quinnet"] = [
    (r"^neighborhood\.", r"embed.neighborhood."),
    (r"^edge_embedding\.", r"embed.edge_embedding."),
    (r"^quin_layers\.(\d+)\.", r"blocks.\1.module."),
    (r"^(vec_out_norm|out_norm)\.", r"head.\1."),
]

KEY_MAPS["freecg"] = [
    (r"^neighborhood\.", r"embed.neighborhood."),
    (r"^edge_embedding\.", r"embed.edge_embedding."),
    (r"^vis_layers\.(\d+)\.", r"blocks.\1.module."),
    (r"^(vec_out_norm|out_norm)\.", r"head.\1."),
]

# The frozen LSRM fixture has six short-range interactions followed by two fragment layers.
KEY_MAPS["visnet_lsrm"] = [
    (r"^(long_rbf(?:_proj)?)\.", r"blocks.6.\1."),
    (r"^long_blocks\.0\.", r"blocks.7.module."),
    (r"^long_blocks\.1\.", r"blocks.8.module."),
    (r"^neighborhood\.", r"embed.neighborhood."),
    (r"^edge_embedding\.", r"embed.edge_embedding."),
    (r"^vis_layers\.(\d+)\.", r"blocks.\1.module."),
    (r"^(vec_out_norm|out_norm)\.", r"head.short.\1."),
    (r"^(out_norm_long|vec_out_norm_long)\.", r"head.\1."),
]

KEY_MAPS["leftnet"] = [
    (r"^(radial|lse|embed_norm)\.", r"embed.\1."),
    (r"^messages\.(\d+)\.", r"blocks.\1.message."),
    (r"^transitions\.(\d+)\.", r"blocks.\1.transition."),
]

KEY_MAPS["alphanet"] = [
    (r"^(radial|lse|embed_norm)\.", r"embed.\1."),
    (r"^messages\.(\d+)\.", r"blocks.\1.message."),
    (r"^transitions\.(\d+)\.", r"blocks.\1.transition."),
    (r"^(kernel1|kernels_real|kernels_imag)$", r"embed.\1"),
    (r"^quantum_mix\.", r"head.quantum_mix."),
]

KEY_MAPS["geodite"] = [
    (r"^(node_init|edge_init)\.", r"embed.\1."),
    (r"^interactions\.(\d+)\.", r"blocks.\1.message."),
    (r"^couplings\.(\d+)\.", r"blocks.\1.coupling."),
]


KEY_MAPS["dimenet"] = KEY_MAPS["dimenet++"] = [
    (r"^sbf\.", "embed.basis."),
    (r"^emb_block\.", "embed.embedding."),
    (r"^interactions\.(\d+)\.", r"blocks.\1.interaction."),
    (r"^atom_readouts\.", "head.readouts."),
]


KEY_MAPS["m3gnet"] = [(r"^edge_embed\.", "embed.edge_embed.")]
KEY_MAPS["mattersim"] = KEY_MAPS["m3gnet"] + [
    (r"^centrality\.", "embed.centrality."),
    (r"^attention\.0\.", "blocks.3."),
    (r"^attention\.1\.", "blocks.4."),
]


for _name in ("so3krates", "so3lr", "efa", "so3lr_efa"):
    KEY_MAPS[_name] = [
        (r"^layers\.", "blocks."),
        (r"^efa_blocks\.(\d+)\.", r"blocks.\1.nonlocal_update."),
    ]


KEY_MAPS["allscaip"] = [
    (r"^input_block\.", "embed.input_block."),
    (r"^graph_blocks\.", "blocks."),
]


def _rename(state: Dict[str, torch.Tensor], rules: List[Tuple[str, str]]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        for pattern, repl in rules:
            key = re.sub(pattern, repl, key)
        if key is not None and key != "":
            out[key] = value
    return out


def remap_snapshot(arch: str, old_state: Dict[str, torch.Tensor], old_param_names: List[str],
                   model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    new_state = model.state_dict()
    from enerzyme.models.compat.main_branch import KEY_RULES, convert_state_dict

    legacy = getattr(model, "architecture", None)
    if legacy in KEY_RULES and set(old_state) != set(new_state):
        return dict(convert_state_dict(legacy, old_state)[0])
    if arch in KEY_MAPS:
        renamed = _rename(old_state, KEY_MAPS[arch])
        merged = dict(new_state)
        for key, value in renamed.items():
            if key in merged:
                merged[key] = value
        missing = [k for k, _ in model.named_parameters() if k not in renamed]
        if missing:
            raise KeyError(f"{arch}: parameters without a snapshot source: {missing[:8]}")
        return merged
    if set(old_state) == set(new_state):
        return dict(old_state)
    new_params = [k for k, _ in model.named_parameters()]
    old_params = [k for k in old_param_names if k in old_state]
    if len(new_params) != len(old_params):
        raise KeyError(
            f"{arch}: {len(old_params)} snapshot parameters vs {len(new_params)} model parameters; "
            "add a KEY_MAPS entry"
        )
    merged = dict(new_state)
    for old_key, new_key in zip(old_params, new_params):
        old_value = old_state[old_key]
        if tuple(old_value.shape) != tuple(new_state[new_key].shape):
            raise ValueError(
                f"{arch}: positional match {old_key} {tuple(old_value.shape)} -> "
                f"{new_key} {tuple(new_state[new_key].shape)}; add a KEY_MAPS entry"
            )
        merged[new_key] = old_value
    param_set = set(new_params)
    for key in new_state:
        if key not in param_set and key in old_state and old_state[key].shape == new_state[key].shape:
            merged[key] = old_state[key]
    return merged
