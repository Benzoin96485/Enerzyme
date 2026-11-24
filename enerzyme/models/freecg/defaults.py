_SILU_READOUT = {
    "dim_feature": 1,
    "initial_alpha": 1.0,
    "initial_beta": 1.0,
    "learnable": False,
}

# Table 7. Eight groups on 256 channels is the reported setting. The
# repository yaml uses 32 groups of 8; the ablation prefers 8 groups.
DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 256,
    "num_rbf": 32,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_MD17_BUILD_PARAMS = dict(DEFAULT_BUILD_PARAMS)

PUBLISHED_MD17_CORE_PARAMS = {
    "num_layers": 9,
    "num_heads": 8,
    "lmax": 2,
    "vertex_type": "none",
    "vecnorm_type": "none",
    "trainable_vecnorm": False,
    "neighbor_embedding": True,
    "include_self": True,
    "activation": "silu",
    "attn_activation": "silu",
    "num_groups": 8,
    "shuffle_scale": 1.5,
    "attention_enhancer": "linear",
}

# Table 7 latent width 512. QM9's ViSNet run uses max_min vector norm; the
# FreeCG table does not restate it, so the QM9 preset keeps that ViSNet choice.
PUBLISHED_QM9_BUILD_PARAMS = {
    "dim_embedding": 512,
    "num_rbf": 64,
    "max_Za": 100,
    "cutoff_sr": 5.0,
    "cutoff_fn": "cosine",
}

PUBLISHED_QM9_CORE_PARAMS = {
    **PUBLISHED_MD17_CORE_PARAMS,
    "vecnorm_type": "max_min",
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {"name": "ExpNormalSmearing", "params": {"trainable": False}},
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": dict(PUBLISHED_MD17_CORE_PARAMS)},
    {
        "name": "SimpleReadout",
        "params": {
            "output_fields": ["Ea"],
            "head_type": "two_layer",
            "pyramidal": True,
            "activation_fn": "swish",
            "activation_params": dict(_SILU_READOUT),
            "keep_feature": False,
        },
    },
    {"name": "EnergyReduce"},
    {"name": "Force"},
]
