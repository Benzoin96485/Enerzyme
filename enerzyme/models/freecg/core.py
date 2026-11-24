"""FreeCG Core (Shao et al., arXiv:2407.02263).

The Core is ViSNet's neighbor embedding, edge state, and vector-scalar stack,
with the group Clebsch–Gordan product on the aggregated abstract edges. It
reads a species embedding and an exponential-normal basis and emits:

* ``atom_feature`` — invariant features ``[N, F]``
* ``atom_vector_feature`` — Cartesian ``l = 1`` features ``[N, 3, F]``
* ``atom_l2_feature`` — polynomial ``l = 2`` features ``[N, 5, F]``

``ExpNormalSmearing``, the first
:class:`~enerzyme.models.layers.atom_embedding.RandomAtomEmbedding`, the
cosine window, and the scalar energy MLP stay outside. The reference
equivariant scalar head is not part of the Core; the stack uses the same
:class:`~enerzyme.models.layers.readout.SimpleReadout` as ViSNet.

Adapted from ``freecg.models.freecg_block`` (MIT):
https://github.com/ShihaoShao-GH/FreeCG
Checkpoints from that package do not load. ``lmax`` is 2: the sparse paths
are exactly ``1o`` and ``2e``.
"""

from __future__ import annotations

from torch.nn import ModuleList

from ..layers.polynomial_harmonics import angular_dim
from ..visnet.core import ViSNetCore
from ..visnet.interaction import normalize_activation, normalize_vecnorm_type, normalize_vertex_type
from .interaction import FreeCGInteraction, normalize_attention_enhancer


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


class FreeCGCore(ViSNetCore):
    """ViSNet stack whose vector residual is a grouped sparse CG product."""

    def __str__(self) -> str:
        return """
################################################################################
# FreeCG (Shao et al., arXiv:2407.02263)                                      #
# atom_feature [N, F], atom_vector_feature [N, 3, F], atom_l2_feature [N, 5, F] #
################################################################################
"""

    def __init__(
        self,
        dim_embedding: int = 256,
        num_rbf: int = 32,
        max_Za: int = 100,
        cutoff_sr: float = 5.0,
        num_layers: int = 9,
        num_heads: int = 8,
        lmax: int = 2,
        vertex_type: object = "none",
        vecnorm_type: object = "none",
        trainable_vecnorm: bool = False,
        neighbor_embedding: bool = True,
        include_self: bool = True,
        activation: object = "silu",
        attn_activation: object = "silu",
        num_groups: int = 8,
        shuffle_scale: float = 1.5,
        attention_enhancer: object = "linear",
    ) -> None:
        if int(lmax) != 2:
            raise ValueError(
                "FreeCG's sparse paths are 1o and 2e, so lmax must be 2, "
                f"got {lmax}"
            )
        groups = int(num_groups)
        if groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        if int(dim_embedding) % groups != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by "
                f"num_groups ({groups})"
            )
        if float(shuffle_scale) < 0:
            raise ValueError(f"shuffle_scale must be >= 0, got {shuffle_scale}")
        enhancer = normalize_attention_enhancer(attention_enhancer)
        # Reject unknown names before the parent builds a stack we would drop.
        normalize_vertex_type(vertex_type)
        normalize_vecnorm_type(vecnorm_type)
        normalize_activation(activation)
        normalize_activation(attn_activation)
        super().__init__(
            dim_embedding=dim_embedding,
            num_rbf=num_rbf,
            max_Za=max_Za,
            cutoff_sr=cutoff_sr,
            num_layers=num_layers,
            num_heads=num_heads,
            lmax=2,
            vertex_type=vertex_type,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            neighbor_embedding=neighbor_embedding,
            include_self=include_self,
            activation=activation,
            attn_activation=attn_activation,
        )
        self.num_groups = groups
        self.shuffle_scale = float(shuffle_scale)
        self.attention_enhancer = enhancer
        self.channels_per_group = self.dim_embedding // groups
        if self.angular_dim != angular_dim(2):
            raise RuntimeError(
                f"FreeCG expects 8 angular channels, got {self.angular_dim}"
            )
        self.vis_layers = ModuleList(
            [
                FreeCGInteraction(
                    self.dim_embedding,
                    self.num_heads,
                    self.vertex_type,
                    self.vecnorm_type,
                    self.trainable_vecnorm,
                    last_layer=(index == self.num_layers - 1),
                    num_groups=self.num_groups,
                    shuffle_scale=self.shuffle_scale,
                    attention_enhancer=self.attention_enhancer,
                )
                for index in range(self.num_layers)
            ]
        )
