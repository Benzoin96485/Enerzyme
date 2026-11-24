"""CHGNet interatomic potential.

Deng et al., Nature Machine Intelligence 5, 1031–1041 (2023). The message
passing follows the update rules in
`CederGroupHub/chgnet <https://github.com/CederGroupHub/chgnet>`_. That
package is not imported, and its checkpoints are not loaded.

The Core reads two ``BesselRBF`` layers (``flavor: gemnet``) and
``RandomAtomEmbedding``. The atom-graph basis uses ``cutoff_sr``. The
bond-graph basis uses ``bond_cutoff``. The magnetic-moment head is
``AbsoluteLinearReadout`` on ``atom_feature_charge``. The energy head is
``SimpleReadout``.
"""



_ENERGY = {
    "output_fields": ["Ea"],
    "head_type": "mlp",
    "num_hidden_layers": 2,
    "activation_fn": "swish",
    "use_bias_out": True,
}

DEFAULT_BUILD_PARAMS = {
    "dim_embedding": 64,
    "num_rbf": 9,
    "num_angular": 9,
    "max_Za": 94,
    "cutoff_sr": 5.0,
    "bond_cutoff": 3.0,
    "envelope_exponent": 5,
    # Energies are eV and distances are Ångström. One Hartree is 27.211 eV.
    # One Bohr is 0.529 Å, so a later pair prior sees Ångström coordinates.
    "Hartree_in_E": 27.211386245988,
    "Bohr_in_R": 0.5291772105638411,
}

DEFAULT_LAYER_PARAMS = [
    {"name": "RangeSeparation"},
    {
        "name": "BesselRBF",
        "params": {"flavor": "gemnet", "trainable": True},
    },
    {
        "name": "BesselRBF",
        "params": {
            "flavor": "gemnet",
            "trainable": True,
            "cutoff_sr": 3.0,
        },
    },
    {"name": "RandomAtomEmbedding"},
    {"name": "Core", "params": {"num_blocks": 4}},
    {"name": "AbsoluteLinearReadout", "params": {"output_fields": ["Ma"]}},
    {"name": "SimpleReadout", "params": dict(_ENERGY)},
    {"name": "EnergyReduce"},
    {"name": "Force"},
]

__all__ = [
    "DEFAULT_BUILD_PARAMS",
    "DEFAULT_LAYER_PARAMS",
]
