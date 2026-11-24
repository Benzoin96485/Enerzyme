"""Default configuration for the optional XequiNet xPaiNN preset."""

DEFAULT_BUILD_PARAMS = {
    'cutoff_sr': 5.0,
}
DEFAULT_LAYER_PARAMS = [
    {'name': 'RangeSeparation'},
    {"name": "Core", "params": {
        'embed_basis': "gfn2-xtb",                  # embedding basis type
        'aux_basis': "aux56",                       # auxiliary basis type
        'node_dim': 128,                            # node irreps for the input
        'edge_irreps': "128x0e + 64x1o + 32x2e",    # edge irreps for the input
        'hidden_dim': 64,                           # hidden dimension for the output
        'hidden_irreps': "64x0e + 32x1o + 16x2e",   # hidden irreps for the output
        'rbf_kernel': "bessel",                     # radial basis function type
        'num_basis': 20,                            # number of the radial basis functions
        'cutoff_fn': "cosine",                      # cutoff function type
        'max_edges': 100,                           # maximum number of the edges
        'action_blocks': 3,                         # number of the action blocks
        'activation': "silu",                       # activation function type
        'norm_type': "nonorm",                      # normalization layer type
        'output_mode': "grad",                    # task type (`scalar` is for energy like, `grad` is for force like, etc.)
        'output_dim': 1,                            # output dimension of multi-task (only for `scalar` mode)
        'node_average': False,                      # whether to add the node average to the output (only for `scalar` mode)
        'default_dtype': "float32"                 # default data type      
    }}
]
