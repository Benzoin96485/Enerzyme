"""Common operations for Cartesian-vector message-passing presets."""

from ..block import Block


def vector_metadata(width):
    return dict(dim_embedding=width, dim_feature_out=width, feature_irreps=f"{width}x0e",
                vector_feature_layout="cartesian_n3f")


class ScalarNormBlock(Block):
    """Normalize the scalar stream while passing the equivariant features through."""

    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, state, ctx):
        return state._replace(scalar=self.norm(state.scalar))
