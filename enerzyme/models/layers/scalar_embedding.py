from typing import Optional, Dict
from torch import Tensor
from . import BaseFFLayer
from ..activation import ACTIVATION_PARAM_TYPE, ACTIVATION_KEY_TYPE
from ..blocks.mlp import DenseLayer, ResidualMLP, INITIAL_WEIGHT_TYPE, INITIAL_BIAS_TYPE


class ScalarEmbedding(BaseFFLayer):
    def __init__(self, 
        embed_field: str
    ) -> None:
        self.input_field = embed_field
        self.output_field = f"{embed_field}_embedding"
        super().__init__(input_fields={embed_field}, output_fields={self.output_field})
        self.embedding = None
        
    def get_output(self, **relevant_input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {self.output_field: self.embedding(relevant_input[self.input_field].unsqueeze(-1))}


class ScalarDenseEmbedding(ScalarEmbedding):
    def __init__(self, 
        dim_embedding: int, embed_field: str, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None, activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_bias: INITIAL_BIAS_TYPE="zero",
        use_bias: bool=True,
    ) -> None:
        super().__init__(embed_field)
        self.embedding = DenseLayer(
            dim_feature_in=1, 
            dim_feature_out=dim_embedding,
            activation_fn=activation_fn,
            activation_params=activation_params,
            initial_weight=initial_weight,
            initial_bias=initial_bias,
            use_bias=use_bias,
            shallow_ensemble_size=1
        )


class ScalarResidualMLPEmbedding(ScalarEmbedding):
    def __init__(self, 
        dim_embedding: int, embed_field: str,
        num_residual: int, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]="swish", 
        activation_params: ACTIVATION_PARAM_TYPE=dict(), 
        initial_weight1: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_weight2: INITIAL_WEIGHT_TYPE="orthogonal", 
        initial_weight_out: INITIAL_WEIGHT_TYPE="orthogonal",
        initial_bias_residual: INITIAL_BIAS_TYPE="zero",
        initial_bias_out: INITIAL_BIAS_TYPE="zero",
        dropout_rate: float=0,
        use_bias_residual: bool=True,
        use_bias_out: bool=True,
        use_residual: bool=True
    ) -> None:
        super().__init__(embed_field)
        self.embedding = ResidualMLP(
            dim_feature_in=1,
            dim_feature_out=dim_embedding,
            num_residual=num_residual,
            activation_fn=activation_fn,
            activation_params=activation_params,
            initial_weight1=initial_weight1,
            initial_weight2=initial_weight2,
            initial_weight_out=initial_weight_out,
            initial_bias_residual=initial_bias_residual,
            initial_bias_out=initial_bias_out,
            dropout_rate=dropout_rate,
            use_bias_residual=use_bias_residual,
            use_bias_out=use_bias_out,
            shallow_ensemble_size=1,
            use_residual=use_residual
        )