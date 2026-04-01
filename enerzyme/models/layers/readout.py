from abc import ABC
from typing import Set, Literal, Dict, Any, Optional, List
from . import BaseFFLayer
from ..blocks.mlp import DenseLayer, ResidualMLP, ResidualLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, POSITIVE_ACTIVATION_KEY_TYPE, get_positive_activation_fn
import torch
from torch.nn import ModuleList, Module, Sequential
from torch import Tensor


class BaseReadout(BaseFFLayer):
    def __init__(self, 
        num_blocks: int,
        output_fields: Set[str], 
        built_layers: List[Module],
        head_type: Literal["dense", "residual_layer", "residual_mlp"],
        dim_embedding: Optional[int]=None,
        shallow_ensemble_size: int=1, 
        keep_feature: bool=False, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        **head_params
    ) -> None:
        super().__init__(
            input_fields=["atom_feature"], 
            output_fields=output_fields | {"atom_feature"} if keep_feature else output_fields
        )
        self.num_blocks = num_blocks
        self.shallow_ensemble_size = shallow_ensemble_size
        self.ordered_output_fields = sorted(list(output_fields))
        self.head_type = head_type
        if len(built_layers) > 0 and hasattr(built_layers[-1], "dim_feature_out"):
            self.dim_feature_in = built_layers[-1].dim_feature_out
        elif dim_embedding is not None:
            self.dim_feature_in = dim_embedding
        else:
            raise ValueError("dim_embedding or dim_feature_out from the last layer must be provided")
        self.dim_feature_out = len(self.ordered_output_fields)
        self.activation_fn = activation_fn
        self.activation_params = activation_params
        self.head_params = head_params

    def _get_head(self):
        if self.head_type == "dense":
            return DenseLayer(
                dim_feature_in=self.dim_feature_in, 
                dim_feature_out=self.dim_feature_out,
                shallow_ensemble_size=self.shallow_ensemble_size,
                **self.head_params
            )
        elif self.head_type == "residual_layer":
            return Sequential(
                ResidualLayer(
                    dim_feature_in=self.dim_feature_in, 
                    dim_feature_out=self.dim_feature_in,
                    activation_fn=self.activation_fn,
                    activation_params=self.activation_params,
                    **self.head_params
                ),
                DenseLayer(
                    dim_feature_in=self.dim_feature_in, 
                    dim_feature_out=self.dim_feature_out,
                    shallow_ensemble_size=self.shallow_ensemble_size,
                    **self.head_params
                )
            )
        elif self.head_type == "residual_mlp":
            return ResidualMLP(
                dim_feature_in=self.dim_feature_in, 
                dim_feature_out=self.dim_feature_out,
                shallow_ensemble_size=self.shallow_ensemble_size,
                activation_fn=self.activation_fn,
                activation_params=self.activation_params,
                **self.head_params
            )


class SimpleReadout(BaseReadout):
    def __init__(self, 
        output_fields: Set[str], 
        built_layers: List[Module],
        head_type: Literal["dense", "residual_layer", "residual_mlp"],
        dim_embedding: Optional[int]=None,
        shallow_ensemble_size: int=1, 
        keep_feature: bool=False, 
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        **head_params
    ) -> None:
        super().__init__(
            num_blocks=1,
            output_fields=output_fields,
            built_layers=built_layers,
            head_type=head_type,
            dim_embedding=dim_embedding,
            shallow_ensemble_size=shallow_ensemble_size,
            keep_feature=keep_feature,
            activation_fn=activation_fn,
            activation_params=activation_params,
            **head_params
        )
        self.head = self._get_head()

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        if atom_feature.ndim == 2:
            output = self.head(atom_feature)
        elif atom_feature.ndim == 3:
            output = self.head(atom_feature[:, :, -1])
        return {
            self.ordered_output_fields[i]: output[:, i] for i in range(self.dim_feature_out)
        }


class NSEReadout(BaseReadout):
    """Readout head specialized for NSE intermediate variables.

    By default, this head predicts four per-atom scalar fields required by
    Neural Spin-Charge Equilibration (NSE):
      - Qa_alpha_tilde
      - Qa_beta_tilde
      - fa_alpha
      - fa_beta

    It remains fully compatible with UMAWrapperQS by consuming `atom_feature`
    in the same way as `SimpleReadout`, and it can optionally be configured
    with custom `output_fields` if needed.
    """

    def __init__(
        self,
        output_fields: Optional[Set[str]] = None,
        built_layers: List[Module] = [],
        head_type: Literal["dense", "residual_layer", "residual_mlp"] = "residual_mlp",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        positive_activation_fn: POSITIVE_ACTIVATION_KEY_TYPE = "softplus",
        **head_params,
    ) -> None:
        if output_fields is None:
            output_fields = {
                "Qa_alpha_tilde",
                "Qa_beta_tilde",
                "fa_alpha",
                "fa_beta",
            }
        super().__init__(
            num_blocks=1,
            output_fields=output_fields,
            built_layers=built_layers,
            head_type=head_type,
            dim_embedding=dim_embedding,
            shallow_ensemble_size=shallow_ensemble_size,
            keep_feature=keep_feature,
            activation_fn=activation_fn,
            activation_params=activation_params,
            **head_params,
        )
        self.head = self._get_head()
        self.positive_activation_fn = get_positive_activation_fn(positive_activation_fn)

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        if atom_feature.ndim == 2:
            output = self.head(atom_feature)
        elif atom_feature.ndim == 3:
            output = self.head(atom_feature[:, :, -1])
        results = dict()
        for i, output_field in enumerate(self.ordered_output_fields):
            if output_field.startswith("fa"):
                results[output_field] = self.positive_activation_fn(output[:, i])
            else:
                results[output_field] = output[:, i]
        return results


class HierachicalReadout(BaseReadout):
    def __init__(self, use_nhloss: bool=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.heads = ModuleList([self._get_head() for _ in range(self.num_blocks)])
        self.use_nhloss = use_nhloss

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        raw_output = 0.
        if self.use_nhloss:
            nhloss = 0.
            lastoutput2 = 0.
        for i in range(self.num_blocks):
            raw_output += self.heads[i](atom_feature[:, :, i])
            if self.use_nhloss:
                output2 = raw_output ** 2
                if i > 0:
                    nhloss += torch.mean(output2 / (output2 + lastoutput2 + 1e-7))
                lastoutput2 = output2
        output = {
            self.ordered_output_fields[i]: raw_output[:, i] for i in range(self.dim_feature_out)
        }
        if self.use_nhloss:
            output["nh_loss"] = nhloss
        return output


class HierachicalNSEReadout(BaseReadout):
    """Hierachical readout head specialized for NSE intermediate variables.

    This readout supports 3D `atom_feature` of shape (N, dim, num_blocks),
    accumulating predictions across blocks (PhysNet-style), and enforces
    non-negativity on `fa_*` outputs via a configurable positive activation.
    """

    def __init__(
        self,
        num_blocks: int,
        output_fields: Optional[Set[str]] = None,
        built_layers: List[Module] = [],
        head_type: Literal["dense", "residual_layer", "residual_mlp"] = "residual_mlp",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        positive_activation_fn: POSITIVE_ACTIVATION_KEY_TYPE = "softplus",
        **head_params,
    ) -> None:
        if output_fields is None:
            output_fields = {
                "Qa_alpha_tilde",
                "Qa_beta_tilde",
                "fa_alpha",
                "fa_beta",
            }
        super().__init__(
            num_blocks=num_blocks,
            output_fields=output_fields,
            built_layers=built_layers,
            head_type=head_type,
            dim_embedding=dim_embedding,
            shallow_ensemble_size=shallow_ensemble_size,
            keep_feature=keep_feature,
            activation_fn=activation_fn,
            activation_params=activation_params,
            **head_params,
        )
        self.heads = ModuleList([self._get_head() for _ in range(self.num_blocks)])
        self.positive_activation_fn = get_positive_activation_fn(positive_activation_fn)

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        if atom_feature.ndim != 3:
            raise ValueError(
                f"HierachicalNSEReadout expects 3D atom_feature (N, dim, num_blocks), got ndim={atom_feature.ndim}"
            )
        if atom_feature.shape[-1] != self.num_blocks:
            raise ValueError(
                f"atom_feature last dim (num_blocks) mismatch: got {atom_feature.shape[-1]}, expected {self.num_blocks}"
            )

        raw_output = 0.0
        for i in range(self.num_blocks):
            raw_output = raw_output + self.heads[i](atom_feature[:, :, i])

        results: Dict[str, Tensor] = {}
        for i, output_field in enumerate(self.ordered_output_fields):
            if output_field.startswith("fa"):
                results[output_field] = self.positive_activation_fn(raw_output[:, i])
            else:
                results[output_field] = raw_output[:, i]
        return results

