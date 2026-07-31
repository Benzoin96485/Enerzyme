from typing import Set, Literal, Dict, Optional, List, Callable
from . import BaseFFLayer
from ..blocks.mlp import DenseLayer, ResidualMLP, ResidualLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, POSITIVE_ACTIVATION_KEY_TYPE, get_positive_activation_fn
from ..e3nn_nn import extract_scalar_0e
import math
import torch
from torch.nn import Embedding, Identity, Linear, ModuleList, Module, Sequential
from torch import Tensor


HEAD_TYPE = Literal[
    "dense",
    "residual_layer",
    "residual_mlp",
    "two_layer",
    "equiformer_linear_rs",
]


def _resolve_feature_irreps(built_layers: List[Module]) -> Optional[str]:
    """Find ``feature_irreps`` on the nearest prior Core (or any layer)."""
    for layer in reversed(built_layers):
        irreps = getattr(layer, "feature_irreps", None)
        if irreps is not None:
            return str(irreps)
    return None


def reshape_ensemble_head_output(
    head_output: Tensor,
    dim_feature_out: int,
    shallow_ensemble_size: int,
) -> Tensor:
    """Normalize head output to ``(N, fields)`` or ``(N, fields, ensemble)``.

    Accepts already-shaped tensors or a flat ``(N, fields * ensemble)`` layout
    from widened last-linear heads (LinearRS, GraphAttention, sphere FFN).
    """
    if shallow_ensemble_size <= 1:
        return head_output
    if head_output.ndim == 3:
        return head_output
    if head_output.ndim == 2 and head_output.shape[-1] == dim_feature_out * shallow_ensemble_size:
        return head_output.view(-1, dim_feature_out, shallow_ensemble_size)
    return head_output


def split_readout_field_outputs(
    head_output: Tensor,
    ordered_output_fields: List[str],
    shallow_ensemble_size: int = 1,
) -> Dict[str, Tensor]:
    """Map head tensor columns onto named fields.

    Head layout matches :class:`~enerzyme.models.blocks.mlp.DenseLayer`:
    ``(N, fields)`` when ``shallow_ensemble_size == 1``, or
    ``(N, fields, ensemble)`` when ``shallow_ensemble_size > 1``.

    Selecting field ``i`` with ``select(1, i)`` therefore yields per-atom
    scalars ``(N,)`` or shallow-ensemble predictions ``(N, ensemble)``.
    That trailing ensemble axis is the Enerzyme contract used by PhysNet /
    SchNet / MACE cores and by ChargeConservation, AtomicAffine, Force,
    WeightedLoss, and ShallowEnsembleReduce — it must not be squeezed away.
    """
    dim_feature_out = len(ordered_output_fields)
    head_output = reshape_ensemble_head_output(
        head_output, dim_feature_out, shallow_ensemble_size
    )
    if head_output.ndim == 2:
        expected = (head_output.shape[0], dim_feature_out)
    elif head_output.ndim == 3:
        expected = (
            head_output.shape[0],
            dim_feature_out,
            shallow_ensemble_size,
        )
        if shallow_ensemble_size <= 1:
            raise ValueError(
                "Readout head returned a 3D tensor but shallow_ensemble_size "
                f"is {shallow_ensemble_size}; expected 2D (N, fields)."
            )
    else:
        raise ValueError(
            f"Readout head output must be 2D or 3D, got shape {tuple(head_output.shape)}"
        )
    if tuple(head_output.shape) != expected:
        raise ValueError(
            f"Readout head output shape {tuple(head_output.shape)} != expected {expected}"
        )
    return {
        ordered_output_fields[i]: head_output.select(1, i)
        for i in range(dim_feature_out)
    }


class BaseReadout(BaseFFLayer):
    def __init__(self,
        num_blocks: int,
        output_fields: Set[str],
        built_layers: List[Module],
        head_type: HEAD_TYPE,
        dim_embedding: Optional[int]=None,
        shallow_ensemble_size: int=1,
        keep_feature: bool=False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        feature_irreps: Optional[str]=None,
        **head_params
    ) -> None:
        if not isinstance(output_fields, set):
            output_fields = set(output_fields)
        super().__init__(
            input_fields=["atom_feature"],
            output_fields=output_fields | {"atom_feature"} if keep_feature else output_fields
        )
        self.num_blocks = num_blocks
        self.shallow_ensemble_size = shallow_ensemble_size
        self.ordered_output_fields = sorted(list(output_fields))
        self.head_type = head_type
        self.feature_irreps = (
            feature_irreps
            if feature_irreps is not None
            else _resolve_feature_irreps(built_layers)
        )
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

    def _scalar_atom_feature(self, atom_feature: Tensor) -> Tensor:
        """Take last hierarchical block if needed, then extract 0e when equivariant."""
        if atom_feature.ndim == 3:
            atom_feature = atom_feature[:, :, -1]
        elif atom_feature.ndim != 2:
            raise ValueError(
                f"atom_feature must be 2D or 3D, got shape {tuple(atom_feature.shape)}"
            )
        return extract_scalar_0e(atom_feature, self.feature_irreps)

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
        elif self.head_type == "two_layer":
            # Dense MLP morphologically like the official energy head (not LinearRS).
            act = self.activation_fn if self.activation_fn is not None else "swish"
            return Sequential(
                DenseLayer(
                    dim_feature_in=self.dim_feature_in,
                    dim_feature_out=self.dim_feature_in,
                    activation_fn=act,
                    activation_params=self.activation_params,
                ),
                DenseLayer(
                    dim_feature_in=self.dim_feature_in,
                    dim_feature_out=self.dim_feature_out,
                    shallow_ensemble_size=self.shallow_ensemble_size,
                    **self.head_params,
                ),
            )
        elif self.head_type == "equiformer_linear_rs":
            # Official Equiformer MD17 scalar energy MLP:
            # LinearRS → Activation(normalize2mom(SiLU)) → LinearRS.
            # Final LinearRS is widened by shallow_ensemble_size; reshape happens
            # in split_readout_field_outputs via reshape_ensemble_head_output.
            from e3nn import o3
            from ..equiformer.interaction import _RESCALE
            from ..e3nn_nn import Activation, LinearRS

            ir_hid = o3.Irreps(f"{self.dim_feature_in}x0e")
            n_out = self.dim_feature_out * self.shallow_ensemble_size
            ir_out = o3.Irreps(f"{n_out}x0e")
            return Sequential(
                LinearRS(ir_hid, ir_hid, rescale=_RESCALE),
                Activation(ir_hid, acts=[torch.nn.SiLU()]),
                LinearRS(ir_hid, ir_out, rescale=_RESCALE),
            )
        else:
            raise ValueError(f"Unknown head_type: {self.head_type}")

    def _split_field_outputs(self, head_output: Tensor) -> Dict[str, Tensor]:
        return split_readout_field_outputs(
            head_output,
            self.ordered_output_fields,
            self.shallow_ensemble_size,
        )


class SimpleReadout(BaseReadout):
    def __init__(self,
        output_fields: Set[str],
        built_layers: List[Module],
        head_type: HEAD_TYPE="dense",
        dim_embedding: Optional[int]=None,
        shallow_ensemble_size: int=1,
        keep_feature: bool=False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE]=None,
        activation_params: ACTIVATION_PARAM_TYPE=dict(),
        feature_irreps: Optional[str]=None,
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
            feature_irreps=feature_irreps,
            **head_params
        )
        self.head = self._get_head()

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        return self._split_field_outputs(self.head(self._scalar_atom_feature(atom_feature)))


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
        head_type: HEAD_TYPE = "residual_mlp",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        positive_activation_fn: POSITIVE_ACTIVATION_KEY_TYPE = "softplus",
        feature_irreps: Optional[str] = None,
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
            feature_irreps=feature_irreps,
            **head_params,
        )
        self.head = self._get_head()
        self.positive_activation_fn = get_positive_activation_fn(positive_activation_fn)

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        output = self.head(self._scalar_atom_feature(atom_feature))
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
            block = extract_scalar_0e(atom_feature[:, :, i], self.feature_irreps)
            raw_output += self.heads[i](block)
            if self.use_nhloss:
                output2 = raw_output ** 2
                if i > 0:
                    nhloss += torch.mean(output2 / (output2 + lastoutput2 + 1e-7))
                lastoutput2 = output2
        output = self._split_field_outputs(raw_output)
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
        head_type: HEAD_TYPE = "residual_mlp",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        positive_activation_fn: POSITIVE_ACTIVATION_KEY_TYPE = "softplus",
        feature_irreps: Optional[str] = None,
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
            feature_irreps=feature_irreps,
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
            block = extract_scalar_0e(atom_feature[:, :, i], self.feature_irreps)
            raw_output = raw_output + self.heads[i](block)

        results: Dict[str, Tensor] = {}
        for i, output_field in enumerate(self.ordered_output_fields):
            if output_field.startswith("fa"):
                results[output_field] = self.positive_activation_fn(raw_output[:, i])
            else:
                results[output_field] = raw_output[:, i]
        return results


class VelocityReadout(BaseReadout):
    """Per-atom velocity head for flow matching (e.g. charge and spin channel velocities)."""

    def __init__(
        self,
        output_fields: Optional[Set[str]] = None,
        built_layers: List[Module] = [],
        head_type: HEAD_TYPE = "dense",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        feature_irreps: Optional[str] = None,
        **head_params,
    ) -> None:
        if output_fields is None:
            output_fields = {"Q_vel_a", "S_vel_a"}
        elif not isinstance(output_fields, set):
            output_fields = set(output_fields)
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
            feature_irreps=feature_irreps,
            **head_params,
        )
        self.head = self._get_head()

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        output = self.head(self._scalar_atom_feature(atom_feature))
        return {
            self.ordered_output_fields[i]: output[:, i] for i in range(self.dim_feature_out)
        }


class HirshfeldReadout(BaseFFLayer):
    """Hirshfeld volume-ratio head (So3krates-torch ``HirshfeldOutputHead``).

    Outputs ``ha = |v_shift + (q ⊙ k) / √d|`` for TS–QDO dispersion (SO3LR).
    SO3LR partial charges use ``SimpleReadout(Qa)`` + ``AtomicAffine(scale=1)``
    instead of a dedicated charge head.
    """

    def __init__(
        self,
        dim_embedding: Optional[int] = None,
        built_layers: Optional[List[Module]] = None,
        regression_dim: Optional[int] = None,
        max_Za: int = 100,
        activation_fn: Optional[Callable[[], Module]] = None,
        **kwargs,
    ) -> None:
        del kwargs
        if dim_embedding is None and built_layers:
            for layer in reversed(built_layers):
                if hasattr(layer, "dim_feature_out"):
                    dim_embedding = int(layer.dim_feature_out)
                    break
                if hasattr(layer, "dim_embedding"):
                    dim_embedding = int(layer.dim_embedding)
                    break
        if dim_embedding is None:
            raise TypeError("dim_embedding value should be provided")
        if dim_embedding % 2 != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be even for HirshfeldReadout"
            )
        super().__init__(
            input_fields={"atom_feature", "Za"},
            output_fields={"ha"},
        )
        half = dim_embedding // 2
        self.v_shift_embedding = Embedding(max_Za + 1, 1)
        self.q_embedding = Embedding(max_Za + 1, half)
        act = Identity if activation_fn is None else activation_fn
        if regression_dim is not None:
            self.transform = Sequential(
                Linear(dim_embedding, regression_dim // 2),
                act(),
                Linear(regression_dim // 2, half),
            )
        else:
            self.transform = Linear(dim_embedding, half)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in (self.v_shift_embedding, self.q_embedding):
            std = 1.0 / (emb.embedding_dim ** 0.5)
            torch.nn.init.normal_(emb.weight, mean=0.0, std=std)
        modules = (
            self.transform
            if isinstance(self.transform, Sequential)
            else [self.transform]
        )
        for m in modules:
            if isinstance(m, Linear):
                std_m = 1.0 / (m.in_features ** 0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std_m)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_ha(self, atom_feature: Tensor, Za: Tensor) -> Tensor:
        v_shift = self.v_shift_embedding(Za.long()).squeeze(-1)
        q = self.q_embedding(Za.long())
        k = self.transform(atom_feature)
        qk = (q * k * (1.0 / math.sqrt(k.shape[-1]))).sum(dim=-1)
        return torch.abs(v_shift + qk)
