from typing import Set, Literal, Dict, Optional, List, Callable
from . import BaseFFLayer
from ..mp.repr import IrrepsSpec
from ..mp.nn.mlp import DenseLayer, GatedMLP, ResidualMLP, ResidualLayer, VariancePreservingMLP
from ..mp.nn.activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, POSITIVE_ACTIVATION_KEY_TYPE, get_positive_activation_fn
from ..mp.equivariant.e3nn import extract_scalar_0e
import math
import torch
from torch.nn import Embedding, Identity, Linear, ModuleList, Module, Sequential
from torch import Tensor


HEAD_TYPE = Literal[
    "dense",
    "residual_layer",
    "residual_mlp",
    "two_layer",
    "mlp",
    "equiformer_linear_rs",
    "variance_preserving",
    "gated",
]


class _IdentityResidual(Module):
    """Add a dense map back to its input when the two shapes agree."""

    def __init__(self, layer: Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer(x)
        if y.shape == x.shape:
            return x + y
        return y


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

    Head layout matches :class:`~enerzyme.models.mp.nn.mlp.DenseLayer`:
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
        spec = IrrepsSpec.from_layers(built_layers, dim_embedding)
        self.feature_irreps = feature_irreps if feature_irreps is not None else spec.irreps
        self.dim_feature_in = spec.scalar_width
        self._hinted_hidden_width = None
        for layer in reversed(built_layers):
            hinted = getattr(layer, "readout_hidden_width", None)
            if hinted is not None:
                self._hinted_hidden_width = int(hinted)
                break
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
            # ``pyramidal: true`` uses one hidden width max(n_out, F // 2), the
            # SchNetPack ``Atomwise(n_layers=2)`` schedule. ``dim_hidden`` sets
            # that width explicitly. The default keeps the hidden width at F.
            head_params = dict(self.head_params)
            pyramidal = bool(head_params.pop("pyramidal", False))
            dim_hidden = head_params.pop("dim_hidden", None)
            # EquFlash AtomicReduce is two bias-free linears. ``activate_hidden``
            # false skips the hidden activation; ``use_bias`` applies to both layers.
            activate_hidden = bool(head_params.pop("activate_hidden", True))
            use_bias = bool(head_params.pop("use_bias", True))
            if pyramidal and dim_hidden is not None:
                raise ValueError(
                    "SimpleReadout two_layer accepts pyramidal or dim_hidden, not both"
                )
            if pyramidal:
                dim_hidden = max(self.dim_feature_out, self.dim_feature_in // 2)
            elif dim_hidden is None:
                dim_hidden = self.dim_feature_in
            else:
                dim_hidden = int(dim_hidden)
            if activate_hidden:
                act = self.activation_fn if self.activation_fn is not None else "swish"
            else:
                act = None
            return Sequential(
                DenseLayer(
                    dim_feature_in=self.dim_feature_in,
                    dim_feature_out=dim_hidden,
                    activation_fn=act,
                    activation_params=self.activation_params,
                    use_bias=use_bias,
                ),
                DenseLayer(
                    dim_feature_in=dim_hidden,
                    dim_feature_out=self.dim_feature_out,
                    shallow_ensemble_size=self.shallow_ensemble_size,
                    use_bias=use_bias,
                    **head_params,
                ),
            )
        elif self.head_type == "mlp":
            # DimeNet-style stacked dense hidden layers + linear property head.
            # ``hidden_dims`` sets each hidden width (NewtonNet v1 is
            # ``[128, 64]``). Omitting it keeps every hidden layer at the
            # incoming feature width.
            head_params = dict(self.head_params)
            num_hidden_layers = head_params.pop("num_hidden_layers", None)
            hidden_dims = head_params.pop("hidden_dims", None)
            use_bias_out = bool(head_params.pop("use_bias_out", False))
            act = self.activation_fn if self.activation_fn is not None else "swish"
            if hidden_dims is None:
                depth = 3 if num_hidden_layers is None else int(num_hidden_layers)
                if depth < 0:
                    raise ValueError(f"num_hidden_layers must be >= 0, got {depth}")
                hidden_dims = [self.dim_feature_in] * depth
            else:
                hidden_dims = [int(width) for width in hidden_dims]
                if num_hidden_layers is not None and int(num_hidden_layers) != len(hidden_dims):
                    raise ValueError(
                        "mlp hidden_dims length must equal num_hidden_layers, "
                        f"got {hidden_dims} and num_hidden_layers={num_hidden_layers}"
                    )
            if any(width < 1 for width in hidden_dims):
                raise ValueError(f"mlp hidden widths must be >= 1, got {hidden_dims}")
            # DPA4C's fitting net adds the input of every hidden map whose
            # width is unchanged. The scalar head stays a plain linear map.
            identity_residual = bool(head_params.pop("identity_residual", False))
            layers = []
            in_dim = self.dim_feature_in
            for width in hidden_dims:
                dense = DenseLayer(
                    dim_feature_in=in_dim,
                    dim_feature_out=width,
                    activation_fn=act,
                    activation_params=self.activation_params,
                    **head_params,
                )
                if identity_residual and in_dim == width:
                    layers.append(_IdentityResidual(dense))
                else:
                    layers.append(dense)
                in_dim = width
            layers.append(
                DenseLayer(
                    dim_feature_in=in_dim,
                    dim_feature_out=self.dim_feature_out,
                    use_bias=use_bias_out,
                    shallow_ensemble_size=self.shallow_ensemble_size,
                    **head_params,
                )
            )
            return Sequential(*layers)
        elif self.head_type == "variance_preserving":
            # Allegro / NequIP scalar MLP: SiLU hidden layers, bias-free by default,
            # variance-preserving uniform init. Hidden width falls back to the
            # Core's ``readout_hidden_width`` (Allegro scalar channels) and then
            # to the latent width.
            head_params = dict(self.head_params)
            depth = int(head_params.pop("hidden_layers_depth", 1))
            width = head_params.pop("hidden_layers_width", None)
            nonlinearity = head_params.pop("nonlinearity", None)
            use_bias = bool(head_params.pop("use_bias", False))
            forward_init = bool(head_params.pop("forward_weight_init", True))
            if head_params:
                unknown = ", ".join(sorted(head_params))
                raise ValueError(
                    f"variance_preserving head received unknown params: {unknown}"
                )
            if nonlinearity is None:
                nonlinearity = self.activation_fn if self.activation_fn is not None else "silu"
            if width is None and self._hinted_hidden_width is not None:
                width = self._hinted_hidden_width
            if width is None:
                width = self.dim_feature_in
            return VariancePreservingMLP(
                input_dim=self.dim_feature_in,
                output_dim=self.dim_feature_out * self.shallow_ensemble_size,
                hidden_layers_depth=depth,
                hidden_layers_width=None if depth == 0 else int(width),
                nonlinearity=None if depth == 0 else nonlinearity,
                bias=use_bias,
                forward_weight_init=forward_init,
            )
        elif self.head_type == "gated":
            # M3GNet atomic head: SiLU hidden gates, sigmoid on the last gate,
            # and no activation on the last value. Hidden width follows ``units``,
            # then the Core's ``readout_hidden_width``, then the latent width.
            head_params = dict(self.head_params)
            hidden_dims = head_params.pop("hidden_dims", None)
            units = head_params.pop("units", None)
            n_hidden = int(head_params.pop("num_hidden_layers", 2))
            activate_last = bool(head_params.pop("activate_last", False))
            use_bias = bool(head_params.pop("use_bias", True))
            if head_params:
                unknown = ", ".join(sorted(head_params))
                raise ValueError(f"gated head received unknown params: {unknown}")
            if hidden_dims is not None and units is not None:
                raise ValueError("gated head accepts hidden_dims or units, not both")
            if n_hidden < 0:
                raise ValueError(f"num_hidden_layers must be >= 0, got {n_hidden}")
            if hidden_dims is None:
                if units is None and self._hinted_hidden_width is not None:
                    units = self._hinted_hidden_width
                if units is None:
                    units = self.dim_feature_in
                hidden_dims = [int(units)] * n_hidden
            return GatedMLP(
                in_features=self.dim_feature_in,
                hidden_dims=[int(width) for width in hidden_dims],
                out_features=self.dim_feature_out * self.shallow_ensemble_size,
                activate_last=activate_last,
                use_bias=use_bias,
            )
        elif self.head_type == "equiformer_linear_rs":
            # Official Equiformer MD17 scalar energy MLP:
            # LinearRS → Activation(normalize2mom(SiLU)) → LinearRS.
            # Final LinearRS is widened by shallow_ensemble_size; reshape happens
            # in split_readout_field_outputs via reshape_ensemble_head_output.
            from e3nn import o3
            from ..equiformer.interaction import _RESCALE
            from ..mp.equivariant.e3nn import Activation, LinearRS

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


class EdgeReadout(SimpleReadout):
    """Property head on a per-edge latent.

    Same heads as :class:`SimpleReadout`, reading ``edge_feature`` instead of
    ``atom_feature``. Allegro writes ``Eij`` here; :class:`EdgeEnergyReduceLayer`
    scatters that pair energy onto atoms. ``head_type: variance_preserving``
    is the Allegro / NequIP scalar MLP.
    """

    def __init__(
        self,
        output_fields: Set[str],
        built_layers: List[Module],
        head_type: HEAD_TYPE = "variance_preserving",
        dim_embedding: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        keep_feature: bool = False,
        activation_fn: Optional[ACTIVATION_KEY_TYPE] = None,
        activation_params: ACTIVATION_PARAM_TYPE = dict(),
        feature_irreps: Optional[str] = None,
        **head_params,
    ) -> None:
        super().__init__(
            output_fields=output_fields,
            built_layers=built_layers,
            head_type=head_type,
            dim_embedding=dim_embedding,
            shallow_ensemble_size=shallow_ensemble_size,
            keep_feature=False,
            activation_fn=activation_fn,
            activation_params=activation_params,
            feature_irreps=feature_irreps,
            **head_params,
        )
        self._input_fields = {"edge_feature"}
        fields = set(output_fields)
        if keep_feature:
            fields.add("edge_feature")
        self._output_fields = fields
        self._relevant_fields = self._input_fields | self._output_fields
        self._name_mapping = {name: name for name in self._relevant_fields}
        self.keep_edge_feature = bool(keep_feature)

    def get_output(self, edge_feature: Tensor) -> Dict[str, Tensor]:
        outputs = self._split_field_outputs(
            self.head(self._scalar_atom_feature(edge_feature))
        )
        if self.keep_edge_feature:
            outputs["edge_feature"] = edge_feature
        return outputs


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
    def __init__(
        self,
        use_nhloss: bool = False,
        num_blocks: Optional[int] = None,
        built_layers: Optional[List[Module]] = None,
        **kwargs,
    ) -> None:
        if built_layers is None:
            built_layers = kwargs.pop("built_layers", [])
        if num_blocks is None:
            for layer in reversed(built_layers):
                inferred = getattr(layer, "num_output_blocks", None)
                if inferred is not None:
                    num_blocks = int(inferred)
                    break
        if num_blocks is None:
            raise TypeError(
                "HierachicalReadout needs num_blocks or a prior Core with "
                "num_output_blocks"
            )
        super().__init__(
            num_blocks=num_blocks, built_layers=built_layers, **kwargs
        )
        self.heads = ModuleList([self._get_head() for _ in range(self.num_blocks)])
        self.use_nhloss = use_nhloss

    def get_output(self, atom_feature: Tensor) -> Dict[str, Tensor]:
        if atom_feature.ndim != 3 or atom_feature.shape[-1] != self.num_blocks:
            raise ValueError(
                "HierachicalReadout expects atom_feature of shape "
                f"(N, F, {self.num_blocks}), got {tuple(atom_feature.shape)}. "
                "A Core output_mode of 'last' is the input of SimpleReadout."
            )
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


class AbsoluteLinearReadout(BaseFFLayer):
    """Linear map of a scalar latent, then an absolute value.

    CHGNet reads the magnetic-moment magnitude from ``atom_feature_charge``,
    the atom state before the last convolution. The absolute value matches
    training on ``|m|``. The head is not part of the interaction.
    """

    def __init__(
        self,
        output_fields: Set[str],
        built_layers: List[Module],
        use_bias: bool = True,
        shallow_ensemble_size: int = 1,
    ) -> None:
        if not isinstance(output_fields, set):
            output_fields = set(output_fields)
        if len(output_fields) < 1:
            raise ValueError("AbsoluteLinearReadout needs at least one output field")
        super().__init__(
            input_fields={"atom_feature_charge"},
            output_fields=set(output_fields),
        )
        self.ordered_output_fields = sorted(output_fields)
        self.shallow_ensemble_size = int(shallow_ensemble_size)
        if self.shallow_ensemble_size < 1:
            raise ValueError(
                f"shallow_ensemble_size must be >= 1, got {shallow_ensemble_size}"
            )
        dim_in = None
        for layer in reversed(built_layers):
            width = getattr(layer, "dim_feature_out", None)
            if width is None:
                continue
            # A property head reports one channel per field. The latent width
            # is the interaction output, which that head leaves unchanged.
            fields = getattr(layer, "ordered_output_fields", None)
            inner = getattr(layer, "dim_feature_in", None)
            if (
                fields is not None
                and int(width) == len(fields)
                and inner is not None
                and int(inner) != int(width)
            ):
                continue
            dim_in = int(width)
            break
        if dim_in is None:
            raise ValueError(
                "AbsoluteLinearReadout requires dim_feature_out on a previous layer"
            )
        self.dim_feature_in = dim_in
        # The linear map has one channel per field. dim_feature_out stays at
        # the latent width so a later energy head still reads atom_feature.
        n_out = len(self.ordered_output_fields)
        self.dim_feature_out = dim_in
        self.head = DenseLayer(
            dim_feature_in=self.dim_feature_in,
            dim_feature_out=n_out,
            use_bias=bool(use_bias),
            shallow_ensemble_size=self.shallow_ensemble_size,
        )

    def get_output(self, atom_feature_charge: Tensor) -> Dict[str, Tensor]:
        if atom_feature_charge.ndim != 2 or atom_feature_charge.shape[-1] != self.dim_feature_in:
            raise ValueError(
                "atom_feature_charge must be "
                f"(N, {self.dim_feature_in}), got {tuple(atom_feature_charge.shape)}"
            )
        projected = self.head(atom_feature_charge).abs()
        return split_readout_field_outputs(
            projected,
            self.ordered_output_fields,
            self.shallow_ensemble_size,
        )
