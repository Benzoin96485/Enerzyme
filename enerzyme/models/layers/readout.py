from typing import Set, Literal, Dict, Optional, List
from . import BaseFFLayer
from ..blocks.mlp import DenseLayer, ResidualMLP, ResidualLayer
from ..activation import ACTIVATION_KEY_TYPE, ACTIVATION_PARAM_TYPE, POSITIVE_ACTIVATION_KEY_TYPE, get_positive_activation_fn
from ..irreps_tools import extract_scalar_0e
import torch
from torch.nn import ModuleList, Module, Sequential
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
            from ..equiformer.attention import _RESCALE
            from ..equiformer.fast_activation import Activation
            from ..equiformer.tensor_product import LinearRS

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


class EquiformerGraphAttentionReadout(BaseFFLayer):
    """Equivariant GraphAttention head mapping full irreps → atomic scalars.

    Unlike :class:`SimpleReadout` (0e extract + MLP), this consumes the full
    ``atom_feature`` irreps tensor and graph edges, producing one 0e channel
    per ``output_fields`` entry (e.g. ``Ea``, ``Qa``). Requires
    ``feature_irreps`` from a prior equivariant Core.

    With ``shallow_ensemble_size > 1``, the final GraphAttention ``LinearRS``
    proj is widened and fields keep a trailing ensemble axis.
    """

    def __init__(
        self,
        output_fields: Set[str],
        built_layers: List[Module],
        irreps_head: str = "16x0e+8x1o+4x2e",
        num_heads: int = 2,
        fc_neurons: Optional[List[int]] = None,
        irreps_sh: str = "1x0e+1x1e+1x2e",
        irreps_node_attr: str = "1x0e",
        irreps_pre_attn: Optional[str] = None,
        rescale_degree: bool = False,
        nonlinear_message: bool = True,
        alpha_drop: float = 0.0,
        proj_drop: float = 0.0,
        feature_irreps: Optional[str] = None,
        num_rbf: Optional[int] = None,
        shallow_ensemble_size: int = 1,
        **kwargs,
    ) -> None:
        from e3nn import o3
        from ..equiformer.attention import GraphAttention

        ordered = sorted(list(output_fields))
        super().__init__(
            input_fields={
                "atom_feature",
                "idx_i_sr",
                "idx_j_sr",
                "vij_sr",
                "rbf",
                "batch_seg",
            },
            output_fields=set(ordered),
        )
        self.ordered_output_fields = ordered
        self.dim_feature_out = len(ordered)
        self.shallow_ensemble_size = int(shallow_ensemble_size)

        irreps_str = (
            feature_irreps
            if feature_irreps is not None
            else _resolve_feature_irreps(built_layers)
        )
        if irreps_str is None:
            raise ValueError(
                "EquiformerGraphAttentionReadout requires feature_irreps "
                "(from an equivariant Core) or an explicit feature_irreps=..."
            )
        self.feature_irreps = o3.Irreps(irreps_str)
        self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)

        if fc_neurons is None:
            fc_neurons = [64, 64]
        if num_rbf is None:
            num_rbf = 16
            for layer in reversed(built_layers):
                if hasattr(layer, "num_rbf"):
                    num_rbf = int(layer.num_rbf)
                    break
                if hasattr(layer, "fc_neurons") and isinstance(layer.fc_neurons, list) and layer.fc_neurons:
                    num_rbf = int(layer.fc_neurons[0])
                    break
        self.fc_neurons = [num_rbf] + list(fc_neurons)

        n_out = self.dim_feature_out * self.shallow_ensemble_size
        self.head = GraphAttention(
            irreps_node_input=self.feature_irreps,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=o3.Irreps(f"{n_out}x0e"),
            fc_neurons=self.fc_neurons,
            irreps_head=o3.Irreps(irreps_head),
            num_heads=num_heads,
            irreps_pre_attn=irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
        )

    def get_output(
        self,
        atom_feature: Tensor,
        idx_i_sr: Tensor,
        idx_j_sr: Tensor,
        vij_sr: Tensor,
        rbf: Tensor,
        batch_seg: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        from e3nn import o3

        if atom_feature.ndim != 2:
            raise ValueError(
                "EquiformerGraphAttentionReadout expects 2D atom_feature "
                f"(N, irreps.dim); got shape {tuple(atom_feature.shape)}"
            )
        if atom_feature.shape[-1] != self.feature_irreps.dim:
            raise ValueError(
                f"atom_feature last dim {atom_feature.shape[-1]} != "
                f"feature_irreps.dim {self.feature_irreps.dim}"
            )
        n_atoms = atom_feature.shape[0]
        if batch_seg is None:
            batch_seg = torch.zeros(n_atoms, dtype=torch.long, device=atom_feature.device)
        # Enerzyme edge convention: edge_src = idx_j, edge_dst = idx_i
        edge_src = idx_j_sr
        edge_dst = idx_i_sr
        node_attr = torch.ones_like(atom_feature.narrow(1, 0, 1))
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=vij_sr,
            normalize=True,
            normalization="component",
        )
        outputs = self.head(
            node_input=atom_feature,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=rbf,
            batch=batch_seg,
        )
        expected_last = self.dim_feature_out * self.shallow_ensemble_size
        if outputs.ndim != 2 or outputs.shape[-1] != expected_last:
            raise ValueError(
                f"GraphAttention output shape {tuple(outputs.shape)} != "
                f"(N, {expected_last})"
            )
        return split_readout_field_outputs(
            outputs,
            self.ordered_output_fields,
            self.shallow_ensemble_size,
        )


class EquiformerV2FeedForwardReadout(BaseFFLayer):
    """Paper EquiformerV2 energy head: sphere FFN → per-atom scalars.

    Consumes ``atom_sphere_feature`` and Core ``SO3_grid`` / FFN hyperparameters.
    Default production stacks use ``SimpleReadout`` on ``atom_feature`` instead.

    With ``shallow_ensemble_size > 1``, the final ``SO3_LinearV2`` is widened;
    only l=0 channels are used as field × ensemble scalars.
    """

    def __init__(
        self,
        output_fields: Set[str],
        built_layers: List[Module],
        ffn_hidden_channels: Optional[int] = None,
        keep_feature: bool = False,
        shallow_ensemble_size: int = 1,
        **_unused,
    ) -> None:
        out = set(output_fields)
        if keep_feature:
            out = out | {"atom_feature", "atom_sphere_feature"}
        super().__init__(
            input_fields={"atom_sphere_feature"},
            output_fields=out,
        )
        self.ordered_output_fields = sorted(list(output_fields))
        self.keep_feature = keep_feature
        self.shallow_ensemble_size = int(shallow_ensemble_size)
        self.dim_feature_out = len(self.ordered_output_fields)

        core = None
        for layer in reversed(built_layers):
            if hasattr(layer, "SO3_grid") and hasattr(layer, "sphere_channels"):
                core = layer
                break
        if core is None:
            raise ValueError(
                "EquiformerV2FeedForwardReadout requires a preceding EquiformerV2 Core"
            )

        from ..equiformer_v2.transformer_block import FeedForwardNetwork
        from ..so3 import SO3_Embedding

        self._SO3_Embedding = SO3_Embedding
        n_out = self.dim_feature_out * self.shallow_ensemble_size
        hidden = (
            ffn_hidden_channels
            if ffn_hidden_channels is not None
            else core.ffn_hidden_channels
        )
        self.ffn = FeedForwardNetwork(
            sphere_channels=core.sphere_channels,
            hidden_channels=hidden,
            output_channels=n_out,
            lmax_list=list(core.lmax_list),
            mmax_list=list(core.mmax_list),
            SO3_grid=core.SO3_grid,
            activation=getattr(core, "ffn_activation", "scaled_silu"),
            use_gate_act=getattr(core, "use_gate_act", False),
            use_grid_mlp=getattr(core, "use_grid_mlp", False),
            use_sep_s2_act=getattr(core, "use_sep_s2_act", True),
        )
        self.lmax_list = list(core.lmax_list)
        self.mmax_list = list(core.mmax_list)
        self.sphere_channels = core.sphere_channels

    def get_output(self, atom_sphere_feature: Tensor) -> Dict[str, Tensor]:
        x = self._SO3_Embedding(
            0,
            self.lmax_list.copy(),
            self.sphere_channels,
            device=atom_sphere_feature.device,
            dtype=atom_sphere_feature.dtype,
        )
        x.set_embedding(atom_sphere_feature)
        # Node features after Core are full degree (mmax == lmax); match Core output.
        x.set_lmax_mmax(self.lmax_list.copy(), self.lmax_list.copy())
        node_out = self.ffn(x)
        # l=0 channels only → (N, n_fields * ensemble)
        scalars = node_out.embedding.narrow(1, 0, 1).squeeze(1)
        result = split_readout_field_outputs(
            scalars,
            self.ordered_output_fields,
            self.shallow_ensemble_size,
        )
        if self.keep_feature:
            result["atom_sphere_feature"] = atom_sphere_feature
            # Concatenate l=0,m=0 channels across resolutions (same as Core).
            features = []
            offset_res = 0
            for i, lmax in enumerate(self.lmax_list):
                features.append(atom_sphere_feature[:, offset_res, :])
                offset_res = offset_res + int((lmax + 1) ** 2)
            result["atom_feature"] = torch.cat(features, dim=-1)
        return result
