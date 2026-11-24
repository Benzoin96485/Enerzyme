"""Compile force-field inference partitions or joint force-training derivatives."""

from typing import Any, Dict

import torch
from torch import Tensor
from torch_geometric.data import Data

from .gradient import ForceLayer


@torch.compiler.assume_constant_result
def compiling_fullgraph() -> bool:
    """Keep strict traces in the joint energy/force transform.

    Dynamo supplies fullgraph mode on its translator, with no public
    torch.compiler predicate. Read it at trace time only. Other tracing
    contexts continue to use the self-contained functional VJP.
    """
    from torch._dynamo.symbolic_convert import InstructionTranslator

    try:
        return InstructionTranslator.current_tx().one_graph
    except AttributeError:
        return True


@torch.compiler.disable(recursive=False)
def retained_inference(core, inputs):
    """Compile each inference partition with reusable backward activations.

    Force evaluation keeps the energy graph for other response heads and for
    subsequent gradients requested by the caller. An eagerly lowered dynamic
    backward can otherwise overwrite its saved tensors on the first use.
    Only this configuration frame is skipped, so the layers still compile.
    """
    from contextlib import nullcontext
    from torch._functorch import config

    if not hasattr(config, "donated_buffer"):
        return core._run_layers(inputs)

    # The Inductor FX cache key on PyTorch 2.13 omits AOTAutograd's donated
    # input indices. A cached single-use backward can overwrite buffers even
    # when this invocation requested reusable ones. Keep the caller's tag.
    compiler_config = getattr(torch.compiler, "config", None)
    if hasattr(compiler_config, "cache_key_tag"):
        cache_scope = compiler_config.patch(
            cache_key_tag=compiler_config.cache_key_tag + ":enerzyme-retained-backward"
        )
    else:
        cache_scope = nullcontext()
    with config.patch(donated_buffer=False), cache_scope:
        return core._run_layers(inputs)


def forward_with_coordinate_gradients(core, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Functional coordinate VJPs make force fitting a single compiled backward.

    The prefix ending at each Force layer returns energy and tensor-valued
    auxiliary outputs in one pass. This preserves random draws and lets losses
    on forces and other heads differentiate the same weights.
    """
    def active(layer):
        return not ((layer.test_only and not core.test_mode) or
                    (layer.test_exclude and core.test_mode))

    stages = [layer for layer in core.pre_sequence if active(layer)]
    stages.append(core._forward)
    stages.extend(layer for layer in core.post_sequence if active(layer))

    def evaluate(net_input, stop):
        # A later Force layer can differentiate quantities produced by an earlier
        # one. Only the innermost prefix runs the actual message-passing stack.
        start = 0
        for i in reversed(range(stop)):
            layer = stages[i]
            if type(layer) is not ForceLayer:
                continue
            if (layer.train_only and not layer.training) or (layer.eval_only and layer.training):
                continue
            position_name, energy_name, force_name = (layer._name_mapping[k] for k in ("Ra", "E", "Fa"))
            if position_name not in net_input:
                continue

            def energy(positions):
                state = evaluate({**net_input, position_name: positions}, i)
                # Non-tensor input metadata (e.g. a list of molecule sizes)
                # passes through the layer stack and does not enter the VJP.
                return state[energy_name], {key: value for key, value in state.items() if isinstance(value, Tensor)}

            energies, pullback, tensor_fields = torch.func.vjp(energy, net_input[position_name], has_aux=True)
            if energies.ndim > 1:
                weights = torch.eye(energies.shape[-1], device=energies.device, dtype=energies.dtype)
                forces = torch.stack([
                    -pullback(row.expand_as(energies))[0] for row in weights
                ], dim=-1)
            else:
                forces = -pullback(torch.ones_like(energies))[0]
            net_input = {**net_input, **tensor_fields, force_name: forces if layer.training else forces.detach()}
            start = i + 1
            break
        for i in range(start, stop):
            net_input = stages[i](net_input)
        return net_input

    if isinstance(inputs, Data):
        # PyG spells the membership vector "batch" and updates its Data in place.
        fields = {name: value for name, value in inputs}
        fields["batch_seg"] = fields.get("batch")
        for name, value in evaluate(fields, len(stages)).items():
            if name != "batch_seg":
                inputs[name] = value
        return inputs
    return evaluate(inputs, len(stages))
