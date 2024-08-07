from typing import Dict, Any
from ..layers import DistanceLayer, BaseRBF, BaseAtomEmbedding, BaseElectronEmbedding
from .interaction import InteractionModule
from torch.nn import Module, ModuleList, Linear

class SpookyNetCore(Module):
    def __str__(self) -> str:
        return """
###############################################
# SpookyNet (Nat. Commun., 2021, 12(1): 7273) #
###############################################
"""

    def __init__(
        self, dim_embedding: int, num_rbf: int, num_modules: int, num_residual_pre: int,
        num_residual_local_x: int, num_residual_local_s: int, num_residual_local_p: int, num_residual_local_d: int, num_residual_local: int,
        num_residual_nonlocal_q: int, num_residual_nonlocal_k: int, num_residual_nonlocal_v: int,
        num_residual_post: int, num_residual_output: int, activation_fn: int
    ) -> None:
        self.nuclear_embeddings: BaseAtomEmbedding = None
        self.charge_embeddings: BaseElectronEmbedding = None
        self.spin_embeddings: BaseElectronEmbedding = None
        self.radial_basis_functions: BaseRBF = None
        self.interaction = ModuleList(
            [
                InteractionModule(
                    dim_embedding=dim_embedding,
                    num_rbf=num_rbf,
                    num_residual_pre=num_residual_pre,
                    num_residual_local_x=num_residual_local_x,
                    num_residual_local_s=num_residual_local_s,
                    num_residual_local_p=num_residual_local_p,
                    num_residual_local_d=num_residual_local_d,
                    num_residual_local=num_residual_local,
                    num_residual_nonlocal_q=num_residual_nonlocal_q,
                    num_residual_nonlocal_k=num_residual_nonlocal_k,
                    num_residual_nonlocal_v=num_residual_nonlocal_v,
                    num_residual_post=num_residual_post,
                    num_residual_output=num_residual_output,
                    activation_fn=activation_fn,
                )   
                for _ in range(num_modules)
            ]
        )
        self.output = Linear(dim_embedding, 2, bias=False)

    @classmethod
    def build(cls, built_layers: Dict[str, Module], **build_params: Dict[str, Any]) -> Module:
        instance = cls(**build_params)
        for layer_name, layer in built_layers.items():
            if isinstance(layer, DistanceLayer):
                instance.distance_layer = layer
            elif isinstance(layer, BaseRBF):
                instance.rbf_layer = layer
            elif isinstance(layer, BaseAtomEmbedding):
                instance.nuclear_embeddings = layer
            elif isinstance(layer, BaseElectronEmbedding):
                if layer.attribute == "spin":
                    instance.spin_embeddings = layer
                elif layer.attribute == "charge":
                    instance.charge_embeddings = layer
        for layer_name in ["distance_layer", "rbf_layer", "nuclear_embeddings", "spin_embeddings", "charge_embeddings"]:
            if getattr(instance, layer_name) is None:
                raise AttributeError(f"{layer_name} is not built")
        return instance