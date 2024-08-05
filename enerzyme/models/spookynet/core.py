from typing import Dict, Any
from ..layers import DistanceLayer, BaseRBF, BaseAtomEmbedding, BaseElectronEmbedding
from .interaction import InteractionModule
from torch import nn
from torch.nn import Module, ModuleList

class SpookyNetCore(Module):
    """
    Neural network for PES construction augmented with optional explicit terms
    for short-range repulsion, electrostatics and dispersion and explicit nonlocal
    interactions.
    IMPORTANT: Angstrom and electron volts are assumed to be the units for
    length and energy (charge is measured in elementary charge). When other
    units are used, some constants for computing short-range repulsion,
    electrostatics and dispersion need to be changed accordingly. If these terms
    are not used, no changes are necessary. It is recommended to work with units
    of Angstrom and electron volts to prevent the need to change the code.

    Arguments:
        activation (str):
            Kind of activation function. Possible values:
            'swish': Swish activation function.
            'ssp': Shifted softplus activation function.
        num_features (int):
            Dimensions of feature space.
        num_basis_functions (int):
            Number of radial basis functions.
        num_modules (int):
            Number of modules (iterations) for constructing atomic features.
        num_residual_electron (int):
            Number of residual blocks applied to features encoding the electronic 
            state.
        num_residual_pre (int):
            Number of residual blocks applied to atomic features in each module
            (before other transformations).
        num_residual_post (int):
            Number of residual blocks applied to atomic features after
            interaction with neighbouring atoms (per module).
        num_residual_pre_local_x (int):
            Number of residual blocks (per module) applied to atomic features in 
            local interaction.
        num_residual_pre_local_s (int):
            Number of residual blocks (per module) applied to s-type interaction features 
            in local interaction.
        num_residual_pre_local_p (int):
            Number of residual blocks (per module) applied to p-type interaction features 
            in local interaction.
        num_residual_pre_local_d (int):
            Number of residual blocks (per module) applied to d-type interaction features 
            in local interaction.
        num_residual_post (int):
            Number of residual blocks applied to atomic features in each module
            (after other transformations).
        num_residual_output (int):
            Number of residual blocks applied to atomic features in output
            branch (per module).
        basis_functions (str):
            Kind of radial basis functions. Possible values:
            'exp-bernstein': Exponential Bernstein polynomials.
            'exp-gaussian': Exponential Gaussian functions.
            'bernstein': Bernstein polynomials.
            'gaussian': Gaussian functions.
        exp_weighting (bool):
            Apply exponentially decaying weights to radial basis functions. Only
            used when 'basis_functions' argument is 'exp-bernstein' or
            'exp-gaussian'. Probably has almost no effect unless the weights of
            radial functions are regularized.
        cutoff (float):
            Cutoff radius for (neural network) interactions.
        lr_cutoff (float or None):
            Cutoff radius for long-range interactions (no cutoff is applied when
            this argument is None).
        use_zbl_repulsion (bool):
            If True, short-range repulsion inspired by the ZBL repulsive
            potential is applied to the energy prediction.
        use_electrostatics (bool):
            If True, point-charge electrostatics for the predicted atomic
            partial charges is applied to the energy prediction.
        use_d4_dispersion (bool):
            If True, Grimme's D4 dispersion correction is applied to the energy
            prediction.
        use_irreps (bool):
            For compatibility with older versions of the code.
        use_nonlinear_embedding (bool):
            For compatibility with older versions of the code.
        compute_d4_atomic (bool):
            If True, atomic polarizabilities and C6 coefficients in Grimme's D4
            dispersion correction are computed.
        module_keep_prob (float):
            Probability of keeping the last module during training. Module
            dropout can be a useful regularization that encourages
            hierarchicacally decaying contributions to the atomic features.
            Earlier modules are dropped with an automatically determined lower
            probability. Should be between 0.0 and 1.0.
        load_from (str or None):
            Load saved parameters from the given path instead of using random
            initialization (when 'load_from' is None).
        Zmax (int):
            Maximum nuclear charge +1 of atoms. The default is 87, so all
            elements up to Rn (Z=86) are supported. Can be kept at the default
            value (has minimal memory impact). Note that Grimme's D4 dispersion
            can only handle elements up to Rn (Z=86).
        zero_init (bool): Initialize parameters with zero whenever possible?
    """
    def __str__(self):
        return """
###############################################
# SpookyNet (Nat. Commun., 2021, 12(1): 7273) #
###############################################
"""

    def __init__(self):
        self.nuclear_embeddings: BaseAtomEmbedding = None
        self.charge_embeddings: BaseElectronEmbedding = None
        self.spin_embeddings: BaseElectronEmbedding = None
        self.radial_basis_functions: BaseRBF = None

        self.module = ModuleList(
            [
                InteractionModule(
                    num_features=self.num_features,
                    num_basis_functions=self.num_basis_functions,
                    num_residual_pre=self.num_residual_pre,
                    num_residual_local_x=self.num_residual_local_x,
                    num_residual_local_s=self.num_residual_local_s,
                    num_residual_local_p=self.num_residual_local_p,
                    num_residual_local_d=self.num_residual_local_d,
                    num_residual_local=self.num_residual_local,
                    num_residual_nonlocal_q=self.num_residual_nonlocal_q,
                    num_residual_nonlocal_k=self.num_residual_nonlocal_k,
                    num_residual_nonlocal_v=self.num_residual_nonlocal_v,
                    num_residual_post=self.num_residual_post,
                    num_residual_output=self.num_residual_output,
                    activation=self.activation,
                )   
                for i in range(self.num_modules)
            ]
        )

    @classmethod
    def build(cls, built_layers: Dict[str, nn.Module], **build_params: Dict[str, Any]) -> nn.Module:
        instance = cls(**build_params)
        for layer_name, layer in built_layers.items():
            if layer_name.endswith("Distance"):
                instance.distance_layer = layer
            elif layer_name.endswith("RBF"):
                instance.rbf_layer = layer
            elif layer_name.endswith("AtomEmbedding"):
                instance.embeddings = layer
        for layer_name in ["distance_layer", "rbf_layer", "embeddings"]:
            if getattr(instance, layer_name) is None:
                raise AttributeError(f"{layer_name} is not built")
        return instance