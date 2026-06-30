from typing import Literal, Dict, Any, Tuple
import torch
import torch.distributed as dist
from ..utils import logger

HYPER_PARAM_KEYS = {
    "Adam": {"learning_rate", "betas", "eps", "weight_decay", "amsgrad"},
    "AdamW": {"learning_rate", "betas", "eps", "weight_decay", "amsgrad"},
    "CoRe": {"learning_rate", "step_sizes", "etas", "betas", "eps", "weight_decay", "score_history", "frozen"},
    "Muon": {
        "learning_rate", "weight_decay", 
        "muon_learning_rate", "momentum", "muon_weight_decay", 
        "aux_learning_rate", "betas", "eps", "aux_weight_decay"
    },
}
MODEL_HEAD_NAMES = {
    "PhysNetCore": {"output_block"},
    "SchNetCore": {"lin1", "lin2"},
    "SpookyNetCore": {"output"},
    "AlphaNet": {"last_layer", "last_layer_quantum"},
    "MACECore": {"readouts"}
}


def get_optimizer_config(**params) -> Tuple[str, Dict[str, Any]]:
    '''
    Get the relevant arguments from the trainer for the optimizer name and hyperparameters, which will be used in the :doc:`get_optimizer <enerzyme.tasks.optimizer.get_optimizer>` function.

    Params:
    ----------
    **params: dict
        The configuration for the :doc:`Trainer <enerzyme.tasks.trainer.Trainer>` class.

    Returns:
    ----------
    name: str
        The name of the optimizer.

    hyper_params: dict
        The hyperparameters for the optimizer.
    '''
    hyper_params = {}
    if "Optimizer" in params:
        name = params["Optimizer"].get("name", "Adam")
        for key in HYPER_PARAM_KEYS[name]:
            if key in params["Optimizer"]:
                hyper_params[key] = params["Optimizer"][key]
    else:
        name = params.get("optimizer", "Adam")
        for key in HYPER_PARAM_KEYS[name]:
            if key in params:
                hyper_params[key] = params[key]
    return name, hyper_params


def get_optimizer(name: Literal["Adam", "AdamW", "CoRe", "Muon"], model: torch.nn.Module, hyper_params: Dict[str, Any]) -> torch.optim.Optimizer:
    '''
    Get an ready-to-use optimizer for a model given the optimizer string and hyperparameters.

    Args:
    ----------
    name: str
        The name of the optimizer. Now it supports the following optimizers:
        
        Adam: 
            Pytorch implementation of Adam.

        AdamW: 
            Pytorch implementation of AdamW.

        CoRe: 
            CoRe optimizer [1]_. It has been proven effective for lifelong learning of NNPs [2]_.

        Muon: 
            Muon optimizer [3]_ for hidden weights and auxiliary AdamW optimizer for the rest. It has been proven effective for fast training convergence and final accuracy of NNPs [4]_.

    model: torch.nn.Module
        The model to optimize.

        Now Muon optimizer only supports the following internal models: PhysNet, SpookyNet, AlphaNet, MACE, and SchNet.
        
    hyper_params: dict
        The hyperparameters for the optimizer, depending on the optimizer `name`.
        
        Adam:
            lr: float, default 1e-3
                Learning rate.
            betas: tuple, default (0.9, 0.999)
                Coefficients used for computing running averages of gradient and its square.
            eps: float, default 1e-6
                Term added to the denominator to improve numerical stability.
            weight_decay: float, default 0.0
                Weight decay (L2 penalty).
            amsgrad: bool, default True
                Whether to use the AMSGrad variant of Adam.
        AdamW:
            lr: float, default 1e-3
                Learning rate.
            betas: tuple, default (0.9, 0.999)
                Coefficients used for computing running averages of gradient and its square.
            eps: float, default 1e-6
                Term added to the denominator to improve numerical stability.
            weight_decay: float, default 0.0
                Weight decay (L2 penalty).
            amsgrad: bool, default True
                Whether to use the AMSGrad variant of Adam.
        CoRe:
            The default hyperparamters are from its application for NNP training [2]_.

            learning_rate: float, default 1e-3
                Learning rate.
            step_sizes: tuple, default (1e-6, 1.0)
                Step sizes for the optimizer.
            etas: tuple, default (0.5, 1.2)
                :math:`\\eta^-` and :math:`\\eta^+` in the paper [1]_.
            betas: tuple, default (0.45, 0.725, 500, 0.999)
                :math:`\\beta_1^{\\mathrm{a}}`, :math:`\\beta_1^{\\mathrm{b}}`, :math:`\\beta_1^{\\mathrm{c}}`, :math:`\\beta_2` in the paper [1]_.
            eps: float, default 1e-8
                Term added to the denominator to improve numerical stability.
            weight_decay: float, default 0.1
                Weight decay (L2 penalty).
            score_history: int, default 500
                :math:`t_{\\mathrm{hist}}` in the paper [1]_.
            frozen: float, default 0.1
                Fraction of parameters to compute the :math:`n_{\\mathrm{frozen}}` in the paper [1]_.
        Muon:
            The usage and hyperparameters are from https://github.com/KellerJordan/Muon?tab=readme-ov-file#usage

            muon_learning_rate: float, default 1e-2
                Learning rate of Muon optimizer. If not provided but with `learning_rate` provided, use the `learning rate`.

            muon_weight_decay: float, default 0.01
                Weight decay of the muon optimizer. If not provided but with `weight_decay` provided, use the `weight_decay`.

            momentum: float, default 0.95
                Momentum of the muon optimizer.

            aux_learning_rate: float, default 3e-4
                Learning rate of the auxiliary AdamW optimizer. If not provided but with `learning_rate` provided, use the `learning_rate`.

            aux_weight_decay: float, default 0.0
                Weight decay of the auxiliary AdamW optimizer. If not provided but with `weight_decay` provided, use the `weight_decay`.

            betas: tuple, default (0.9, 0.95)
                Coefficients of the auxiliary AdamW optimizer used for computing running averages of gradient and its square.

            eps: float, default 1e-10
                Term added to the denominator to improve numerical stability of the auxiliary AdamW optimizer.
                

    .. [1] Eckhoff, M.; Reiher, M. CoRe Optimizer: An All-in-One Solution for Machine Learning. Mach. Learn.: Sci. Technol. 2024, 5 (1), 015018. https://doi.org/10.1088/2632-2153/ad1f76.
    .. [2] Eckhoff, M.; Reiher, M. Lifelong Machine Learning Potentials. J. Chem. Theory Comput. 2023, 19 (12), 3509–3525. https://doi.org/10.1021/acs.jctc.3c00279.
    .. [3] Muon: An optimizer for hidden layers in neural networks | Keller Jordan blog. https://kellerjordan.github.io/posts/muon/ (accessed 2025-08-27).
    .. [4] Koker, T.; Smidt, T. Training a Foundation Model for Materials on a Budget. arXiv August 22, 2025. https://doi.org/10.48550/arXiv.2508.16067.
 
    Returns:
    ----------
    optimizer: torch.optim.Optimizer
        The optimizer for the model.
        
    Raises:
    ----------
    KeyError: 
        If the optimizer string is not supported.

    TypeError: 
        If the model is not supported by the optimizer.

    ImportError: 
        If the optimizer is not in Pytorch and the dependency is not installed.

    .. tip::
        To install the dependencies:

        CoRe: 
            :code:`pip install core-optimizer`

        Muon: 
            :code:`pip install muon-optimizer`

    '''
    if name == "Adam":
        from torch.optim import Adam
        lr = hyper_params.get("learning_rate", 1e-3)
        weight_decay = hyper_params.get("weight_decay", 0.0)
        betas = hyper_params.get("betas", (0.9, 0.999))
        eps = hyper_params.get("eps", 1e-6)
        amsgrad = hyper_params.get("amsgrad", True)
        optimizer = Adam(
            model.parameters(), 
            lr=lr, 
            betas=betas,
            eps=eps, 
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        logger.info(f"Using Adam optimizer with learning rate {lr}, weight decay {weight_decay}, betas {betas}, eps {eps}, and amsgrad {amsgrad}")
    elif name == "AdamW":
        from torch.optim import AdamW
        lr = hyper_params.get("learning_rate", 1e-3)
        weight_decay = hyper_params.get("weight_decay", 0.0)
        betas = hyper_params.get("betas", (0.9, 0.999))
        eps = hyper_params.get("eps", 1e-6)
        amsgrad = hyper_params.get("amsgrad", True)
        optimizer = AdamW(
            model.parameters(), 
            lr=lr, 
            betas=betas,
            eps=eps, 
            weight_decay=weight_decay, 
            amsgrad=amsgrad
        )
        logger.info(f"Using AdamW optimizer with learning rate {lr}, weight decay {weight_decay}, betas {betas}, eps {eps}, and amsgrad {amsgrad}")
    elif name == "CoRe":
        try:
            from core_optimizer import CoRe
        except ImportError:
            raise ImportError("CoRe optimizer is not installed. Please install it with `pip install core-optimizer`.")
        lr = hyper_params.get("learning_rate", 1e-3)
        step_sizes = hyper_params.get("step_sizes", (1e-6, 1.0))
        etas = hyper_params.get("etas", (0.5, 1.2))
        betas = hyper_params.get("betas", (0.45, 0.725, 500, 0.999))
        eps = hyper_params.get("eps", 1e-8)
        weight_decay = hyper_params.get("weight_decay", 0.1)
        score_history = hyper_params.get("score_history", 500)
        frozen = hyper_params.get("frozen", 0.1)
        optimizer = CoRe(
            model.parameters(), 
            lr=lr, 
            step_sizes=step_sizes,
            etas=etas,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            score_history=score_history,
            frozen=frozen
        )
        logger.info(f"Using CoRe optimizer with learning rate {lr}, step sizes {step_sizes}, etas {etas}, betas {betas}, eps {eps}, weight decay {weight_decay}, score history {score_history}, and frozen {frozen}")
    elif name == "Muon":
        try:
            if dist.is_initialized():
                from muon import MuonWithAuxAdam as Muon
            else:
                from muon import SingleDeviceMuonWithAuxAdam as Muon
        except ImportError:
            raise ImportError("Muon optimizer is not installed. Please install it with `pip install muon-optimizer`.")
        
        # check if the model is supported
        core_name = model.__class__.__name__
        if core_name not in MODEL_HEAD_NAMES:
            raise TypeError(f"Muon optimizer is not supported for {core_name}.")
        
        # get the param groups
        head_names = MODEL_HEAD_NAMES[core_name]
        hidden_weights = []
        hidden_gains_biases = []
        nonhidden_params = []
        for name, param in model.named_parameters():
            name_prefix = name.split(".")[0]
            if name_prefix in ["pre_sequence", "post_sequence"]:
                nonhidden_params.append(param)
            elif name_prefix in head_names:
                nonhidden_params.append(param)
            else:
                if param.dim() >= 2:
                    hidden_weights.append(param)
                else:
                    hidden_gains_biases.append(param)
        muon_lr = hyper_params.get("muon_learning_rate", 
            hyper_params.get("learning_rate", 1e-2)
        )
        muon_weight_decay = hyper_params.get("muon_weight_decay", 
            hyper_params.get("weight_decay", 0.01)
        )
        aux_lr = hyper_params.get("aux_learning_rate", 
            hyper_params.get("learning_rate", 3e-4)
        )
        aux_weight_decay = hyper_params.get("aux_weight_decay", 
            hyper_params.get("weight_decay", 0.)
        )
        momentum = hyper_params.get("momentum", 0.95)
        betas = hyper_params.get("betas", (0.9, 0.95))
        eps = hyper_params.get("eps", 1e-10)
        param_groups = [
            {
                "params": hidden_weights, "use_muon": True,
                "lr": muon_lr,
                "weight_decay": muon_weight_decay,
                "momentum": momentum
            },
            {
                "params": hidden_gains_biases + nonhidden_params, "use_muon": False,
                "lr": aux_lr,
                "weight_decay": aux_weight_decay,
                "betas": betas,
                "eps": eps
            }
        ]
        logger.info(f"Using Muon optimizer with muon learning rate {muon_lr}, muon weight decay {muon_weight_decay}, aux learning rate {aux_lr}, aux weight decay {aux_weight_decay}, momentum {momentum}, betas {betas}, eps {eps}")
        optimizer = Muon(param_groups)
    else:
        raise ValueError(f"Optimizer {name} not supported.")
    return optimizer