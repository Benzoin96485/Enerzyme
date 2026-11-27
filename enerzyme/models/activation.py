import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Literal, Dict, Union, Callable
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter


class SmoothLeakyReLU(Module):
    """Smooth leaky ReLU used for Equiformer attention scores.

    .. math::

        f(x) = \\frac{1+\\alpha}{2} x
             + \\frac{1-\\alpha}{2} x \\,(2\\sigma(x)-1)
    """

    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.alpha = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        # f(x) = alpha*x + (1-alpha)*x*sigmoid(x).
        return torch.addcmul(self.alpha * x, x, torch.sigmoid(x), value=1 - self.alpha)

    def extra_repr(self) -> str:
        return "negative_slope={}".format(self.alpha)


class ScaledSigmoid(Module):
    """Sigmoid with e3nn ``Activation`` variance-preserving scale (TACE / TECE)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.8467055342154763

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * self.scale_factor

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"


class ScaledSiLU(Module):
    """SiLU with e3nn ``Activation`` variance-preserving scale (TACE / TECE)."""

    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1.6791767923989418

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x) * self.scale_factor

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"


def get_scaled_activation(name: str) -> Module:
    """Return fixed-scale ``sigmoid`` / ``silu`` used by SO(2) gates."""
    if name == "sigmoid":
        return ScaledSigmoid()
    if name == "silu":
        return ScaledSiLU()
    raise ValueError(f"Unknown scaled activation={name!r}; expected 'sigmoid' or 'silu'")


class SwiGLU(Module):
    """SwiGLU over the last dim: ``value * silu(gate)`` (DPA4 / SeZM order).

    Distinct from :class:`~enerzyme.models.so3.activation_v3.SwiGLU`, which
    applies ``silu`` to the first half (EquiformerV3 convention).
    """

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even last dimension")
        value, gate = x.chunk(2, dim=-1)
        return value * F.silu(gate)


class BaseScaledTemperedActivation(ABC, Module):
    def __init__(self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.0, learnable: bool=False) -> None:
        super().__init__()
        if float(initial_alpha) == 1.0 and float(initial_beta) == 1.0 and not learnable:
            self.simple = True
            self.alpha = 1.0
            self.beta = 1.0
            self._activation_fn = self.simple_activation_fn
        else:
            self.simple = False
            self.register_parameter("alpha", Parameter(torch.full((dim_feature,), initial_alpha), requires_grad=learnable))
            self.register_parameter("beta", Parameter(torch.full((dim_feature,), initial_beta), requires_grad=learnable))
            self._activation_fn = self.activation_fn

    @abstractmethod
    def simple_activation_fn(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def activation_fn(self, x: Tensor) -> Tensor:
        ...

    def forward(self, x: Tensor) -> Tensor:
        return self._activation_fn(x)


class ShiftedSoftplus(BaseScaledTemperedActivation):
    def __init__(self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.0, learnable: bool=False) -> None:
        """
        Shifted softplus activation function with learnable feature-wise parameters:
        f(x) = alpha/beta * (softplus(beta*x) - log(2))
        softplus(x) = log(exp(x) + 1)
        For beta -> 0  : f(x) -> 0.5*alpha*x
        For beta -> inf: f(x) -> max(0, alpha*x)

        Arguments:
            num_features (int):
                Dimensions of feature space.
            initial_alpha (float):
                Initial "scale" alpha of the softplus function.
            initial_beta (float):
                Initial "temperature" beta of the softplus function.
        """
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)
        self.log2 = math.log(2.0)
    
    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.log2
    
    def activation_fn(self, x: Tensor) -> Tensor:
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self.log2) / self.beta,
            0.5 * x
        )


class ReluLike(BaseScaledTemperedActivation):
    """REANN repository activation ``alpha * SiLU(beta * x)``.

    Zhang, Xia, and Jiang write ``eta * x * sigmoid(beta * x)`` (J. Chem. Phys.
    2022, Eq. 7). The released ``Relu_like`` module is ``alpha * SiLU(beta * x)``,
    the same family with an extra factor of ``beta`` inside the scale. ``alpha``
    and ``beta`` are learnable per channel. ``beta -> 0`` is a linear map and
    ``beta -> inf`` approaches a scaled ReLU.
    """

    def __init__(
        self,
        dim_feature: int = 1,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)

    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return F.silu(x)

    def activation_fn(self, x: Tensor) -> Tensor:
        return self.alpha * F.silu(self.beta * x)


class TanhLike(BaseScaledTemperedActivation):
    """REANN hyperbolic activation ``alpha * x / sqrt(1 + (beta * x)^2)``.

    This is Eq. 7 of Zhang, Xia, and Jiang (J. Chem. Phys. 2022) and the
    released ``Tanh_like`` module. ``beta -> 0`` recovers ``alpha * x``.
    Large ``|x|`` saturates at ``alpha / |beta|``.
    """

    def __init__(
        self,
        dim_feature: int = 1,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        learnable: bool = True,
    ) -> None:
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)

    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return x / torch.sqrt(1.0 + x.square())

    def activation_fn(self, x: Tensor) -> Tensor:
        return self.alpha * x / torch.sqrt(1.0 + (self.beta * x).square())


class Swish(BaseScaledTemperedActivation):
    """
    Swish activation function with learnable feature-wise parameters:
    f(x) = alpha*x * sigmoid(beta*x)
    sigmoid(x) = 1/(1 + exp(-x))
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the "linear component".
        initial_beta (float):
            Initial "temperature" of the "sigmoid component". The default value
            of 1.702 has the effect of initializing swish to an approximation
            of the Gaussian Error Linear Unit (GELU) activation function from
            Hendrycks, Dan, and Gimpel, Kevin. "Gaussian error linear units
            (GELUs)."
    """

    def __init__(
        self, dim_feature: int=1, initial_alpha: float=1.0, initial_beta: float=1.702, learnable: bool=True
    ) -> None:
        """ Initializes the Swish class. """
        super().__init__(dim_feature, initial_alpha, initial_beta, learnable)

    def simple_activation_fn(self, x: Tensor) -> Tensor:
        return F.silu(x)

    def activation_fn(self, x: Tensor) -> Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, num_features]):
                Input features.

        Returns:
            y (FloatTensor [:, num_features]):
                Activated features.
        """
        return self.alpha * x * torch.sigmoid(self.beta * x)


ACTIVATION_REGISTER = {
    "shifted_softplus": ShiftedSoftplus,
    "swish": Swish,
    "relu_like": ReluLike,
    "tanh_like": TanhLike,
}
ACTIVATION_KEY_TYPE = Literal["shifted_softplus", "swish", "relu_like", "tanh_like"]
ACTIVATION_PARAM_TYPE = Dict[Literal["dim_feature", "initial_alpha", "initial_beta", "learnable"], Union[int, float, bool]]


def get_activation_fn(
    key: ACTIVATION_KEY_TYPE, 
    activation_params: ACTIVATION_PARAM_TYPE=dict()
) -> BaseScaledTemperedActivation:
    return ACTIVATION_REGISTER[key](**activation_params)


POSITIVE_ACTIVATION_REGISTER = {
    "softplus": F.softplus,
    "square": torch.square,
    "exp": torch.exp,
}
POSITIVE_ACTIVATION_KEY_TYPE = Literal["softplus", "square", "exp"]


def get_positive_activation_fn(
    key: POSITIVE_ACTIVATION_KEY_TYPE,
) -> Callable[[Tensor], Tensor]:
    return POSITIVE_ACTIVATION_REGISTER[key]


def shifted_softplus(x: Tensor) -> Tensor:
    """``softplus(x) - log(2)``. Zero at the origin, used by e3nn gates."""
    return F.softplus(x) - math.log(2.0)


def silut_constants(threshold: float) -> tuple:
    """Slope and offset that match SiLU's value and first derivative at ``threshold``.

    The tail is ``tanh(a (x - t)) + b``. Its second derivative is 0 at the
    join, while SiLU's is not, so the blend is C1. That is the DeePMD-kit
    SiLUT (threshold 3). The DPA3 paper asks for C2; this one-parameter tanh
    tail cannot supply it.
    """
    if float(threshold) <= 0.0:
        raise ValueError(f"SiLUT threshold must be positive, got {threshold}")
    sigmoid_t = 1.0 / (1.0 + math.exp(-float(threshold)))
    slope = sigmoid_t + float(threshold) * sigmoid_t * (1.0 - sigmoid_t)
    const = float(threshold) * sigmoid_t
    return slope, const


def silut(x: Tensor, threshold: float = 3.0) -> Tensor:
    """SiLU for ``x < threshold``, C1 tanh tail above (DPA3 training activation)."""
    slope, const = silut_constants(threshold)
    tail = torch.tanh(slope * (x - float(threshold))) + const
    return torch.where(x >= float(threshold), tail, F.silu(x))


def _silut_default(x: Tensor) -> Tensor:
    return silut(x, 3.0)


SCALAR_ACTIVATION_REGISTER: Dict[str, Callable[[Tensor], Tensor]] = {
    "silu": F.silu,
    "relu": F.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "ssp": shifted_softplus,
    "silut": _silut_default,
    "abs": torch.abs,
}


def resolve_scalar_activation(name: str) -> Callable[[Tensor], Tensor]:
    """Callable nonlinearity for an e3nn gate or radial MLP.

    Distinct from :func:`get_activation_fn`, which returns a tempered module
    (``swish`` / ``shifted_softplus`` with per-channel scale). ``silut`` uses
    threshold 3. ``silut:2.5`` selects another threshold.
    """
    key = name.lower()
    if key.startswith("silut:"):
        return partial(silut, threshold=float(key.split(":", 1)[1]))
    if key not in SCALAR_ACTIVATION_REGISTER:
        known = ", ".join(sorted(SCALAR_ACTIVATION_REGISTER))
        raise ValueError(f"Unknown scalar activation {name!r}; expected one of {known}")
    return SCALAR_ACTIVATION_REGISTER[key]
