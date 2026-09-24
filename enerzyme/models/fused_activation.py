'''
Fused learnable Swish, alpha * x * sigmoid(beta * x) with feature-wise alpha/beta,
for inference on CUDA devices (gradient w.r.t. x only). The kernels are compiled
with NVRTC through torch.cuda.jiterator, whose launch overhead is lower than that
of several separate elementwise ops or a Triton kernel.
'''
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

try:
    _swish_fwd = torch.cuda.jiterator._create_jit_fn(
        """
        template <typename T> T swish_fwd(T x, T a, T b) {
            T s = T(1) / (T(1) + exp(-(b * x)));
            return a * x * s;
        }
        """
    )
    _swish_bwd = torch.cuda.jiterator._create_jit_fn(
        """
        template <typename T> T swish_bwd(T g, T x, T a, T b) {
            T s = T(1) / (T(1) + exp(-(b * x)));
            return g * a * s * (T(1) + b * x * (T(1) - s));
        }
        """
    )
    JITERATOR_AVAILABLE = True
except (AttributeError, RuntimeError):  # pragma: no cover
    JITERATOR_AVAILABLE = False


class FusedSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        ctx.save_for_backward(x, alpha, beta)
        return _swish_fwd(x, alpha, beta)

    @staticmethod
    @once_differentiable
    def backward(ctx, g: Tensor):
        x, alpha, beta = ctx.saved_tensors
        return _swish_bwd(g, x, alpha, beta), None, None


def can_fuse_swish(x: Tensor, alpha: Tensor, beta: Tensor) -> bool:
    return (
        JITERATOR_AVAILABLE and x.is_cuda and x.dim() > 0 and x.shape[-1] == alpha.numel()
        and alpha.dtype == x.dtype and beta.dtype == x.dtype
        and not alpha.requires_grad and not beta.requires_grad
    )


def fused_swish(x: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
    return FusedSwish.apply(x, alpha, beta)
