from typing import Union, Optional
import numpy as np
import torch
from torch import Tensor


USE_SEGMENT_COO = False


if USE_SEGMENT_COO:
    from torch_scatter import segment_sum_coo
else:
    from torch_scatter import scatter_sum
    def segment_sum_coo(src: Tensor, idx: Tensor, dim_size: Optional[int]=None) -> Tensor:
        return scatter_sum(src, idx, dim=idx.dim() - 1, dim_size=dim_size)


def softplus_inverse(x: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus). 
    
    The indirect implementation is for numerical stability.
    """
    if not isinstance(x, Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


def gather_nd(params: Tensor, indices: Tensor) -> Tensor:
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    implemented at https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/37
    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()
