import math
import torch
import torch.nn.functional as F_

LOG2 = math.log(2.0)

def shifted_softplus(x):
    #return softplus(x) - np.log(2.0)
    return F_.softplus(x) - LOG2

def gather_nd(params, indices):
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